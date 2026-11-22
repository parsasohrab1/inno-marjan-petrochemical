"""Phase 4 tests: secrets loader, backup, restore.

Covers the new Phase 4 modules:
- mrto/secrets.py: backend chain (env → file → Vault → SOPS), caching,
  require_secret raising on missing.
- scripts/backup.py: SQLite online backup, models tar, rotation,
  manifest, --dry-run.
- scripts/restore.py: SQLite restore with pre-restore snapshot,
  --latest, confirmation prompt.
"""
import os
import sys
import json
import sqlite3
import tarfile
import subprocess
from pathlib import Path

import pytest

# Make scripts/ importable.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from mrto import secrets as _sec
import backup as _bk
import restore as _rs


@pytest.fixture(autouse=True)
def _reset_secret_cache():
    _sec.reset_cache()
    yield
    _sec.reset_cache()


# ── secrets.py: env backend ─────────────────────────────────────────
def test_secret_from_env(monkeypatch):
    monkeypatch.setenv("MRTO_TEST_SECRET", "from-env")
    assert _sec.get_secret("MRTO_TEST_SECRET") == "from-env"


def test_secret_cached(monkeypatch):
    monkeypatch.setenv("MRTO_TEST_CACHE", "v1")
    assert _sec.get_secret("MRTO_TEST_CACHE") == "v1"
    monkeypatch.setenv("MRTO_TEST_CACHE", "v2")
    # Second call returns the cached value, not the new env value.
    assert _sec.get_secret("MRTO_TEST_CACHE") == "v1"


def test_secret_default_when_missing(monkeypatch):
    monkeypatch.delenv("MRTO_NONEXISTENT", raising=False)
    assert _sec.get_secret("MRTO_NONEXISTENT", "default-x") == "default-x"


def test_require_secret_raises_when_missing(monkeypatch):
    monkeypatch.delenv("MRTO_REQUIRED_BUT_MISSING", raising=False)
    with pytest.raises(RuntimeError, match="MRTO_REQUIRED_BUT_MISSING"):
        _sec.require_secret("MRTO_REQUIRED_BUT_MISSING")


def test_require_secret_returns_value(monkeypatch):
    monkeypatch.setenv("MRTO_REQUIRED_PRESENT", "yes")
    assert _sec.require_secret("MRTO_REQUIRED_PRESENT") == "yes"


# ── secrets.py: file backend ────────────────────────────────────────
def test_secret_from_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "jwt.txt"
    secret_file.write_text("from-file-content\n")
    monkeypatch.setenv("MRTO_TEST_FILE_SECRET_FILE", str(secret_file))
    # Make sure the env var itself is NOT set, so file backend wins.
    monkeypatch.delenv("MRTO_TEST_FILE_SECRET", raising=False)
    assert _sec.get_secret("MRTO_TEST_FILE_SECRET") == "from-file-content"


def test_secret_env_takes_priority_over_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "jwt.txt"
    secret_file.write_text("from-file-content\n")
    monkeypatch.setenv("MRTO_TEST_PRIORITY_FILE", str(secret_file))
    monkeypatch.setenv("MRTO_TEST_PRIORITY", "from-env")
    assert _sec.get_secret("MRTO_TEST_PRIORITY") == "from-env"


def test_secret_file_missing_falls_through(monkeypatch):
    monkeypatch.setenv("MRTO_TEST_MISSING_FILE_FILE",
                       "/nonexistent/path/secret.txt")
    monkeypatch.delenv("MRTO_TEST_MISSING_FILE", raising=False)
    assert _sec.get_secret("MRTO_TEST_MISSING_FILE", "fallback") == "fallback"


# ── secrets.py: vault/sops gracefully degrade ──────────────────────
def test_vault_backend_disabled_without_addr(monkeypatch):
    monkeypatch.delenv("MRTO_VAULT_ADDR", raising=False)
    # Even with a fake name, Vault backend returns None — chain falls through.
    monkeypatch.delenv("MRTO_TEST_VAULT_DEGRADES", raising=False)
    assert _sec.get_secret("MRTO_TEST_VAULT_DEGRADES", "fallback") == "fallback"


def test_sops_backend_disabled_without_sops(monkeypatch, tmp_path):
    monkeypatch.setenv("MRTO_SOPS_FILE", str(tmp_path / "nonexistent.sops.yaml"))
    monkeypatch.delenv("MRTO_TEST_SOPS_DEGRADES", raising=False)
    assert _sec.get_secret("MRTO_TEST_SOPS_DEGRADES", "fallback") == "fallback"


# ── backup.py ────────────────────────────────────────────────────────
def _make_sqlite_db(path: Path, n_tubes: int = 5) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS tubes (tube_id TEXT PRIMARY KEY);"
    )
    for i in range(n_tubes):
        conn.execute("INSERT OR REPLACE INTO tubes VALUES (?)", (f"T{i:03d}",))
    conn.commit()
    conn.close()


def test_backup_sqlite_online(tmp_path, monkeypatch):
    db = tmp_path / "src.sqlite3"
    _make_sqlite_db(db, n_tubes=7)
    monkeypatch.setenv("MRTO_DB_PATH", str(db))
    monkeypatch.setenv("MRTO_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    # Run backup.py as a subprocess (it's a CLI script).
    out_dir = tmp_path / "backups"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "backup.py"),
         "--out", str(out_dir), "--retain", "3"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    # A DB backup file exists.
    db_backups = list(out_dir.glob("mrto_db_*.sqlite3"))
    assert len(db_backups) == 1, f"expected 1 db backup, got {db_backups}"
    # Verify the backup is readable and has the rows.
    conn = sqlite3.connect(str(db_backups[0]))
    n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
    conn.close()
    assert n == 7
    # A manifest exists.
    manifests = list(out_dir.glob("mrto_backup_*.manifest"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["files"][0]["kind"] == "db"
    assert manifest["files"][0]["ok"] is True


def test_backup_dry_run_writes_nothing(tmp_path, monkeypatch):
    db = tmp_path / "src.sqlite3"
    _make_sqlite_db(db)
    monkeypatch.setenv("MRTO_DB_PATH", str(db))
    monkeypatch.setenv("MRTO_MODELS_DIR", str(tmp_path / "models"))
    out_dir = tmp_path / "backups"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "backup.py"),
         "--out", str(out_dir), "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_backup_retention_deletes_old_sets(tmp_path, monkeypatch):
    """Creating 5 backups with --retain 3 keeps only the newest 3 sets."""
    db = tmp_path / "src.sqlite3"
    _make_sqlite_db(db)
    monkeypatch.setenv("MRTO_DB_PATH", str(db))
    monkeypatch.setenv("MRTO_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    out_dir = tmp_path / "backups"
    for _ in range(5):
        # Force distinct timestamps by sleeping briefly.
        import time; time.sleep(1.05)
        subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "backup.py"),
             "--out", str(out_dir), "--retain", "3"],
            capture_output=True, text=True, check=True,
        )
    db_backups = list(out_dir.glob("mrto_db_*.sqlite3"))
    assert len(db_backups) == 3, (
        f"expected 3 retained, got {len(db_backups)}: {[f.name for f in db_backups]}"
    )


# ── restore.py ───────────────────────────────────────────────────────
def test_restore_sqlite_with_prerestore_snapshot(tmp_path, monkeypatch):
    # Original DB with 5 rows.
    live_db = tmp_path / "live.sqlite3"
    _make_sqlite_db(live_db, n_tubes=5)
    # Backup it.
    backup_file = tmp_path / "backup.sqlite3"
    _bk.backup_sqlite(str(live_db), backup_file)
    # Now corrupt the live DB to 0 rows.
    conn = sqlite3.connect(str(live_db))
    conn.execute("DELETE FROM tubes")
    conn.commit(); conn.close()
    # Restore the backup.
    monkeypatch.setenv("MRTO_DB_PATH", str(live_db))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "restore.py"),
         "--db", str(backup_file), "--yes"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    # Live DB now has 5 rows again.
    conn = sqlite3.connect(str(live_db))
    n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
    conn.close()
    assert n == 5
    # A pre-restore snapshot exists (the corrupted 0-row version).
    snaps = list(tmp_path.glob("live.sqlite3.prerestore.*"))
    assert len(snaps) == 1


def test_restore_declines_without_yes(tmp_path, monkeypatch):
    """Without --yes, restore refuses to overwrite (no stdin → EOF → decline)."""
    live_db = tmp_path / "live.sqlite3"
    _make_sqlite_db(live_db, n_tubes=3)
    backup_file = tmp_path / "backup.sqlite3"
    _bk.backup_sqlite(str(live_db), backup_file)
    monkeypatch.setenv("MRTO_DB_PATH", str(live_db))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "restore.py"),
         "--db", str(backup_file)],
        capture_output=True, text=True,
        stdin=subprocess.DEVNULL,
    )
    # Exit 3 = user declined.
    assert r.returncode == 3, r.stderr + r.stdout
    # Live DB untouched.
    conn = sqlite3.connect(str(live_db))
    n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
    conn.close()
    assert n == 3


def test_restore_latest_picks_newest(tmp_path, monkeypatch):
    """--latest restores the newest backup in MRTO_BACKUP_DIR."""
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    # Create two backups with different content.
    db_v1 = tmp_path / "v1.sqlite3"; _make_sqlite_db(db_v1, n_tubes=2)
    db_v2 = tmp_path / "v2.sqlite3"; _make_sqlite_db(db_v2, n_tubes=9)
    import time
    _bk.backup_sqlite(str(db_v1), backup_dir / "mrto_db_20240101_000000.sqlite3")
    time.sleep(0.01)
    _bk.backup_sqlite(str(db_v2), backup_dir / "mrto_db_20240102_000000.sqlite3")
    # Live DB to overwrite.
    live_db = tmp_path / "live.sqlite3"; _make_sqlite_db(live_db, n_tubes=0)
    monkeypatch.setenv("MRTO_DB_PATH", str(live_db))
    monkeypatch.setenv("MRTO_BACKUP_DIR", str(backup_dir))
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "restore.py"),
         "--latest", "--yes"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    conn = sqlite3.connect(str(live_db))
    n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
    conn.close()
    assert n == 9  # the newer backup won
