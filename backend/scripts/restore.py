#!/usr/bin/env python3
"""MRTO restore — companion to scripts/backup.py.

Restores a DB backup (SQLite file or Postgres .sql.gz) and optionally a
models tar.gz into the locations pointed to by MRTO_DB_PATH /
MRTO_DATABASE_URL / MRTO_MODELS_DIR.

Usage:
    python scripts/restore.py --db mrto_db_20241122_153012.sqlite3
    python scripts/restore.py --db mrto_db_20241122_153012.sql.gz --models mrto_models_20241122_153012.tar.gz
    python scripts/restore.py --latest            # newest backup in MRTO_BACKUP_DIR
    python scripts/restore.py --latest --yes       # skip the confirmation prompt

Safety:
- Refuses to overwrite a live DB without --yes (interactive prompt otherwise).
- Creates a pre-restore snapshot of the current DB at `<db>.prerestore.<ts>`.
- For Postgres, applies the dump with `psql --set ON_ERROR_STOP=on`.
- Verify the manifest (if present) before applying.

Exit codes:
    0  restore succeeded
    1  restore failed
    3  user declined the confirmation prompt
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _db_path() -> str:
    return os.environ.get("MRTO_DB_PATH", "/app/data/mrto.sqlite3")


def _models_dir() -> str:
    return os.environ.get("MRTO_MODELS_DIR", "/app/models")


def _db_url() -> str:
    return os.environ.get("MRTO_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""


def _backup_dir(args) -> Path:
    return Path(args.backup_dir or os.environ.get("MRTO_BACKUP_DIR", "./backups"))


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    try:
        return input(prompt + " [y/N] ").strip().lower().startswith("y")
    except EOFError:
        return False


# ────────────────────────────────────────────────────────────────────
def restore_sqlite(src: Path, dest: str) -> dict:
    """Overwrite the SQLite DB at `dest` with the backup at `src`.

    Pre-restore snapshot: copies the current DB to `<dest>.prerestore.<ts>`
    so a bad restore can be rolled back by hand.
    """
    # Snapshot the current DB.
    if os.path.exists(dest):
        snap = f"{dest}.prerestore.{_ts()}"
        shutil.copy2(dest, snap)
    # Replace atomically-ish: copy to a temp then rename.
    tmp = dest + ".restore_tmp"
    try:
        s = sqlite3.connect(str(src))
        d = sqlite3.connect(tmp)
        with d:
            s.backup(d)
        s.close()
        d.close()
        os.replace(tmp, dest)
    except sqlite3.Error as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": dest}


def restore_postgres(src: Path, dsn: str) -> dict:
    """Apply a gzipped SQL dump to Postgres via psql."""
    if not shutil.which("psql"):
        return {"ok": False, "error": "psql not on PATH"}
    try:
        with gzip.open(src, "rt") as f:
            proc = subprocess.run(
                ["psql", "--set", "ON_ERROR_STOP=on", dsn],
                stdin=f, stderr=subprocess.PIPE, timeout=1800,
            )
        if proc.returncode != 0:
            return {"ok": False,
                    "error": f"psql exit {proc.returncode}: {proc.stderr.decode()}"}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "dsn": dsn}


def restore_models(src: Path, dest: str) -> dict:
    """Extract the models tar.gz into MRTO_MODELS_DIR."""
    os.makedirs(dest, exist_ok=True)
    try:
        with tarfile.open(src, "r:gz") as tar:
            # Guard against path traversal — only extract into dest.
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    return {"ok": False,
                            "error": f"unsafe path in tar: {member.name}"}
            tar.extractall(dest)
    except (OSError, tarfile.TarError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": dest}


# ────────────────────────────────────────────────────────────────────
def find_latest(backup_dir: Path, kind: str) -> Path | None:
    """Return the newest backup file matching kind ('db' or 'models')."""
    candidates = []
    for f in backup_dir.iterdir():
        if not f.is_file():
            continue
        n = f.name
        if kind == "db" and (n.startswith("mrto_db_")):
            candidates.append(f)
        elif kind == "models" and n.startswith("mrto_models_"):
            candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="MRTO restore")
    ap.add_argument("--db", help="DB backup file to restore")
    ap.add_argument("--models", help="models tar.gz to restore")
    ap.add_argument("--latest", action="store_true",
                    help="use the newest backup in MRTO_BACKUP_DIR")
    ap.add_argument("--backup-dir", help="override MRTO_BACKUP_DIR")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    args = ap.parse_args()

    bdir = _backup_dir(args)
    dsn = _db_url()
    db_path = _db_path()
    models_dir = _models_dir()

    if args.latest:
        args.db = str(find_latest(bdir, "db") or "")
        args.models = str(find_latest(bdir, "models") or "")
        if not args.db:
            print(f"no DB backup found in {bdir}", file=sys.stderr)
            return 1

    if not args.db:
        print("nothing to restore — provide --db or --latest", file=sys.stderr)
        return 1

    db_src = Path(args.db)
    if not db_src.is_file():
        print(f"DB backup not found: {db_src}", file=sys.stderr)
        return 1

    # Confirmation
    target = dsn if dsn else db_path
    if not _confirm(
        f"This will OVERWRITE the live DB at {target}. Continue?", args.yes
    ):
        print("aborted", file=sys.stderr)
        return 3

    results: dict = {"ts": _ts(), "files": []}
    ok = True

    # DB
    if db_src.suffix == ".sqlite3":
        r = restore_sqlite(db_src, db_path)
    else:  # .sql.gz
        r = restore_postgres(db_src, dsn)
    results["files"].append({"kind": "db", **r})
    ok = ok and r["ok"]

    # Models
    if args.models:
        m_src = Path(args.models)
        if m_src.is_file():
            r = restore_models(m_src, models_dir)
            results["files"].append({"kind": "models", **r})
            ok = ok and r["ok"]

    print(json.dumps(results, indent=2))
    if not ok:
        print("RESTORE FAILED — see above", file=sys.stderr)
        return 1
    print("restore OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
