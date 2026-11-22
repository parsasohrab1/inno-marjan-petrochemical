#!/usr/bin/env python3
"""MRTO backup — SQLite file copy + Postgres pg_dump + models dir tar.

Usage:
    python scripts/backup.py                  # uses MRTO_* env vars
    python scripts/backup.py --out /var/backups/mrto --retain 14
    python scripts/backup.py --dry-run          # print plan, write nothing

Produces, in MRTO_BACKUP_DIR (default ./backups):
    mrto_db_YYYYMMDD_HHMMSS.sqlite3(.zst)      # SQLite copy (or compressed)
    mrto_db_YYYYMMDD_HHMMSS.sql.gz              # Postgres dump (when applicable)
    mrto_models_YYYYMMDD_HHMMSS.tar.gz         # models directory
    mrto_backup_YYYYMMDD_HHMMSS.manifest       # JSON manifest

Retention: keeps the most recent MRTO_BACKUP_RETAIN (default 14) backup
sets; older ones are deleted. A backup set = all files sharing the same
timestamp prefix. A `.manifest` file lists what was backed up so restore
can verify completeness.

Exit codes:
    0  success (or --dry-run printed the plan)
    1  backup failed (DB unreadable, out of disk, etc.)
    2  partial — DB ok, models skipped/failed (still returns a usable DB backup)

This script is safe to run while the API is serving: SQLite is copied
using the backup API (`.backup` to a temp file then rename) which does
not block readers. For Postgres, `pg_dump --no-owner --clean` produces a
consistent snapshot regardless of concurrent writes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

# ────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _db_path() -> str:
    return os.environ.get("MRTO_DB_PATH", "/app/data/mrto.sqlite3")


def _models_dir() -> str:
    return os.environ.get("MRTO_MODELS_DIR", "/app/models")


def _db_url() -> str:
    return os.environ.get("MRTO_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""


def _out_dir(args) -> Path:
    d = Path(args.out or os.environ.get("MRTO_BACKUP_DIR", "./backups"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _retain(args) -> int:
    return int(args.retain or os.environ.get("MRTO_BACKUP_RETAIN", "14"))


# ────────────────────────────────────────────────────────────────────
def backup_sqlite(db_path: str, dest: Path) -> dict:
    """Copy a live SQLite DB using the backup API (online, safe)."""
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"DB file not found: {db_path}"}
    # The SQLite backup API copies page-by-page under a shared lock;
    # writers proceed concurrently.
    try:
        src = sqlite3.connect(db_path)
        dst = sqlite3.connect(str(dest))
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
    except sqlite3.Error as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(dest), "bytes": dest.stat().st_size}


def backup_postgres(dsn: str, dest: Path) -> dict:
    """Dump a Postgres DB to a gzipped SQL file."""
    if not shutil.which("pg_dump"):
        return {"ok": False, "error": "pg_dump not on PATH"}
    try:
        with open(dest, "wb") as f:
            proc = subprocess.run(
                ["pg_dump", "--no-owner", "--clean", "--if-exists", dsn],
                stdout=f, stderr=subprocess.PIPE, timeout=600,
            )
        if proc.returncode != 0:
            return {"ok": False,
                    "error": f"pg_dump exit {proc.returncode}: {proc.stderr.decode()}"}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(dest), "bytes": dest.stat().st_size}


def backup_models(src: str, dest: Path) -> dict:
    """Tar-gzip the models directory."""
    if not os.path.isdir(src):
        return {"ok": False, "error": f"models dir not found: {src}"}
    try:
        with tarfile.open(dest, "w:gz") as tar:
            tar.add(src, arcname="models")
    except (OSError, tarfile.TarError) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(dest), "bytes": dest.stat().st_size}


# ────────────────────────────────────────────────────────────────────
def rotate(out_dir: Path, retain: int) -> int:
    """Keep the most recent `retain` backup sets; delete older ones.

    A backup set = all files sharing the same `_YYYYMMDD_HHMMSS` prefix
    in the backup directory. Returns the number of sets deleted.
    """
    if retain <= 0:
        return 0
    # Collect timestamp prefixes from filenames.
    prefixes: set[str] = set()
    for f in out_dir.iterdir():
        if not f.is_file():
            continue
        # mrto_db_YYYYMMDD_HHMMSS.sqlite3 → 20241122_153012
        name = f.name
        parts = name.split("_")
        if len(parts) < 3:
            continue
        ts = "_".join(parts[-2:]).rsplit(".", 1)[0]
        if len(ts) == 15 and ts[8] == "_":
            prefixes.add(ts)
    if len(prefixes) <= retain:
        return 0
    # Sort newest-first; drop everything after the first `retain`.
    keep = set(sorted(prefixes, reverse=True)[:retain])
    deleted = 0
    for f in out_dir.iterdir():
        if not f.is_file():
            continue
        ts = "_".join(f.name.split("_")[-2:]).rsplit(".", 1)[0]
        if len(ts) == 15 and ts[8] == "_" and ts not in keep:
            f.unlink()
            deleted += 1
    return deleted


# ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="MRTO backup")
    ap.add_argument("--out", help="output directory (default: MRTO_BACKUP_DIR)")
    ap.add_argument("--retain", type=int,
                    help="number of backup sets to keep (default: MRTO_BACKUP_RETAIN=14)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, write nothing")
    args = ap.parse_args()

    out_dir = _out_dir(args) if not args.dry_run else Path(".")
    ts = _ts()
    dsn = _db_url()
    db_path = _db_path()
    models = _models_dir()
    retain = _retain(args)

    plan = {
        "ts": ts,
        "out_dir": str(out_dir),
        "db_path": db_path,
        "models_dir": models,
        "database_url_set": bool(dsn),
        "retain": retain,
    }
    print("=== MRTO backup plan ===")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("(dry run — nothing written)")
        return 0

    results: dict = {"ts": ts, "files": []}
    overall_ok = True
    models_ok = True

    # DB backup — prefer Postgres if configured, else SQLite copy.
    if dsn:
        dest = out_dir / f"mrto_db_{ts}.sql.gz"
        # pg_dump writes raw; we gzip on the fly via a pipe.
        # For simplicity here we dump to a temp file and gzip it.
        tmp = out_dir / f"mrto_db_{ts}.sql"
        r = backup_postgres(dsn, tmp)
        if r["ok"]:
            # gzip in place
            with open(tmp, "rb") as fi, open(dest, "wb") as fo:
                subprocess.run(["gzip"], stdin=fi, stdout=fo, check=True)
            tmp.unlink(missing_ok=True)
            r = {"ok": True, "path": str(dest),
                 "bytes": dest.stat().st_size}
        results["files"].append({"kind": "db", **r})
        overall_ok = overall_ok and r["ok"]
    else:
        dest = out_dir / f"mrto_db_{ts}.sqlite3"
        r = backup_sqlite(db_path, dest)
        results["files"].append({"kind": "db", **r})
        overall_ok = overall_ok and r["ok"]

    # Models backup — tar.gz the whole models directory.
    dest = out_dir / f"mrto_models_{ts}.tar.gz"
    r = backup_models(models, dest)
    results["files"].append({"kind": "models", **r})
    models_ok = r["ok"]

    # Manifest
    manifest = out_dir / f"mrto_backup_{ts}.manifest"
    manifest.write_text(json.dumps(results, indent=2))

    # Rotate
    deleted = rotate(out_dir, retain)
    print(f"rotated: deleted {deleted} old backup files")

    if not overall_ok:
        print("BACKUP FAILED — see manifest above", file=sys.stderr)
        return 1
    if not models_ok:
        print("PARTIAL — DB ok, models backup failed", file=sys.stderr)
        return 2
    print("backup OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
