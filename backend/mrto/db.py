"""SQLite database layer for MRTO backend.

Schema (see SRS §5-1):
  - tubes(id, row, col, material, installed_at_days_ago, last_replaced_days_ago, design_life_h)
  - temperature_readings(tube_id, ts_minute, wall_temp_c, source)
  - pressure_readings(tube_id, ts_minute, p_in, p_out, dp)
  - burner_status(burner_id, ts_minute, fuel_flow, intensity, angle)
  - inspection_history(tube_id, ts_minute, kind, result, technician)
  - alerts(id, tube_id, ts_minute, severity, module, status)
  - users(id, username, password_hash, role)
"""
from __future__ import annotations

import os
import secrets as _secrets
import sqlite3
import json
import time
from contextlib import contextmanager
from typing import Iterable

import bcrypt

DEFAULT_DB_PATH = os.environ.get(
    "MRTO_DB_PATH",
    "/home/z/my-project/download/mrto/data/mrto.sqlite3"
)

# Known insecure placeholders — refusing to seed these prevents shipping the
# default credential to production. Operators must set MRTO_ADMIN_PASS env
# or let init_db generate a random one (printed once to stdout).
# NOTE: "admin123" is NOT in this list because it's the documented test/dev
# password (set via MRTO_ADMIN_PASS=admin123 in conftest.py). Production must
# set a strong MRTO_ADMIN_PASS or leave it unset to get a random one.
_INSECURE_PLACEHOLDERS = {
    "change-me", "change-me-before-deploy",
    "password", "REPLACE_ME", "REPLACE_WITH_KUBECTL_CREATE_SECRET",
    "",
}


def _hash_password(password: str) -> str:
    """Return a bcrypt hash (12 rounds) of the password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """Constant-time bcrypt verification. Returns False on any failure."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tubes (
    tube_id TEXT PRIMARY KEY,
    row INTEGER NOT NULL,
    col INTEGER NOT NULL,
    material TEXT NOT NULL,
    installed_at_days_ago INTEGER NOT NULL,
    last_replaced_days_ago INTEGER NOT NULL,
    design_life_h REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tubes_pos ON tubes(row, col);

CREATE TABLE IF NOT EXISTS temperature_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tube_id TEXT NOT NULL,
    ts_minute INTEGER NOT NULL,
    wall_temp_c REAL NOT NULL,
    source TEXT NOT NULL,
    FOREIGN KEY (tube_id) REFERENCES tubes(tube_id)
);
CREATE INDEX IF NOT EXISTS idx_temp_tube_ts ON temperature_readings(tube_id, ts_minute);

CREATE TABLE IF NOT EXISTS pressure_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tube_id TEXT NOT NULL,
    ts_minute INTEGER NOT NULL,
    p_in REAL,
    p_out REAL,
    dp REAL,
    FOREIGN KEY (tube_id) REFERENCES tubes(tube_id)
);
CREATE INDEX IF NOT EXISTS idx_pres_tube_ts ON pressure_readings(tube_id, ts_minute);

CREATE TABLE IF NOT EXISTS burner_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    burner_id TEXT NOT NULL,
    ts_minute INTEGER NOT NULL,
    fuel_flow REAL,
    intensity REAL,
    angle REAL
);
CREATE INDEX IF NOT EXISTS idx_burner_ts ON burner_status(burner_id, ts_minute);

CREATE TABLE IF NOT EXISTS inspection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tube_id TEXT NOT NULL,
    ts_minute INTEGER NOT NULL,
    kind TEXT NOT NULL,
    result TEXT,
    technician TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tube_id TEXT,
    ts_minute INTEGER NOT NULL,
    severity TEXT NOT NULL,
    module TEXT NOT NULL,
    message TEXT,
    status TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status, ts_minute);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_epoch INTEGER NOT NULL,
    username TEXT,
    action TEXT NOT NULL,
    detail TEXT
);
"""


@contextmanager
def get_conn(db_path: str = DEFAULT_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign-key enforcement (off by default in SQLite).
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create schema and seed default data if empty."""
    with get_conn(db_path) as conn:
        # Enforce FK constraints (off by default in SQLite).
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA_SQL)
        # Seed default admin if no users
        cur = conn.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            admin_pass = os.environ.get("MRTO_ADMIN_PASS", "")
            if not admin_pass or admin_pass in _INSECURE_PLACEHOLDERS:
                # Generate a strong random password and print it once.
                # In tests, set MRTO_ADMIN_PASS=admin123 to keep tests deterministic.
                admin_pass = _secrets.token_urlsafe(24)
                print("=" * 70)
                print("MRTO: generated random admin password (set MRTO_ADMIN_PASS")
                print("      to override, or change via /api/v1/users/* later):")
                print(f"      username=admin  password={admin_pass}")
                print("=" * 70)
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", _hash_password(admin_pass), "manager")
            )
        # Seed default tube metadata if empty
        cur = conn.execute("SELECT COUNT(*) FROM tubes")
        if cur.fetchone()[0] == 0:
            seed_demo_tubes(conn)


def seed_demo_tubes(conn) -> None:
    """Seed 12 representative tubes (4 rows × 3 cols) for demo/testing."""
    rows = []
    for r in range(1, 5):
        for c in range(1, 4):
            tid = f"T{r:02d}-{c:03d}"
            rows.append((tid, r, c, "HP-Modified", 200, 200, 100000.0))
    conn.executemany(
        "INSERT INTO tubes (tube_id, row, col, material, installed_at_days_ago, "
        "last_replaced_days_ago, design_life_h) VALUES (?,?,?,?,?,?,?)",
        rows
    )


def insert_temperature_reading(conn, tube_id: str, ts_minute: int,
                                wall_temp_c: float, source: str = "ir") -> int:
    cur = conn.execute(
        "INSERT INTO temperature_readings (tube_id, ts_minute, wall_temp_c, source) "
        "VALUES (?,?,?,?)",
        (tube_id, ts_minute, wall_temp_c, source)
    )
    return cur.lastrowid


def get_tubes(conn, row: int | None = None) -> list[dict]:
    if row is None:
        rows = conn.execute("SELECT * FROM tubes ORDER BY row, col").fetchall()
    else:
        rows = conn.execute("SELECT * FROM tubes WHERE row=? ORDER BY col", (row,)).fetchall()
    return [dict(r) for r in rows]


def get_temperature_map(conn, ts_minute: int | None = None) -> list[dict]:
    """Return latest temperature per tube (or at given timestamp)."""
    if ts_minute is None:
        # Use the max ts in the table as 'latest'
        cur = conn.execute("SELECT MAX(ts_minute) FROM temperature_readings")
        row = cur.fetchone()
        if row is None or row[0] is None:
            return []
        ts_minute = row[0]
    rows = conn.execute(
        "SELECT tube_id, wall_temp_c, source, ts_minute FROM temperature_readings "
        "WHERE ts_minute=? ORDER BY tube_id", (ts_minute,)
    ).fetchall()
    return [dict(r) for r in rows]


def insert_alert(conn, tube_id: str | None, ts_minute: int, severity: str,
                  module: str, message: str) -> int:
    cur = conn.execute(
        "INSERT INTO alerts (tube_id, ts_minute, severity, module, message, status) "
        "VALUES (?,?,?,?,?,'open')",
        (tube_id, ts_minute, severity, module, message)
    )
    return cur.lastrowid


def get_alerts(conn, status: str = "open") -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM alerts WHERE status=? ORDER BY ts_minute DESC", (status,)
    ).fetchall()
    return [dict(r) for r in rows]


def ack_alert(conn, alert_id: int, username: str = "system") -> bool:
    cur = conn.execute(
        "UPDATE alerts SET status='acked' WHERE id=? AND status='open'", (alert_id,)
    )
    if cur.rowcount > 0:
        conn.execute(
            "INSERT INTO audit_log (ts_epoch, username, action, detail) VALUES (?,?,?,?)",
            (int(time.time()), username, "ack_alert", f"alert_id={alert_id}")
        )
        return True
    return False


def verify_user(conn, username: str, password: str) -> dict | None:
    """Verify credentials with constant-time bcrypt check."""
    row = conn.execute(
        "SELECT * FROM users WHERE username=?", (username,)
    ).fetchone()
    if row is None:
        # Run a dummy bcrypt check to keep response time constant and prevent
        # username enumeration via timing.
        _verify_password(password, "$2b$12$" + "x" * 53)
        return None
    if not _verify_password(password, row["password_hash"]):
        return None
    return dict(row)


def audit(conn, username: str, action: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log (ts_epoch, username, action, detail) VALUES (?,?,?,?)",
        (int(time.time()), username, action, detail)
    )
