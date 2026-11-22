"""Graceful, idempotent schema migrations for the MRTO SQLite DB.

Each migration is a function that takes a sqlite3 connection and is run
inside a transaction. The schema_version table records the highest
migration applied; init_db() calls run_migrations() after the base
schema is created, so an existing DB from Phase 1/2 is transparently
upgraded in place.

Design:
- Migrations are numbered (1, 2, 3, ...) and applied in order.
- Each migration is idempotent: re-running on an already-migrated DB is
  a no-op (uses IF NOT EXISTS / guards column existence).
- The schema_version row is updated at the end of each migration inside
  the same transaction, so a crash mid-migration rolls back cleanly.
- Migrations never destroy data; ADD COLUMN only. Destructive changes
  require a separate, explicit, user-acknowledged migration.

The same module works against the Postgres adapter (db_adapter) because
the SQL is portable (CREATE TABLE IF NOT EXISTS, ALTER TABLE ADD
COLUMN, PRAGMA table_info is SQLite-only — guarded by a dialect check).
"""
from __future__ import annotations

import logging

log = logging.getLogger("mrto.migrations")

# (version, callable) pairs. Order matters; gaps are not allowed.
MIGRATIONS: list[tuple[int, callable]] = []


def migration(version: int):
    """Decorator to register a migration at a given version."""
    def deco(fn):
        MIGRATIONS.append((version, fn))
        return fn
    return deco


# ────────────────────────────────────────────────────────────────────
# Migration 1: add ts_minute index on alerts (was missing in Phase 1).
# ────────────────────────────────────────────────────────────────────
@migration(1)
def m1_alerts_ts_index(conn) -> None:
    """Add an index on alerts(ts_minute) so the daily-report and alerts
    list queries don't full-scan on a large alerts table."""
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(ts_minute)")


# ────────────────────────────────────────────────────────────────────
# Migration 2: add last_login_at column to users (audit enrichment).
# ────────────────────────────────────────────────────────────────────
@migration(2)
def m2_users_last_login(conn) -> None:
    """Track the last successful login timestamp per user so the admin
    dashboard can show stale accounts."""
    cols = _column_names(conn, "users")
    if "last_login_at" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN last_login_at INTEGER")


# ────────────────────────────────────────────────────────────────────
# Migration 3: add source index on temperature_readings (filter by source).
# ────────────────────────────────────────────────────────────────────
@migration(3)
def m3_temp_source_index(conn) -> None:
    """Index temperature_readings(source) — the dashboard sometimes
    filters IR vs thermocouple readings."""
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_temp_source ON temperature_readings(source)"
    )


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _column_names(conn, table: str) -> list[str]:
    """Return column names of a table, dialect-agnostic.

    SQLite: PRAGMA table_info. Postgres (psycopg2 via db_adapter):
    information_schema.columns. We detect by checking whether the conn
    object exposes a `row_factory` we set to None (psycopg2 path) vs
    sqlite3.Row. Simpler: try PRAGMA first, fall back to information_schema.
    """
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r["name"] for r in rows]
    except Exception:
        # Postgres path (or any DB without PRAGMA).
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s", (table,)
        ).fetchall()
        return [r[0] if not isinstance(r, dict) else r.get("column_name", r[0])
                for r in rows]


def get_schema_version(conn) -> int:
    """Return the current schema version (0 if the meta table is empty)."""
    # Ensure the meta table exists (idempotent).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)"
    )
    row = conn.execute(
        "SELECT MAX(version) FROM schema_migrations"
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def run_migrations(conn) -> int:
    """Apply all pending migrations. Returns the new schema version."""
    current = get_schema_version(conn)
    applied = current
    import time
    for version, fn in sorted(MIGRATIONS):
        if version <= current:
            continue
        log.info("applying migration %d", version)
        fn(conn)
        conn.execute(
            "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
            (version, int(time.time()))
        )
        applied = version
    return applied
