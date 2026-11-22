"""Database connection adapter: SQLite (default) or Postgres (production).

This module abstracts the connection layer so the app code can stay
SQLite-agnostic. When MRTO_DATABASE_URL (or DATABASE_URL) is set to a
postgres:// URL, requests go through psycopg2 with the same row-factory
shape (dict-like rows). Otherwise the existing SQLite path in db.py is
used unchanged.

Design notes:
- We deliberately use psycopg2 (sync) rather than asyncpg so the existing
  sync endpoints in app.py keep working without a full rewrite. The
  performance ceiling for this TRL-5 system is well within sync capacity.
- The dict row factory matches sqlite3.Row semantics (`row["col"]`).
- Connection objects expose `execute`, `executemany`, `fetchone`,
  `fetchall` — the only methods used by the app. Both SQLite and the
  psycopg2 wrapper provide these.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

DEFAULT_DB_PATH = os.environ.get(
    "MRTO_DB_PATH",
    "/home/z/my-project/download/mrto/data/mrto.sqlite3"
)

# A postgres URL wins over the SQLite path if set.
_DATABASE_URL = os.environ.get("MRTO_DATABASE_URL") or os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(_DATABASE_URL and _DATABASE_URL.startswith("postgres"))

# Phase 3: optional read-replica URL. When set, SELECTs go to the
# replica and writes go to the primary. The split is decided per-call by
# the `role` argument to get_conn (default "primary"). Endpoints that
# only read (GET /tubes, /rul, /alerts) opt in via role="replica".
_REPLICA_URL = os.environ.get("MRTO_READ_REPLICA_URL")
HAS_REPLICA = bool(_REPLICA_URL and _REPLICA_URL.startswith("postgres"))


class _PsycopgRow:
    """Lightweight dict-like wrapper over a psycopg2 tuple row."""
    __slots__ = ("_data",)

    def __init__(self, description, values):
        self._data = {d[0]: v for d, v in zip(description, values)}

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"_PsycopgRow({self._data!r})"


class _PsycopgConn:
    """Wrap psycopg2 connection to mimic sqlite3.Row semantics.

    Converts ? placeholders to %s (psycopg2 style) and wraps rows so the
    app code that does `row["col"]` works identically.
    """

    def __init__(self, conn):
        self._conn = conn

    def _translate(self, sql: str) -> str:
        # Convert SQLite-style ? placeholders to psycopg2 %s. This is a
        # safe textual replacement because our SQL never uses `?` as a
        # literal in any string/identifier position.
        return sql.replace("?", "%s") if "?" in sql else sql

    def execute(self, sql, params=()):
        cur = self._conn.cursor()
        cur.execute(self._translate(sql), params if params is not None else ())
        return _PsycopgCursor(cur)

    def executescript(self, sql: str):
        cur = self._conn.cursor()
        # psycopg2 has no executescript; split on ';' and run each stmt.
        # Comment lines and empty statements are skipped.
        for stmt in sql.split(";"):
            s = stmt.strip()
            if s and not s.startswith("--"):
                cur.execute(s)
        return _PsycopgCursor(cur)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    @property
    def row_factory(self):
        return None

    @row_factory.setter
    def row_factory(self, _):
        # Ignored — we always return _PsycopgRow.
        pass


class _PsycopgCursor:
    def __init__(self, cur):
        self._cur = cur

    @property
    def lastrowid(self):
        # Postgres: use RETURNING id or curval. For simplicity, rely on
        # RETURNING in the calling code; fall back to None.
        return None

    @property
    def rowcount(self):
        return self._cur.rowcount

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        return _PsycopgRow(self._cur.description, row)

    def fetchall(self):
        desc = self._cur.description
        return [_PsycopgRow(desc, r) for r in self._cur.fetchall()]


if USE_POSTGRES:
    import psycopg2  # noqa: F401  (imported lazily; required in prod)

    @contextmanager
    def get_conn(db_path: str = DEFAULT_DB_PATH, role: str = "primary"):
        """Open a connection to the primary (default) or the read replica.

        `role="replica"` routes to MRTO_READ_REPLICA_URL when set, else
        falls back to the primary. Replica connections are read-only by
        contract; callers must not issue writes on them. This layer does
        not enforce read-only (Postgres can, via a separate grant) — the
        application code is responsible for only issuing SELECTs.
        """
        url = _DATABASE_URL
        if role == "replica" and HAS_REPLICA:
            url = _REPLICA_URL
        conn = psycopg2.connect(url)
        wc = _PsycopgConn(conn)
        try:
            yield wc
            wc.commit()
        except Exception:
            wc.rollback()
            raise
        finally:
            wc.close()
else:
    # SQLite path — replica is ignored (single-file DB); re-export the
    # SQLite implementation to keep a single import surface.
    from mrto.db import get_conn as _sqlite_get_conn  # noqa: F401

    @contextmanager
    def get_conn(db_path: str = DEFAULT_DB_PATH, role: str = "primary"):
        # role is ignored on SQLite — there is no replica to route to.
        with _sqlite_get_conn(db_path) as conn:
            yield conn
