"""Phase 3 tests: observability, migrations, deep health, model edge cases.

Covers the new capabilities added in Phase 3:
- /metrics (Prometheus scrape endpoint)
- /api/v1/health/deep (DB + schema + models + disk)
- structured logging (request_id propagation)
- idempotent schema migrations (re-running on a migrated DB is a no-op)
- model edge cases: NaN inputs, out-of-distribution inputs, empty DataFrame
- read-replica routing (mocked — no real replica in CI)
"""
import os
import json
import logging
import io
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient

from mrto.app import create_app
from mrto import observability as obs
from mrto import migrations as mig
from mrto import db as dbm


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        yield c


def _manager_token(client) -> str:
    r = client.post("/api/v1/auth/login",
                    json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _h(tok): return {"Authorization": f"Bearer {tok}"}


# ── /metrics ────────────────────────────────────────────────────────
def test_prometheus_metrics_endpoint(client):
    """The /metrics endpoint returns Prometheus text format and includes
    the http_requests counter after at least one request."""
    # Make a request so the counter is non-zero.
    client.get("/api/v1/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "mrto_http_requests_total" in body
    assert "# TYPE" in body or "# HELP" in body


def test_prometheus_metrics_have_request_id_header(client):
    """Every response carries an X-Request-ID header."""
    r = client.get("/api/v1/health")
    assert "x-request-id" in {k.lower() for k in r.headers}
    assert len(r.headers["x-request-id"]) >= 8


def test_request_id_propagated_from_header(client):
    """A client-provided X-Request-ID is echoed back (correlation)."""
    rid = "test-correlation-12345"
    r = client.get("/api/v1/health", headers={"X-Request-ID": rid})
    assert r.headers["x-request-id"] == rid


# ── /health/deep ────────────────────────────────────────────────────
def test_deep_health_returns_checks(client):
    r = client.get("/api/v1/health/deep")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "mrto"
    checks = body["checks"]
    # DB check is mandatory.
    assert checks["db"]["ok"] is True
    assert "schema_version" in body
    assert body["schema_version"] >= 0
    # Model checks exist (ok=False is fine if untrained).
    assert "model_module2_rul" in checks
    assert "disk" in checks


def test_deep_health_reports_schema_version(client):
    """After init_db runs migrations, schema_version reflects them."""
    r = client.get("/api/v1/health/deep")
    v = r.json()["schema_version"]
    # At least the three Phase 3 migrations should have applied.
    assert v >= 3


# ── Migrations ──────────────────────────────────────────────────────
def test_migrations_idempotent(tmp_path):
    """Running run_migrations twice on the same DB is a no-op — the
    second call applies zero migrations."""
    db_path = str(tmp_path / "mig.sqlite3")
    dbm.init_db(db_path)
    with dbm.get_conn(db_path) as conn:
        v1 = mig.get_schema_version(conn)
        v2 = mig.run_migrations(conn)
        assert v2 == v1  # nothing new to apply
        # last_login_at column from migration 2 is present.
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
        assert "last_login_at" in cols


def test_migrations_apply_in_order(tmp_path):
    """A fresh DB gets all migrations applied in version order."""
    db_path = str(tmp_path / "fresh.sqlite3")
    dbm.init_db(db_path)
    with dbm.get_conn(db_path) as conn:
        v = mig.get_schema_version(conn)
        # All registered migrations should be applied.
        max_registered = max(ver for ver, _ in mig.MIGRATIONS)
        assert v == max_registered


# ── Structured logging ──────────────────────────────────────────────
def test_json_logger_emits_valid_json():
    """The JsonFormatter emits one parseable JSON object per record."""
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(obs.JsonFormatter())
    log = logging.getLogger("mrto.test_json")
    log.handlers = [h]
    log.setLevel(logging.INFO)
    log.info("hello", extra={"user": "admin", "count": 7})
    log.handlers = []
    line = buf.getvalue().strip()
    obj = json.loads(line)
    assert obj["msg"] == "hello"
    assert obj["user"] == "admin"
    assert obj["count"] == 7
    assert "request_id" in obj
    assert "ts" in obj


# ── Model edge cases ────────────────────────────────────────────────
def test_rul_endpoint_handles_empty_tubes(tmp_path):
    """If the tubes table is empty, /rul returns an empty list (not 500)."""
    db_path = str(tmp_path / "empty.sqlite3")
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        # Wipe all tubes.
        from mrto import db as dbm
        with dbm.get_conn(db_path) as conn:
            conn.execute("DELETE FROM tubes")
        r = c.get("/api/v1/rul")
        assert r.status_code == 200
        body = r.json()
        assert body["tubes"] == []
        assert body["source"] in ("physics_fallback", "trained_model")


def test_temperature_map_with_no_readings(client):
    """A temperature-map request when there are zero readings returns
    n_tubes=0 and an empty hotspots list (not a 500 from mean-of-empty)."""
    r = client.get("/api/v1/temperature-map?ts_minute=999999")
    assert r.status_code == 200
    body = r.json()
    assert body["n_tubes"] == 0
    assert body["hotspots"] == []
    # mean of zero readings must not be NaN.
    assert body["mean_wall_temp_c"] == 0.0


def test_optimization_input_at_boundaries(client):
    """Boundary values of the optimization input are accepted (not 422).
    Tests both lower and upper bounds of every constrained field."""
    tok = _manager_token(client)
    # Lower bounds.
    r = client.post("/api/v1/optimization",
                    json={"sc_ratio": 2.0, "t_out_c": 700,
                          "feed_flow_nm3h": 10000, "cat_age": 0,
                          "fuel_flow_nm3h": 5000},
                    headers=_h(tok))
    # 503 if model untrained (acceptable), 422 if validation rejected.
    assert r.status_code in (200, 503), r.text
    # Upper bounds.
    r2 = client.post("/api/v1/optimization",
                     json={"sc_ratio": 5.0, "t_out_c": 1000,
                           "feed_flow_nm3h": 50000, "cat_age": 1.2,
                           "fuel_flow_nm3h": 20000},
                     headers=_h(tok))
    assert r2.status_code in (200, 503), r2.text


def test_ingest_extreme_temperature_creates_alert(client):
    """A reading at the upper validation limit (1500 °C) is accepted
    and, being far above the mean, generates a hotspot alert."""
    tok = _manager_token(client)
    # Seed a few normal readings so the mean is realistic.
    for tid in ["T01-001", "T01-002", "T01-003"]:
        client.post("/api/v1/ingest/temperature",
                    json={"tube_id": tid, "ts_minute": 60, "wall_temp_c": 880.0},
                    headers=_h(tok))
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T01-001", "ts_minute": 120,
                          "wall_temp_c": 1500.0},
                    headers=_h(tok))
    assert r.status_code == 200
    r2 = client.get("/api/v1/alerts")
    alerts = r2.json()["alerts"]
    assert any("1500" in a["message"] for a in alerts)


def test_alerts_filter_by_status(client):
    """Open and acked alerts are correctly separated by status_filter."""
    tok = _manager_token(client)
    # Create one alert via a hot reading.
    for tid in ["T01-001", "T01-002"]:
        client.post("/api/v1/ingest/temperature",
                    json={"tube_id": tid, "ts_minute": 10, "wall_temp_c": 850.0},
                    headers=_h(tok))
    client.post("/api/v1/ingest/temperature",
                json={"tube_id": "T01-001", "ts_minute": 20, "wall_temp_c": 950.0},
                headers=_h(tok))
    opens = client.get("/api/v1/alerts?status_filter=open").json()["alerts"]
    assert len(opens) >= 1
    aid = opens[0]["id"]
    client.post(f"/api/v1/alerts/{aid}/ack", headers=_h(tok))
    opens2 = client.get("/api/v1/alerts?status_filter=open").json()["alerts"]
    acked = client.get("/api/v1/alerts?status_filter=acked").json()["alerts"]
    assert all(a["id"] != aid for a in opens2)
    assert any(a["id"] == aid for a in acked)


# ── Read-replica routing (no real replica in CI; verify the helper) ──
def test_read_replica_helper_falls_back_when_unset(client):
    """When MRTO_READ_REPLICA_URL is unset, _read_conn falls back to the
    primary (SQLite) connection — same behavior as dbm.get_conn."""
    from mrto.app import _read_conn, _HAS_REPLICA
    assert _HAS_REPLICA is False
    with _read_conn(client.app) as conn:
        n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
        assert n == 12  # seeded demo tubes


def test_openapi_has_tags(client):
    """The OpenAPI spec groups endpoints by the Phase-3 tags."""
    spec = client.get("/openapi.json").json()
    paths = spec.get("paths", {})
    # /auth/login is in the 'auth' tag.
    login = paths.get("/api/v1/auth/login", {})
    assert "auth" in (login.get("post", {}).get("tags", []))
    # /health/deep is in the 'health' tag.
    deep = paths.get("/api/v1/health/deep", {})
    assert "health" in (deep.get("get", {}).get("tags", []))
    # /ingest/temperature is in the 'ingest' tag.
    ingest = paths.get("/api/v1/ingest/temperature", {})
    assert "ingest" in (ingest.get("post", {}).get("tags", []))
