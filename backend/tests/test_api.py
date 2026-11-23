"""Integration tests for the FastAPI backend using TestClient."""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from mrto.app import create_app


@pytest.fixture
def client(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    app = create_app(db_path=db_path)
    with TestClient(app) as c:
        yield c


def _manager_token(client) -> str:
    """Log in as the seeded admin (manager role) and return a Bearer access token."""
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _login_full(client) -> dict:
    """Log in and return the full token bundle (access + refresh)."""
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    return r.json()


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "mrto"
    assert "n_tubes" in body
    assert "open_alerts" in body


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "MRTO Backend"
    assert "version" in body


def test_login_success(client):
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert "refresh_token" in body          # Phase 2: refresh token issued
    assert body["token_type"] == "Bearer"
    assert body["role"] == "manager"
    assert body["expires_in"] > 0           # Phase 2: TTL exposed for UI


def test_login_failure(client):
    r = client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_get_tubes(client):
    r = client.get("/api/v1/tubes")
    assert r.status_code == 200
    tubes = r.json()
    assert isinstance(tubes, list)
    assert len(tubes) == 12


def test_get_tubes_filtered_by_row(client):
    r = client.get("/api/v1/tubes?row=2")
    assert r.status_code == 200
    tubes = r.json()
    assert len(tubes) == 3
    assert all(t["row"] == 2 for t in tubes)


def test_ingest_temperature_and_hotspot_alert(client):
    tok = _manager_token(client)
    h = _auth_headers(tok)
    # Ingest a baseline
    for tid in ["T01-001", "T01-002", "T01-003"]:
        r = client.post("/api/v1/ingest/temperature",
                        json={"tube_id": tid, "ts_minute": 60, "wall_temp_c": 880.0},
                        headers=h)
        assert r.status_code == 200, r.text
    # Ingest a hot reading
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T01-001", "ts_minute": 120, "wall_temp_c": 950.0},
                    headers=h)
    assert r.status_code == 200
    # An alert should be generated
    r = client.get("/api/v1/alerts")
    alerts = r.json()["alerts"]
    assert len(alerts) >= 1
    assert "Hotspot" in alerts[0]["message"]


def test_ingest_creates_missing_tube(client):
    tok = _manager_token(client)
    h = _auth_headers(tok)
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T09-099", "ts_minute": 200, "wall_temp_c": 880.0},
                    headers=h)
    assert r.status_code == 200
    r = client.get("/api/v1/tubes")
    tids = [t["tube_id"] for t in r.json()]
    assert "T09-099" in tids


def test_ingest_rejects_anonymous(client):
    """RBAC: anonymous viewers (dev fallback) must NOT be allowed to ingest."""
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T01-001", "ts_minute": 60, "wall_temp_c": 880.0})
    assert r.status_code == 403


def test_ingest_rejects_bad_input(client):
    """Input validation: out-of-range wall_temp_c is rejected."""
    tok = _manager_token(client)
    h = _auth_headers(tok)
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T01-001", "ts_minute": 60, "wall_temp_c": 9999.0},
                    headers=h)
    assert r.status_code == 422


def test_temperature_map_endpoint(client):
    tok = _manager_token(client)
    h = _auth_headers(tok)
    # Ingest
    for tid in ["T01-001", "T01-002", "T01-003"]:
        client.post("/api/v1/ingest/temperature",
                    json={"tube_id": tid, "ts_minute": 60, "wall_temp_c": 870.0},
                    headers=h)
    r = client.get("/api/v1/temperature-map?ts_minute=60")
    assert r.status_code == 200
    body = r.json()
    assert body["n_tubes"] == 3
    assert abs(body["mean_wall_temp_c"] - 870.0) < 1
    assert "hotspots" in body
    assert isinstance(body["readings"], list)


def test_rul_endpoint_fallback(client):
    """If no model trained, RUL endpoint returns physics-fallback."""
    r = client.get("/api/v1/rul")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] in ("physics_fallback", "trained_model")
    assert isinstance(body["tubes"], list)
    for t in body["tubes"]:
        assert "rul_h" in t and t["rul_h"] >= 0
        assert 0 <= t["p_fail_90d"] <= 1


def test_alert_ack_workflow(client):
    tok = _manager_token(client)
    h = _auth_headers(tok)
    # Generate an alert
    client.post("/api/v1/ingest/temperature",
                json={"tube_id": "T01-001", "ts_minute": 10, "wall_temp_c": 950.0},
                headers=h)
    # Insert enough normal readings first to push mean down
    for i in range(5):
        client.post("/api/v1/ingest/temperature",
                    json={"tube_id": f"T01-00{i+2:03d}", "ts_minute": 10, "wall_temp_c": 850.0},
                    headers=h)
    r = client.get("/api/v1/alerts")
    alerts = r.json()["alerts"]
    assert len(alerts) > 0
    aid = alerts[0]["id"]
    # Ack it
    r = client.post(f"/api/v1/alerts/{aid}/ack", headers=h)
    assert r.status_code == 200
    assert r.json()["status"] == "acked"
    # Verify it's gone from open list
    r = client.get("/api/v1/alerts")
    assert all(a["id"] != aid for a in r.json()["alerts"])


def test_alert_ack_nonexistent(client):
    r = client.post("/api/v1/alerts/99999/ack")
    assert r.status_code == 404


def test_burners_endpoint_empty(client):
    r = client.get("/api/v1/burners")
    assert r.status_code == 200
    body = r.json()
    assert "burners" in body and "n" in body


def test_metrics_endpoint(client):
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    body = r.json()
    # All four modules should have keys (may be None if not trained)
    for k in ["module1_temperature_map", "module2_rul",
              "module3_optimization", "module4_burner_catalyst"]:
        assert k in body


def test_daily_report(client):
    r = client.get("/api/v1/reports/daily")
    assert r.status_code == 200
    body = r.json()
    assert "report_date" in body
    assert "open_alerts" in body
    assert "n_tubes" in body


def test_invalid_optimization_input(client):
    # S/C below 2.0 should fail validation
    r = client.post("/api/v1/optimization", json={
        "sc_ratio": 1.5, "t_out_c": 880, "feed_flow_nm3h": 32000,
        "cat_age": 0.3, "fuel_flow_nm3h": 11500,
    })
    assert r.status_code == 422


def test_openapi_docs_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    assert "MRTO" in r.json()["info"]["title"]


# ── JWT security tests ──────────────────────────────────────────────
def test_jwt_tampered_payload_rejected(client):
    """A token with a tampered payload must fail verification (401)."""
    tok = _manager_token(client)
    # Tamper: flip the last char of the payload section.
    parts = tok.split(".")
    tampered = parts[0] + "." + parts[1][:-1] + ("A" if parts[1][-1] != "A" else "B") + "." + parts[2]
    r = client.get("/api/v1/tubes", headers={"Authorization": f"Bearer {tampered}"})
    assert r.status_code == 401


def test_jwt_expired_rejected(client):
    """An expired token must be rejected."""
    import time as _t
    from mrto.app import make_jwt, JWT_SECRET
    # Build a token that expired 1 second ago.
    import base64, hmac, hashlib, json as _json
    def _b64(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    header = _b64(_json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = {"sub": "admin", "role": "manager", "iat": int(_t.time()) - 100,
               "exp": int(_t.time()) - 1}
    body = _b64(_json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64(hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest())
    expired = f"{header}.{body}.{sig}"
    r = client.get("/api/v1/tubes", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


def test_jwt_alg_none_rejected(client):
    """An alg=none token must be rejected (alg-confusion defense)."""
    import base64, json as _json, time as _t
    def _b64(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    header = _b64(_json.dumps({"alg": "none", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = {"sub": "admin", "role": "manager", "iat": int(_t.time()),
               "exp": int(_t.time()) + 3600}
    body = _b64(_json.dumps(payload, separators=(",", ":")).encode())
    fake = f"{header}.{body}."
    r = client.get("/api/v1/tubes", headers={"Authorization": f"Bearer {fake}"})
    assert r.status_code == 401


def test_protected_endpoint_rejects_bad_token(client):
    """A malformed (non-JWT) bearer string must be rejected."""
    r = client.get("/api/v1/tubes", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401


# ── Phase 2: refresh token + rotation tests ────────────────────────
def test_refresh_token_rotation(client):
    """A refresh token can be exchanged for a new access+refresh pair,
    and the OLD refresh token is then invalid (single-use rotation)."""
    bundle = _login_full(client)
    old_refresh = bundle["refresh_token"]
    # Refresh once → new tokens.
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["access_token"] != bundle["access_token"]
    assert new["refresh_token"] != old_refresh
    # Old refresh token is now dead (reuse → family revoked).
    r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert r2.status_code == 401
    assert "reuse" in r2.json()["detail"].lower()


def test_refresh_token_reuse_revokes_family(client):
    """If an attacker replays a rotated refresh token, the whole family
    is revoked — so the legitimate client's NEW refresh also dies."""
    bundle = _login_full(client)
    old_refresh = bundle["refresh_token"]
    # Attacker uses the old (still-valid) token first → gets new tokens.
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    legit_new = r.json()["refresh_token"]
    # Victim (legitimate client) now tries the same old token → reuse detected,
    # family revoked.
    r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert r2.status_code == 401
    # The legitimately-rotated token is now ALSO dead (family revoked).
    r3 = client.post("/api/v1/auth/refresh", json={"refresh_token": legit_new})
    assert r3.status_code == 401


def test_refresh_rejects_access_token(client):
    """An access token presented as a refresh token must be rejected."""
    bundle = _login_full(client)
    access = bundle["access_token"]
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": access})
    assert r.status_code == 401


def test_access_rejects_refresh_token(client):
    """A refresh token presented as a Bearer access token must be rejected."""
    bundle = _login_full(client)
    r = client.get("/api/v1/tubes",
                    headers={"Authorization": f"Bearer {bundle['refresh_token']}"})
    assert r.status_code == 401


def test_logout_revokes_refresh_family(client):
    """Logout revokes the refresh family so subsequent refresh fails."""
    bundle = _login_full(client)
    access = bundle["access_token"]
    refresh = bundle["refresh_token"]
    r = client.post("/api/v1/auth/logout", json={"refresh_token": refresh},
                     headers=_auth_headers(access))
    assert r.status_code == 200
    assert r.json()["revoked"] >= 1
    # Refresh now fails (token revoked).
    r2 = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh})
    assert r2.status_code == 401


def test_refresh_invalid_token(client):
    """A garbage refresh token is rejected with 401."""
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": "garbage.payload.sig"})
    assert r.status_code == 401


# ── Phase 2: rate limiting tests ───────────────────────────────────
def test_login_rate_limit(client):
    """More than 10 failed logins/minute from the same IP → 429."""
    for _ in range(10):
        client.post("/api/v1/auth/login",
                    json={"username": "admin", "password": "wrong"})
    r = client.post("/api/v1/auth/login",
                    json={"username": "admin", "password": "wrong"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers or "retry_after_s" in r.json()


def test_rate_limit_headers_present(client):
    """Successful requests carry X-RateLimit-* headers."""
    r = client.get("/api/v1/health")
    # /health is exempt from the limiter, so test a real endpoint.
    r = client.get("/api/v1/tubes")
    assert "X-RateLimit-Limit" in r.headers
    assert "X-RateLimit-Remaining" in r.headers


# ── Phase 2: model versioning tests ────────────────────────────────
def test_model_versions_empty_by_default(client):
    """Before any training, the versions endpoint returns an empty list
    but requires auth (manager role)."""
    tok = _manager_token(client)
    r = client.get("/api/v1/models/versions", headers=_auth_headers(tok))
    assert r.status_code == 200
    assert r.json()["n"] == 0


def test_model_versions_rejects_viewer(client):
    """Anonymous viewers cannot read the model-version audit trail."""
    r = client.get("/api/v1/models/versions")
    # Anonymous → viewer role → 403 (manager-only endpoint).
    assert r.status_code == 403


def test_active_model_404_when_untrained(client):
    """Asking for an active model that was never registered → 404."""
    r = client.get("/api/v1/models/module2_rul/active")
    assert r.status_code == 404


def test_register_and_list_model_version(client, tmp_path):
    """Register a model version directly via the db layer and verify the
    endpoint returns it with sha256 + metrics snapshot."""
    from mrto import db as dbm
    db_path = str(tmp_path / "test.sqlite3")
    dbm.init_db(db_path)
    with dbm.get_conn(db_path) as conn:
        dbm.register_model_version(
            conn, module="module2_rul", version="20260101.120000",
            file_name="module2_rul.joblib", sha256="a" * 64,
            metrics_json='{"mae_hours": 5000}', make_active=True,
        )
    tok = _manager_token(client)
    r = client.get("/api/v1/models/versions?module=module2_rul",
                    headers=_auth_headers(tok))
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 1
    v = body["versions"][0]
    assert v["module"] == "module2_rul"
    assert v["version"] == "20260101.120000"
    assert v["is_active"] == 1
    r2 = client.get("/api/v1/models/module2_rul/active")
    assert r2.status_code == 200
    assert r2.json()["sha256"].startswith("a" * 12)
    assert r2.json()["metrics"]["mae_hours"] == 5000


def test_temperature_batch_dedups_hotspot_alerts(client):
    h = _auth_headers(_manager_token(client))
    readings = []
    for minute in range(3):
        readings.append({"tube_id": "T01-001", "ts_minute": minute, "wall_temp_c": 860.0})
        readings.append({"tube_id": "T01-002", "ts_minute": minute, "wall_temp_c": 862.0})
        readings.append({"tube_id": "T01-003", "ts_minute": minute, "wall_temp_c": 950.0})
    r = client.post("/api/v1/ingest/temperature/batch", json={"readings": readings}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["n"] == 9
    alerts = client.get("/api/v1/alerts").json()["alerts"]
    hot = [a for a in alerts if a["tube_id"] == "T01-003" and a["status"] == "open"]
    assert len(hot) == 1


def test_burners_keep_latest_row_per_burner(tmp_path):
    from mrto import db as dbm
    db_path = str(tmp_path / "burners.sqlite3")
    app = create_app(db_path=db_path)
    with dbm.get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO burner_status (burner_id, ts_minute, fuel_flow, intensity, angle) VALUES (?,?,?,?,?)",
            ("B01-L", 1, 190, 0.95, 0),
        )
        conn.execute(
            "INSERT INTO burner_status (burner_id, ts_minute, fuel_flow, intensity, angle) VALUES (?,?,?,?,?)",
            ("B01-L", 2, 250, 0.40, 1),
        )
    with TestClient(app) as c:
        body = c.get("/api/v1/burners").json()
    by_id = {b["burner_id"]: b for b in body["burners"]}
    assert by_id["B01-L"]["ts_minute"] == 2
    assert body["n"] == 1
    cat = body["catalyst"]
    assert cat["status"] in {"healthy", "aging", "replace", "unknown"}
    assert cat["age_h"] == 8000.0
    assert cat["dp_bar"] == 1.6
    assert cat["sc_avg"] == 3.0
    assert cat["activity"] == body["catalyst_activity"]


def test_temperature_map_reports_latest_timestamp(client):
    h = _auth_headers(_manager_token(client))
    client.post("/api/v1/ingest/temperature",
                json={"tube_id": "T01-001", "ts_minute": 40, "wall_temp_c": 870.0},
                headers=h)
    body = client.get("/api/v1/temperature-map").json()
    assert body["ts_minute"] == 40
