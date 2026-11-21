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
    assert body["token_type"] == "Bearer"
    assert body["role"] == "manager"


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
    # Ingest a baseline
    for tid in ["T01-001", "T01-002", "T01-003"]:
        r = client.post("/api/v1/ingest/temperature",
                        json={"tube_id": tid, "ts_minute": 60, "wall_temp_c": 880.0})
        assert r.status_code == 200
    # Ingest a hot reading
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T01-001", "ts_minute": 120, "wall_temp_c": 950.0})
    assert r.status_code == 200
    # An alert should be generated
    r = client.get("/api/v1/alerts")
    alerts = r.json()["alerts"]
    assert len(alerts) >= 1
    assert "Hotspot" in alerts[0]["message"]


def test_ingest_creates_missing_tube(client):
    r = client.post("/api/v1/ingest/temperature",
                    json={"tube_id": "T09-099", "ts_minute": 200, "wall_temp_c": 880.0})
    assert r.status_code == 200
    r = client.get("/api/v1/tubes")
    tids = [t["tube_id"] for t in r.json()]
    assert "T09-099" in tids


def test_temperature_map_endpoint(client):
    # Ingest
    for tid in ["T01-001", "T01-002", "T01-003"]:
        client.post("/api/v1/ingest/temperature",
                    json={"tube_id": tid, "ts_minute": 60, "wall_temp_c": 870.0})
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
    # Generate an alert
    client.post("/api/v1/ingest/temperature",
                json={"tube_id": "T01-001", "ts_minute": 10, "wall_temp_c": 950.0})
    # Insert enough normal readings first to push mean down
    for i in range(5):
        client.post("/api/v1/ingest/temperature",
                    json={"tube_id": f"T01-00{i+2:03d}", "ts_minute": 10, "wall_temp_c": 850.0})
    r = client.get("/api/v1/alerts")
    alerts = r.json()["alerts"]
    assert len(alerts) > 0
    aid = alerts[0]["id"]
    # Ack it
    r = client.post(f"/api/v1/alerts/{aid}/ack")
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
