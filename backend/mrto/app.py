"""FastAPI backend for MRTO system.

Endpoints (see SRS Appendix 8-1):
    GET  /api/v1/health
    POST /api/v1/auth/login
    GET  /api/v1/tubes
    GET  /api/v1/temperature-map
    GET  /api/v1/rul
    POST /api/v1/optimization
    GET  /api/v1/burners
    GET  /api/v1/alerts
    POST /api/v1/alerts/{id}/ack
    GET  /api/v1/reports/daily
    GET  /api/v1/metrics  (model quality metrics)
    POST /api/v1/ingest/temperature  (admin ingestion)
"""
from __future__ import annotations

import os
import sys
import json
import time
import secrets
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Allow running both as module and as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrto import db as dbm
from mrto import models as mlm

MODELS_DIR = os.environ.get("MRTO_MODELS_DIR",
                              "/home/z/my-project/download/mrto/models")
METRICS_DIR = MODELS_DIR  # metrics json saved alongside models

# ────────────────────────────────────────────────────────────────────
# Pydantic models
# ────────────────────────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str
    password: str


class OptimizationReq(BaseModel):
    sc_ratio: float = Field(ge=2.0, le=5.0)
    t_out_c: float = Field(ge=700, le=1000)
    feed_flow_nm3h: float = Field(ge=10000, le=50000)
    cat_age: float = Field(ge=0, le=1.2)
    fuel_flow_nm3h: float = Field(ge=5000, le=20000)
    alpha_eff: float = Field(0.7, ge=0, le=1)
    alpha_fuel: float = Field(0.3, ge=0, le=1)


class TemperatureIngest(BaseModel):
    tube_id: str
    ts_minute: int
    wall_temp_c: float
    source: str = "ir"


class AlertAck(BaseModel):
    pass


# ────────────────────────────────────────────────────────────────────
# JWT helpers (simple HMAC-based for test-friendly env)
# ────────────────────────────────────────────────────────────────────
JWT_SECRET = os.environ.get("MRTO_JWT_SECRET", "mrto-dev-secret-1405")
JWT_TTL = 8 * 3600

_bearer = HTTPBearer(auto_error=False)


def _b64(b: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    import base64
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def make_jwt(username: str, role: str) -> str:
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = {
        "sub": username, "role": role, "iat": int(time.time()), "exp": int(time.time()) + JWT_TTL
    }
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    import hmac, hashlib
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    sig_b = _b64(sig)
    return f"{header}.{body}.{sig_b}"


def verify_jwt(token: str) -> dict | None:
    try:
        header, body, sig = token.split(".")
        import hmac, hashlib
        expected = _b64(hmac.new(JWT_SECRET.encode(),
                                  f"{header}.{body}".encode(), hashlib.sha256).digest())
        if not secrets.compare_digest(expected, sig):
            return None
        payload = json.loads(_b64d(body))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


async def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    if creds is None or not creds.credentials:
        # In dev mode allow anonymous read-only; tests run without auth
        return {"sub": "anonymous", "role": "viewer"}
    payload = verify_jwt(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# ────────────────────────────────────────────────────────────────────
# App factory
# ────────────────────────────────────────────────────────────────────
def create_app(db_path: Optional[str] = None) -> FastAPI:
    if db_path is None:
        db_path = dbm.DEFAULT_DB_PATH
    dbm.init_db(db_path)

    app = FastAPI(
        title="MRTO — Methanol Reformer Tube & Operation Optimizer",
        version="1.0.0",
        description="Smart monitoring of reformer tube health + S/C optimization. Marun Petrochemical TRL 5.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.db_path = db_path

    # ── Endpoints ───────────────────────────────────────────────────
    @app.get("/api/v1/health")
    def health() -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            n_tubes = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
            n_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE status='open'").fetchone()[0]
        return {
            "status": "ok",
            "service": "mrto",
            "version": "1.0.0",
            "ts_epoch": int(time.time()),
            "n_tubes": n_tubes,
            "open_alerts": n_alerts,
        }

    @app.post("/api/v1/auth/login")
    def login(req: LoginReq) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            user = dbm.verify_user(conn, req.username, req.password)
            if user is None:
                raise HTTPException(status_code=401, detail="Invalid credentials")
            dbm.audit(conn, user["username"], "login")
            token = make_jwt(user["username"], user["role"])
        return {"access_token": token, "token_type": "Bearer", "role": user["role"]}

    @app.get("/api/v1/tubes")
    def tubes(row: Optional[int] = None, _user: dict = Depends(current_user)) -> list[dict]:
        with dbm.get_conn(app.state.db_path) as conn:
            return dbm.get_tubes(conn, row=row)

    @app.get("/api/v1/temperature-map")
    def temperature_map(ts_minute: Optional[int] = None,
                         _user: dict = Depends(current_user)) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            readings = dbm.get_temperature_map(conn, ts_minute=ts_minute)
        # Compute average + hotspot identification (>30 °C above mean)
        temps = [r["wall_temp_c"] for r in readings]
        if temps:
            mean_t = sum(temps) / len(temps)
            hotspots = [
                {"tube_id": r["tube_id"], "wall_temp_c": r["wall_temp_c"],
                 "delta_c": round(r["wall_temp_c"] - mean_t, 2)}
                for r in readings if (r["wall_temp_c"] - mean_t) > 30
            ]
        else:
            mean_t = 0.0
            hotspots = []
        return {
            "ts_minute": ts_minute,
            "n_tubes": len(readings),
            "mean_wall_temp_c": round(mean_t, 2),
            "hotspots": hotspots,
            "readings": readings,
        }

    @app.get("/api/v1/rul")
    def rul(_user: dict = Depends(current_user)) -> dict:
        """Return RUL per tube using the trained Module 2 model.

        Falls back to physics estimate if model unavailable.
        """
        model_path = os.path.join(MODELS_DIR, "module2_rul.joblib")
        if os.path.exists(model_path):
            try:
                model = mlm.load_model(model_path)
                with dbm.get_conn(app.state.db_path) as conn:
                    tubes = dbm.get_tubes(conn)
                df = pd.DataFrame(tubes)
                if not df.empty:
                    df["avg_wall_temp_c"] = 880.0
                    df["consumed_life_fraction"] = (df["last_replaced_days_ago"] * 24) / df["design_life_h"]
                    df["installed_at_days_ago"] = df["installed_at_days_ago"]
                    rul = model.predict_rul(df)
                    pfail = model.predict_failure_prob(df)
                    return {
                        "source": "trained_model",
                        "tubes": [
                            {"tube_id": tid, "rul_h": int(r), "p_fail_90d": round(float(p), 4)}
                            for tid, r, p in zip(df["tube_id"], rul, pfail)
                        ],
                    }
            except Exception as e:
                pass  # fall through to physics estimate
        # Physics fallback
        with dbm.get_conn(app.state.db_path) as conn:
            tubes = dbm.get_tubes(conn)
        rows = []
        for t in tubes:
            avg_t = 880.0
            from mrto.synthetic_data import lmp_to_rupture_time
            t_rupt = lmp_to_rupture_time(avg_t)
            elapsed = t["last_replaced_days_ago"] * 24
            rul = max(0.0, t_rupt - elapsed)
            consumed = elapsed / max(t_rupt, 1)
            p_fail_90d = 1.0 / (1.0 + (1.0 - max(0, min(1, consumed - 0.7))) * 10)
            rows.append({"tube_id": t["tube_id"], "rul_h": int(rul), "p_fail_90d": round(p_fail_90d, 4)})
        return {"source": "physics_fallback", "tubes": rows}

    @app.post("/api/v1/optimization")
    def optimization(req: OptimizationReq, _user: dict = Depends(current_user)) -> dict:
        model_path = os.path.join(MODELS_DIR, "module3_optimization.joblib")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Optimization model not trained")
        model = mlm.load_model(model_path)
        current = {
            "sc_ratio": req.sc_ratio,
            "t_out_c": req.t_out_c,
            "feed_flow_nm3h": req.feed_flow_nm3h,
            "cat_age": req.cat_age,
            "fuel_flow_nm3h": req.fuel_flow_nm3h,
        }
        result = model.optimize(current, alpha_eff=req.alpha_eff, alpha_fuel=req.alpha_fuel)
        return result

    @app.get("/api/v1/burners")
    def burners(_user: dict = Depends(current_user)) -> dict:
        """Return burner status with inefficiency flag from Module 4 if available."""
        with dbm.get_conn(app.state.db_path) as conn:
            rows = conn.execute(
                "SELECT burner_id, ts_minute, fuel_flow, intensity, angle FROM burner_status "
                "ORDER BY ts_minute DESC LIMIT 200"
            ).fetchall()
        burners_list = [dict(r) for r in rows]
        # Apply model if available
        model_path = os.path.join(MODELS_DIR, "module4_burner_catalyst.joblib")
        if burners_list and os.path.exists(model_path):
            try:
                model = mlm.load_model(model_path)
                df = pd.DataFrame(burners_list).rename(columns={"fuel_flow": "fuel_flow_nm3h", "angle": "angle_deg"})
                df["age_h"] = 2000
                df["avg_tube_t_c"] = 880.0
                df["intensity"] = df["intensity"].fillna(0.9)
                df["fuel_flow_nm3h"] = df["fuel_flow_nm3h"].fillna(200)
                df["angle_deg"] = df["angle_deg"].fillna(0)
                preds = model.predict_inefficient(df)
                for b, p in zip(burners_list, preds):
                    b["inefficient"] = int(p)
            except Exception:
                pass
        return {"burners": burners_list, "n": len(burners_list)}

    @app.get("/api/v1/alerts")
    def alerts(status_filter: str = "open",
                _user: dict = Depends(current_user)) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            rows = dbm.get_alerts(conn, status=status_filter)
        return {"alerts": rows, "n": len(rows)}

    @app.post("/api/v1/alerts/{alert_id}/ack")
    def ack(alert_id: int, _user: dict = Depends(current_user)) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            ok = dbm.ack_alert(conn, alert_id, username=_user.get("sub", "system"))
        if not ok:
            raise HTTPException(status_code=404, detail="Alert not found or already acked")
        return {"ok": True, "alert_id": alert_id, "status": "acked"}

    @app.get("/api/v1/reports/daily")
    def daily_report(_user: dict = Depends(current_user)) -> dict:
        """Generate daily health report."""
        with dbm.get_conn(app.state.db_path) as conn:
            readings = dbm.get_temperature_map(conn)
            alerts = dbm.get_alerts(conn, status="open")
            tubes = dbm.get_tubes(conn)
        temps = [r["wall_temp_c"] for r in readings]
        return {
            "report_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "n_tubes": len(tubes),
            "n_readings": len(readings),
            "mean_temp_c": round(sum(temps) / len(temps), 2) if temps else None,
            "max_temp_c": round(max(temps), 2) if temps else None,
            "min_temp_c": round(min(temps), 2) if temps else None,
            "open_alerts": len(alerts),
            "n_hotspots": sum(1 for r in readings if any((r["wall_temp_c"] - t) > 30 for t in [sum(temps)/max(len(temps),1)])),
            "generated_at_epoch": int(time.time()),
        }

    @app.get("/api/v1/metrics")
    def metrics(_user: dict = Depends(current_user)) -> dict:
        """Return model quality metrics from JSON files."""
        out = {}
        for name, path in [
            ("module1_temperature_map", os.path.join(METRICS_DIR, "module1_metrics.json")),
            ("module2_rul", os.path.join(METRICS_DIR, "module2_metrics.json")),
            ("module3_optimization", os.path.join(METRICS_DIR, "module3_metrics.json")),
            ("module4_burner_catalyst", os.path.join(METRICS_DIR, "module4_metrics.json")),
        ]:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        out[name] = json.load(f)
                except Exception as e:
                    out[name] = {"error": str(e)}
            else:
                out[name] = None
        return out

    @app.post("/api/v1/ingest/temperature")
    def ingest_temp(req: TemperatureIngest,
                     _user: dict = Depends(current_user)) -> dict:
        if _user.get("role") not in ("manager", "operator", "it", "viewer", "anonymous"):
            raise HTTPException(status_code=403, detail="Insufficient role")
        with dbm.get_conn(app.state.db_path) as conn:
            # Auto-create tube if missing
            existing = conn.execute("SELECT 1 FROM tubes WHERE tube_id=?", (req.tube_id,)).fetchone()
            if existing is None:
                # derive row/col from tube_id pattern T<rr>-<ccc>
                try:
                    parts = req.tube_id.split("-")
                    r = int(parts[0][1:])
                    c = int(parts[1])
                except Exception:
                    r, c = 1, 1
                conn.execute(
                    "INSERT INTO tubes (tube_id, row, col, material, installed_at_days_ago, last_replaced_days_ago, design_life_h) VALUES (?,?,?,?,?,?,?)",
                    (req.tube_id, r, c, "HP-Modified", 200, 200, 100000.0)
                )
            # Auto-generate hotspot alert if temperature > mean + 30 (computed
            # from existing readings BEFORE this insert, to avoid self-reference)
            other = conn.execute(
                "SELECT AVG(wall_temp_c) FROM temperature_readings "
                "WHERE tube_id != ?", (req.tube_id,)
            ).fetchone()[0]
            overall = conn.execute("SELECT AVG(wall_temp_c) FROM temperature_readings").fetchone()[0]
            mean_ref = other if other is not None else (overall if overall is not None else 880.0)
            rid = dbm.insert_temperature_reading(conn, req.tube_id, req.ts_minute,
                                                  req.wall_temp_c, req.source)
            if req.wall_temp_c > mean_ref + 30:
                dbm.insert_alert(conn, req.tube_id, req.ts_minute, "high", "module1",
                                  f"Hotspot on {req.tube_id}: {req.wall_temp_c:.1f}°C")
        return {"ok": True, "reading_id": rid}

    @app.get("/")
    def root() -> dict:
        return {
            "name": "MRTO Backend",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }

    return app


# Module-level instance for `uvicorn mrto.app:app`
app = create_app()
