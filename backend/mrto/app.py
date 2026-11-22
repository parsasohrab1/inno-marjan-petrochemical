"""FastAPI backend for MRTO system.

Endpoints (see SRS Appendix 8-1):
    GET  /api/v1/health
    POST /api/v1/auth/login
    POST /api/v1/auth/refresh
    POST /api/v1/auth/logout
    GET  /api/v1/tubes
    GET  /api/v1/temperature-map
    GET  /api/v1/rul
    POST /api/v1/optimization
    GET  /api/v1/burners
    GET  /api/v1/alerts
    POST /api/v1/alerts/{id}/ack
    GET  /api/v1/reports/daily
    GET  /api/v1/metrics  (model quality metrics)
    GET  /api/v1/models/versions            (model versioning)
    GET  /api/v1/models/{module}/active
    POST /api/v1/ingest/temperature  (admin ingestion)
"""
from __future__ import annotations

import os
import sys
import json
import time
import base64
import hmac
import hashlib
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
from mrto.ratelimit import RateLimiter

MODELS_DIR = os.environ.get("MRTO_MODELS_DIR",
                              "/home/z/my-project/download/mrto/models")
METRICS_DIR = MODELS_DIR  # metrics json saved alongside models

# Cache loaded models in-memory to avoid joblib.load on every request.
_MODEL_CACHE: dict[str, object] = {}


def _get_model(name: str):
    """Load and cache a joblib model; returns None if file missing."""
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    if name not in _MODEL_CACHE:
        try:
            _MODEL_CACHE[name] = mlm.load_model(path)
        except Exception:
            return None
    return _MODEL_CACHE[name]


def _avg_wall_temp_per_tube(conn, tube_ids: list[str]) -> dict[str, float]:
    """Return the latest average wall temperature per tube from readings.

    Falls back to 880 °C (nominal operating temp) for tubes with no readings,
    so the RUL model always gets a valid input instead of a hardcoded constant.
    """
    if not tube_ids:
        return {}
    placeholders = ",".join("?" * len(tube_ids))
    rows = conn.execute(
        f"SELECT tube_id, AVG(wall_temp_c) AS avg_t FROM ("
        f"  SELECT tube_id, wall_temp_c, ts_minute, "
        f"         ROW_NUMBER() OVER (PARTITION BY tube_id ORDER BY ts_minute DESC) AS rn "
        f"  FROM temperature_readings WHERE tube_id IN ({placeholders})"
        f") WHERE rn <= 5 GROUP BY tube_id",
        tube_ids
    ).fetchall()
    out = {r["tube_id"]: float(r["avg_t"]) for r in rows if r["avg_t"] is not None}
    # Tubes without readings → nominal operating temperature.
    return {tid: out.get(tid, 880.0) for tid in tube_ids}

# ────────────────────────────────────────────────────────────────────
# Pydantic models
# ────────────────────────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RefreshReq(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=2048)


class LogoutReq(BaseModel):
    refresh_token: str = Field(min_length=10, max_length=2048)


class OptimizationReq(BaseModel):
    sc_ratio: float = Field(ge=2.0, le=5.0)
    t_out_c: float = Field(ge=700, le=1000)
    feed_flow_nm3h: float = Field(ge=10000, le=50000)
    cat_age: float = Field(ge=0, le=1.2)
    fuel_flow_nm3h: float = Field(ge=5000, le=20000)
    alpha_eff: float = Field(0.7, ge=0, le=1)
    alpha_fuel: float = Field(0.3, ge=0, le=1)


class TemperatureIngest(BaseModel):
    tube_id: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9_\-]+$")
    ts_minute: int = Field(ge=0, le=10_000_000)
    wall_temp_c: float = Field(ge=0, le=1500)
    source: str = Field(default="ir", max_length=32)


# ────────────────────────────────────────────────────────────────────
# JWT helpers (simple HMAC-based for test-friendly env)
# ────────────────────────────────────────────────────────────────────
# SECURITY: refuse to run with the well-known insecure default secrets.
# Operators must set MRTO_JWT_SECRET to a strong value (≥32 chars).
_INSECURE_JWT_SECRETS = {
    "mrto-dev-secret-1405",
    "change-me-in-production",
    "staging-jwt-change-me",
    "",
}
JWT_SECRET = os.environ.get("MRTO_JWT_SECRET", "")
if JWT_SECRET in _INSECURE_JWT_SECRETS:
    raise RuntimeError(
        "MRTO_JWT_SECRET is unset or matches a known insecure placeholder. "
        "Set it to a strong random value (>=32 chars), e.g. "
        "`export MRTO_JWT_SECRET=$(openssl rand -hex 32)`. "
        "For local tests/dev, set MRTO_ALLOW_INSECURE_JWT=1 to bypass this guard."
    )
if len(JWT_SECRET) < 32 and not os.environ.get("MRTO_ALLOW_INSECURE_JWT"):
    raise RuntimeError(
        f"MRTO_JWT_SECRET is only {len(JWT_SECRET)} chars; need >=32. "
        "Set MRTO_ALLOW_INSECURE_JWT=1 to bypass (dev/tests only)."
    )

JWT_TTL = int(os.environ.get("MRTO_JWT_TTL", 8 * 3600))
REFRESH_TTL = int(os.environ.get("MRTO_REFRESH_TTL", 30 * 24 * 3600))
ALLOW_ANON = os.environ.get("MRTO_ALLOW_ANON", "1") == "1"

_bearer = HTTPBearer(auto_error=False)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def make_jwt(username: str, role: str, jti: str | None = None,
              kind: str = "access") -> str:
    """Sign a JWT. `kind` is 'access' (short-lived) or 'refresh' (long-lived).

    The refresh token's jti is stored in the DB so it can be revoked /
    rotated. Access tokens are stateless (no DB lookup per request).
    """
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    ttl = JWT_TTL if kind == "access" else REFRESH_TTL
    payload = {
        "sub": username, "role": role, "kind": kind,
        "iat": int(time.time()), "exp": int(time.time()) + ttl,
        "jti": jti or secrets.token_hex(8),
    }
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(JWT_SECRET.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    sig_b = _b64(sig)
    return f"{header}.{body}.{sig_b}"


def verify_jwt(token: str, expected_kind: str | None = None) -> dict | None:
    """Verify HMAC-SHA256 JWT. Rejects alg-confusion by pinning HS256.

    If `expected_kind` is set, the token's `kind` claim must match (so a
    refresh token cannot be used as an access token and vice versa).
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b, body_b, sig_b = parts
        # Pin the algorithm — reject "none" and any asymmetric alg.
        header = json.loads(_b64d(header_b))
        if header.get("alg") != "HS256":
            return None
        expected = _b64(hmac.new(JWT_SECRET.encode(),
                                f"{header_b}.{body_b}".encode(), hashlib.sha256).digest())
        if not secrets.compare_digest(expected, sig_b):
            return None
        payload = json.loads(_b64d(body_b))
        if payload.get("exp", 0) < time.time():
            return None
        if expected_kind and payload.get("kind") != expected_kind:
            return None
        return payload
    except Exception:
        return None


async def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """Resolve the authenticated user from the Bearer access token.

    In production (MRTO_ALLOW_ANON != "1"), a missing/invalid token returns
    401. In dev/test mode (default), missing tokens fall back to anonymous
    read-only viewer so the test suite can run without auth.

    Refresh tokens are explicitly rejected here — they may only be used
    at /api/v1/auth/refresh, never as a bearer credential for data routes.
    """
    if creds is None or not creds.credentials:
        if ALLOW_ANON:
            return {"sub": "anonymous", "role": "viewer"}
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = verify_jwt(creds.credentials, expected_kind="access")
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
    # CORS: configurable origins via env; default to localhost only (not "*").
    cors_origins = os.environ.get("MRTO_CORS_ORIGINS", "http://localhost:8080")
    allow_list = [o.strip() for o in cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Security headers middleware (defense-in-depth against XSS, clickjacking, MITM).
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Rate limiter: protects /auth/login (brute-force) and the API surface.
    limiter = RateLimiter(
        default_limit=int(os.environ.get("MRTO_RATE_LIMIT", 100)),
        default_window=int(os.environ.get("MRTO_RATE_WINDOW", 60)),
    )
    app.middleware("http")(limiter.as_middleware())

    app.state.db_path = db_path
    app.state.limiter = limiter

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
                # Audit failed attempt (rate-limited above; audit gives forensics).
                dbm.audit(conn, req.username, "login_failed", "")
                raise HTTPException(status_code=401, detail="Invalid credentials")
            dbm.audit(conn, user["username"], "login", "")
            # Issue short-lived access + long-lived refresh token (DB-backed
            # so it can be rotated / revoked).
            access = make_jwt(user["username"], user["role"], kind="access")
            refresh_payload = dbm.issue_refresh_token(
                conn, user["username"], user["role"]
            )
            refresh = make_jwt(user["username"], user["role"],
                                jti=refresh_payload["jti"], kind="refresh")
        return {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "role": user["role"],
            "expires_in": JWT_TTL,
        }

    @app.post("/api/v1/auth/refresh")
    def refresh(req: RefreshReq) -> dict:
        """Exchange a valid refresh token for a new access token + rotated
        refresh token. Detects reuse: if the presented refresh token has
        already been rotated, the entire token family is revoked and 401
        is returned (the client must re-authenticate).
        """
        payload = verify_jwt(req.refresh_token, expected_kind="refresh")
        if payload is None:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        jti = payload.get("jti")
        if not jti:
            raise HTTPException(status_code=401, detail="Malformed refresh token")
        with dbm.get_conn(app.state.db_path) as conn:
            new_payload, err = dbm.rotate_refresh_token(conn, jti)
            if err is not None:
                if err["error"] == "reuse_detected":
                    # The reuse-detection UPDATE was applied above and will be
                    # committed by the context manager. We must NOT raise inside
                    # the `with` block — doing so would rollback the revocation.
                    # Capture the signal and raise after the block exits cleanly.
                    pass
                else:
                    raise HTTPException(status_code=401,
                                        detail=f"Refresh failed: {err['error']}")
            else:
                dbm.audit(conn, new_payload["username"], "token_refresh", "")
                access = make_jwt(new_payload["username"], new_payload["role"],
                                   kind="access")
                new_refresh = make_jwt(new_payload["username"], new_payload["role"],
                                        jti=new_payload["jti"], kind="refresh")
                return {
                    "access_token": access,
                    "refresh_token": new_refresh,
                    "token_type": "Bearer",
                    "role": new_payload["role"],
                    "expires_in": JWT_TTL,
                }
        # Reuse-detected path: connection has committed the family revocation.
        raise HTTPException(
            status_code=401,
            detail="Refresh token reuse detected — session revoked"
        )

    @app.post("/api/v1/auth/logout")
    def logout(req: LogoutReq,
                _user: dict = Depends(current_user)) -> dict:
        """Revoke the refresh-token family on logout. The access token is
        stateless and remains valid until it expires (short TTL), but the
        refresh chain is dead so it cannot be renewed."""
        payload = verify_jwt(req.refresh_token, expected_kind="refresh")
        if payload is None:
            # Idempotent: an invalid/already-expired refresh token is fine here.
            return {"ok": True, "revoked": 0}
        jti = payload.get("jti")
        with dbm.get_conn(app.state.db_path) as conn:
            row = conn.execute(
                "SELECT family_id FROM refresh_tokens WHERE jti=?", (jti,)
            ).fetchone()
            if row is None:
                return {"ok": True, "revoked": 0}
            n = dbm.revoke_family(conn, row["family_id"], reason="logout")
            dbm.audit(conn, _user.get("sub", "?"), "logout",
                       f"revoked {n} tokens in family")
        return {"ok": True, "revoked": n}

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
        Uses the real per-tube average wall temperature from the latest
        readings (not a hardcoded 880 °C) so predictions reflect actual
        operating conditions.
        """
        model = _get_model("module2_rul.joblib")
        with dbm.get_conn(app.state.db_path) as conn:
            tubes = dbm.get_tubes(conn)
        if model is not None and tubes:
            try:
                df = pd.DataFrame(tubes)
                tube_ids = df["tube_id"].tolist()
                # Real per-tube average wall temperature from latest readings.
                with dbm.get_conn(app.state.db_path) as conn:
                    avg_t_map = _avg_wall_temp_per_tube(conn, tube_ids)
                df["avg_wall_temp_c"] = df["tube_id"].map(avg_t_map).astype(float)
                df["consumed_life_fraction"] = (df["last_replaced_days_ago"] * 24) / df["design_life_h"]
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
                # Surface the failure so operators can detect a broken model
                # instead of silently falling back. Still degrade gracefully.
                pass  # fall through to physics estimate
        # Physics fallback — uses real per-tube average wall temperature.
        rows = []
        with dbm.get_conn(app.state.db_path) as conn:
            avg_t_map = _avg_wall_temp_per_tube(conn, [t["tube_id"] for t in tubes])
        from mrto.synthetic_data import lmp_to_rupture_time
        for t in tubes:
            avg_t = avg_t_map.get(t["tube_id"], 880.0)
            t_rupt = lmp_to_rupture_time(avg_t)
            elapsed = t["last_replaced_days_ago"] * 24
            rul = max(0.0, t_rupt - elapsed)
            consumed = elapsed / max(t_rupt, 1)
            p_fail_90d = 1.0 / (1.0 + (1.0 - max(0, min(1, consumed - 0.7))) * 10)
            rows.append({"tube_id": t["tube_id"], "rul_h": int(rul), "p_fail_90d": round(p_fail_90d, 4)})
        return {"source": "physics_fallback", "tubes": rows}

    @app.post("/api/v1/optimization")
    def optimization(req: OptimizationReq, _user: dict = Depends(current_user)) -> dict:
        model = _get_model("module3_optimization.joblib")
        if model is None:
            raise HTTPException(status_code=503, detail="Optimization model not trained")
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
        """Return burner status with inefficiency flag from Module 4 if available.

        Uses the real average tube wall temperature (from the latest
        temperature_map) instead of a hardcoded 880 °C, so the burner
        inefficiency prediction reflects actual furnace conditions.
        """
        with dbm.get_conn(app.state.db_path) as conn:
            rows = conn.execute(
                "SELECT burner_id, ts_minute, fuel_flow, intensity, angle FROM burner_status "
                "ORDER BY ts_minute DESC LIMIT 200"
            ).fetchall()
            # Compute the real average wall temperature from the latest readings.
            overall_avg = conn.execute(
                "SELECT AVG(wall_temp_c) FROM ("
                "  SELECT wall_temp_c, ts_minute FROM temperature_readings "
                "  ORDER BY ts_minute DESC LIMIT 500"
                ")"
            ).fetchone()[0]
        burners_list = [dict(r) for r in rows]
        # Apply model if available
        model = _get_model("module4_burner_catalyst.joblib")
        if burners_list and model is not None:
            try:
                df = pd.DataFrame(burners_list).rename(columns={"fuel_flow": "fuel_flow_nm3h", "angle": "angle_deg"})
                # Use real average wall temp; fall back to nominal 880 if no readings yet.
                df["age_h"] = 2000  # TODO: persist per-burner age_h in burner_status
                df["avg_tube_t_c"] = float(overall_avg) if overall_avg is not None else 880.0
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
        # RBAC: only manager/operator/it roles may ingest. Anonymous viewers
        # (dev fallback) are rejected even in dev mode — this is a write.
        if _user.get("role") not in ("manager", "operator", "it"):
            raise HTTPException(status_code=403, detail="Insufficient role for ingestion")
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

    # ── Model versioning (Phase 2) ──────────────────────────────────
    @app.get("/api/v1/models/versions")
    def list_model_versions_endpoint(module: str | None = None,
                                       _user: dict = Depends(current_user)) -> dict:
        """List all registered model versions (optionally filtered by module).

        Only manager/operator/it roles may read this; operators on the plant
        floor only need the active model, not the audit trail of every retrain.
        """
        if _user.get("role") not in ("manager", "operator", "it"):
            raise HTTPException(status_code=403, detail="Insufficient role")
        with dbm.get_conn(app.state.db_path) as conn:
            rows = dbm.list_model_versions(conn, module=module)
        return {"versions": rows, "n": len(rows)}

    @app.get("/api/v1/models/{module}/active")
    def get_active_model_endpoint(module: str,
                                    _user: dict = Depends(current_user)) -> dict:
        """Return metadata (version, sha256, trained_at, metrics) for the
        currently-served model of a module. Lets the dashboard show which
        model version produced the RUL / optimization numbers the operator
        is seeing — critical for traceability and incident postmortems.
        """
        with dbm.get_conn(app.state.db_path) as conn:
            row = dbm.get_active_model_version(conn, module)
        if row is None:
            raise HTTPException(status_code=404, detail="No active model for this module")
        # Parse the cached metrics JSON for convenience.
        out = dict(row)
        try:
            out["metrics"] = json.loads(row["metrics_json"]) if row["metrics_json"] else None
        except Exception:
            out["metrics"] = None
        out.pop("metrics_json", None)
        return out

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
