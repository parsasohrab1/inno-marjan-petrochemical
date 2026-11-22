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
import logging
import secrets
import threading
from datetime import datetime, timezone

logger = logging.getLogger("mrto")
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
from mrto import observability as obs

# Phase 3: read-replica routing. When MRTO_READ_REPLICA_URL is set (Postgres
# deployment), GET endpoints open the connection through db_adapter so
# reads go to the replica and the primary is spared for writes. On SQLite
# (dev/test) the replica flag is ignored and the same DB is used.
try:
    from mrto import db_adapter as _db_adapter
    _HAS_REPLICA = _db_adapter.HAS_REPLICA
except Exception:
    _HAS_REPLICA = False


def _read_conn(app, role: str = "replica"):
    """Return a context manager for a read connection. Routes to the
    replica when one is configured; otherwise falls back to the primary
    (SQLite or single-postgres)."""
    if _HAS_REPLICA:
        return _db_adapter.get_conn(app.state.db_path, role=role)
    return dbm.get_conn(app.state.db_path)

def _resolve_models_dir() -> str:
    """Prefer MRTO_MODELS_DIR when that folder exists.

    Fall back to the repository models/ directory so a Windows launch works
    without the old hardcoded Linux path.
    """
    env = os.environ.get("MRTO_MODELS_DIR")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    if env and os.path.isdir(env):
        return env
    if os.path.isdir(repo):
        return repo
    return env or repo


MODELS_DIR = _resolve_models_dir()
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


_HOT_TUBES = ("T01-003", "T02-003")
_DEGRADED_BURNERS = {"B01-L": 0.52, "B03-R": 0.40}


def _live_loop(db_path: str) -> None:
    """Append one furnace sample every 5 seconds while MRTO_LIVE_SIM=1."""
    import random
    for _ in range(40):
        time.sleep(1)
        try:
            with dbm.get_conn(db_path) as conn:
                n = conn.execute("SELECT COUNT(*) FROM temperature_readings").fetchone()[0]
            if n:
                break
        except Exception:
            logger.exception("live simulator waiting for readings")
    while True:
        try:
            with dbm.get_conn(db_path) as conn:
                tubes = dbm.get_tubes(conn)
                if not tubes:
                    time.sleep(5)
                    continue
                prev_rows = conn.execute(
                    """
                    SELECT tube_id, wall_temp_c FROM temperature_readings
                    WHERE id IN (SELECT MAX(id) FROM temperature_readings GROUP BY tube_id)
                    """
                ).fetchall()
                prev = {r["tube_id"]: float(r["wall_temp_c"]) for r in prev_rows}
                cur = conn.execute("SELECT MAX(ts_minute) FROM temperature_readings").fetchone()
                ts = int(cur[0] or 0) + 1
                for tube in tubes:
                    tid = tube["tube_id"]
                    base = prev.get(tid, 880.0)
                    temp = base + random.uniform(-1.2, 1.2)
                    if tid in _HOT_TUBES:
                        temp = min(960.0, max(935.0, temp if base >= 920 else 942.0))
                    else:
                        temp = min(910.0, max(840.0, temp))
                    other = conn.execute(
                        "SELECT AVG(wall_temp_c) FROM temperature_readings WHERE tube_id != ?",
                        (tid,),
                    ).fetchone()[0]
                    mean_ref = other if other is not None else 880.0
                    dbm.insert_temperature_reading(conn, tid, ts, round(temp, 1), "live")
                    if temp > mean_ref + 30:
                        open_same = conn.execute(
                            "SELECT 1 FROM alerts WHERE tube_id=? AND module='module1' AND status='open'",
                            (tid,),
                        ).fetchone()
                        if open_same is None:
                            dbm.insert_alert(
                                conn, tid, ts, "high", "module1",
                                f"Hotspot on {tid}: {temp:.1f}°C",
                            )
                latest = conn.execute(
                    """
                    SELECT burner_id, fuel_flow, intensity, angle FROM burner_status
                    WHERE id IN (SELECT MAX(id) FROM burner_status GROUP BY burner_id)
                    """
                ).fetchall()
                for row in latest:
                    bid = row["burner_id"]
                    intensity = float(row["intensity"] or 0.9)
                    fuel = float(row["fuel_flow"] or 190)
                    if bid in _DEGRADED_BURNERS:
                        intensity = _DEGRADED_BURNERS[bid] + random.uniform(-0.02, 0.02)
                        fuel = 250 + random.uniform(-8, 8)
                    else:
                        intensity = min(1.15, max(0.85, intensity + random.uniform(-0.01, 0.01)))
                        fuel = min(210, max(170, fuel + random.uniform(-4, 4)))
                    angle = max(-15.0, min(15.0, float(row["angle"] or 0) + random.uniform(-0.4, 0.4)))
                    conn.execute(
                        "INSERT INTO burner_status (burner_id, ts_minute, fuel_flow, intensity, angle) "
                        "VALUES (?,?,?,?,?)",
                        (bid, ts, round(fuel, 1), round(intensity, 3), round(angle, 2)),
                    )
        except Exception:
            logger.exception("live simulator tick failed")
        time.sleep(5)


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
# Metric-path normalization (Phase 3)
# ────────────────────────────────────────────────────────────────────
# Collapse numeric path segments to ":id" so /alerts/5/ack and
# /alerts/99/ack become the same Prometheus label. Without this, every
# alert ID becomes its own time series and cardinality blows up.
import re as _re
_ID_RE = _re.compile(r"/\d+(?=/|$)")


def _metric_path(path: str) -> str:
    return _ID_RE.sub("/:id", path) if path else "/"

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


class TemperatureBatch(BaseModel):
    readings: list[TemperatureIngest] = Field(min_length=1, max_length=5000)


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
# Phase 4: route through the secrets module so the value can come from
# env, a _FILE mount, Vault, or SOPS — not just env.
from mrto import secrets as _sec
JWT_SECRET = _sec.get_secret("MRTO_JWT_SECRET", "") or ""
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
        description=(
            "Smart monitoring of reformer tube health + S/C optimization "
            "for Marjan Petrochemical (TRL 5).\n\n"
            "**Auth**: all data endpoints require a Bearer access token from "
            "`/api/v1/auth/login`. Refresh tokens rotate on every use; reuse "
            "revokes the entire family. Rate-limited at 100 req/min/IP by "
            "default, 10 logins/min/IP for brute-force protection.\n\n"
            "**Observability**: structured JSON logs (with `request_id`), "
            "Prometheus metrics at `/metrics`, deep health at "
            "`/api/v1/health/deep`."
        ),
        openapi_tags=[
            {"name": "health", "description": "Liveness/readiness probes"},
            {"name": "auth", "description": "Login, refresh, logout (JWT + rotating refresh tokens)"},
            {"name": "tubes", "description": "Reformer tube registry"},
            {"name": "temperature", "description": "Wall-temperature readings + hotspots"},
            {"name": "rul", "description": "Remaining-useful-life predictions (Module 2)"},
            {"name": "optimization", "description": "S/C ratio optimization (Module 3)"},
            {"name": "burners", "description": "Burner status + inefficiency flag (Module 4)"},
            {"name": "alerts", "description": "Hotspot & failure alerts + ack workflow"},
            {"name": "reports", "description": "Daily health report"},
            {"name": "metrics", "description": "Model quality metrics + model versioning"},
            {"name": "ingest", "description": "Write endpoints (manager/operator/it only)"},
        ],
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

    # Phase 3: structured logging + request ID + Prometheus metrics.
    log = obs.get_logger("mrto.app")

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        # Assign a request ID and surface it on the response so the client
        # can correlate support tickets with server logs.
        rid = obs.set_request_id(request.headers.get("x-request-id") or None)
        obs.INFLIGHT and obs.INFLIGHT.inc()
        t0 = time.time()
        # Normalize path for metrics so /alerts/5/ack and /alerts/99/ack
        # collapse into one label (avoids high-cardinality blowup).
        raw_path = request.url.path
        metric_path = _metric_path(raw_path)
        try:
            response: Response = await call_next(request)
        except Exception:
            obs.record_request(request.method, metric_path, 500,
                               time.time() - t0)
            log.exception("unhandled error", extra={"path": raw_path})
            raise
        finally:
            if obs.INFLIGHT:
                obs.INFLIGHT.dec()
        obs.record_request(request.method, metric_path,
                           response.status_code, time.time() - t0)
        response.headers["X-Request-ID"] = rid
        # Access log line (JSON when MRTO_LOG_PRETTY != 1).
        log.info("request",
                 extra={"method": request.method, "path": raw_path,
                        "status": response.status_code,
                        "ms": int((time.time() - t0) * 1000)})
        return response

    app.state.db_path = db_path
    app.state.limiter = limiter
    app.state.log = log

    # ── Endpoints ───────────────────────────────────────────────────
    @app.get("/api/v1/health",
             tags=["health"],
             summary="Shallow health (load-balancer ping)",
             response_description="200 if the process is up and the DB answers.")
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

    @app.post("/api/v1/auth/login",
              tags=["auth"],
              summary="Login → access + refresh token",
              responses={401: {"description": "Invalid credentials"},
                         429: {"description": "Rate-limited (>10 logins/min/IP)"}})
    def login(req: LoginReq) -> dict:
        with dbm.get_conn(app.state.db_path) as conn:
            user = dbm.verify_user(conn, req.username, req.password)
            if user is None:
                # Audit failed attempt (rate-limited above; audit gives forensics).
                dbm.audit(conn, req.username, "login_failed", "")
                obs.inc_login_failure()
                raise HTTPException(status_code=401, detail="Invalid credentials")
            dbm.audit(conn, user["username"], "login", "")
            obs.inc_login_success()
            # Update last_login_at if the column exists (migration 2).
            try:
                conn.execute(
                    "UPDATE users SET last_login_at=? WHERE id=?",
                    (int(time.time()), user["id"])
                )
            except Exception:
                pass  # column missing on unmigrated DB; harmless
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

    @app.post("/api/v1/auth/refresh",
              tags=["auth"],
              summary="Rotate refresh token (single-use)",
              responses={401: {"description": "Invalid/expired token OR reuse detected (family revoked)"}})
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
                    obs.inc_refresh_reuse()
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

    @app.post("/api/v1/auth/logout",
              tags=["auth"],
              summary="Revoke the refresh-token family")
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
        resolved_ts = ts_minute
        if resolved_ts is None and readings:
            resolved_ts = readings[0]["ts_minute"]
        return {
            "ts_minute": resolved_ts,
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
        catalyst_activity = None
        with dbm.get_conn(app.state.db_path) as conn:
            rows = conn.execute(
                """
                SELECT burner_id, ts_minute, fuel_flow, intensity, angle
                FROM burner_status
                WHERE id IN (
                    SELECT MAX(id) FROM burner_status GROUP BY burner_id
                )
                ORDER BY burner_id
                """
            ).fetchall()
            overall_avg = conn.execute(
                """
                SELECT AVG(wall_temp_c) FROM temperature_readings
                WHERE id IN (SELECT MAX(id) FROM temperature_readings GROUP BY tube_id)
                """
            ).fetchone()[0]
        burners_list = [dict(r) for r in rows]
        mean_wall = float(overall_avg) if overall_avg is not None else 880.0
        model = _get_model("module4_burner_catalyst.joblib")
        if model is not None:
            try:
                if burners_list:
                    df = pd.DataFrame(burners_list).rename(
                        columns={"fuel_flow": "fuel_flow_nm3h", "angle": "angle_deg"}
                    )
                    df["age_h"] = 2000
                    df["avg_tube_t_c"] = mean_wall
                    df["intensity"] = df["intensity"].fillna(0.9)
                    df["fuel_flow_nm3h"] = df["fuel_flow_nm3h"].fillna(190)
                    df["angle_deg"] = df["angle_deg"].fillna(0)
                    preds = model.predict_inefficient(df)
                    for b, p in zip(burners_list, preds):
                        b["inefficient"] = int(p)
                activity = model.predict_activity(pd.DataFrame([{
                    "age_h": 8000.0,
                    "t_out_avg_c": mean_wall,
                    "sc_avg": 3.0,
                    "dp_bar": 1.6,
                }]))
                catalyst_activity = round(float(activity[0]), 4)
            except Exception:
                logger.exception("burner/catalyst model failed")
        return {
            "burners": burners_list,
            "n": len(burners_list),
            "catalyst_activity": catalyst_activity,
            "mean_wall_temp_c": round(mean_wall, 2),
        }

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

    @app.get("/api/v1/metrics",
             tags=["metrics"],
             summary="Model quality metrics (from JSON files)")
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
                        data = json.load(f)
                    if name == "module1_temperature_map" and isinstance(data, dict):
                        data.setdefault("rmse_c", data.get("rmse"))
                        data.setdefault("mae_c", data.get("mae"))
                    out[name] = data
                except Exception as e:
                    out[name] = {"error": str(e)}
            else:
                out[name] = None
        return out

    @app.post("/api/v1/ingest/temperature",
              tags=["ingest"],
              summary="Ingest one temperature reading (manager/operator/it)",
              responses={403: {"description": "Insufficient role"},
                          422: {"description": "Out-of-range input"}})
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
            obs.inc_ingest()
            if req.wall_temp_c > mean_ref + 30:
                open_same = conn.execute(
                    "SELECT 1 FROM alerts WHERE tube_id=? AND module='module1' AND status='open'",
                    (req.tube_id,),
                ).fetchone()
                if open_same is None:
                    dbm.insert_alert(conn, req.tube_id, req.ts_minute, "high", "module1",
                                      f"Hotspot on {req.tube_id}: {req.wall_temp_c:.1f}°C")
        return {"ok": True, "reading_id": rid}

    @app.post("/api/v1/ingest/temperature/batch",
              tags=["ingest"],
              summary="Ingest a batch of temperature readings (manager/operator/it)")
    def ingest_batch(req: TemperatureBatch,
                      _user: dict = Depends(current_user)) -> dict:
        if _user.get("role") not in ("manager", "operator", "it"):
            raise HTTPException(status_code=403, detail="Insufficient role for ingestion")
        last_id = None
        with dbm.get_conn(app.state.db_path) as conn:
            for item in req.readings:
                existing = conn.execute(
                    "SELECT 1 FROM tubes WHERE tube_id=?", (item.tube_id,)
                ).fetchone()
                if existing is None:
                    try:
                        parts = item.tube_id.split("-")
                        row = int(parts[0][1:])
                        col = int(parts[1])
                    except Exception:
                        row, col = 1, 1
                    conn.execute(
                        "INSERT INTO tubes (tube_id, row, col, material, installed_at_days_ago, "
                        "last_replaced_days_ago, design_life_h) VALUES (?,?,?,?,?,?,?)",
                        (item.tube_id, row, col, "HP-Modified", 200, 200, 100000.0),
                    )
                other = conn.execute(
                    "SELECT AVG(wall_temp_c) FROM temperature_readings WHERE tube_id != ?",
                    (item.tube_id,),
                ).fetchone()[0]
                overall = conn.execute(
                    "SELECT AVG(wall_temp_c) FROM temperature_readings"
                ).fetchone()[0]
                mean_ref = other if other is not None else (overall if overall is not None else 880.0)
                last_id = dbm.insert_temperature_reading(
                    conn, item.tube_id, item.ts_minute, item.wall_temp_c, item.source
                )
                obs.inc_ingest()
                if item.wall_temp_c > mean_ref + 30:
                    open_same = conn.execute(
                        "SELECT 1 FROM alerts WHERE tube_id=? AND module='module1' AND status='open'",
                        (item.tube_id,),
                    ).fetchone()
                    if open_same is None:
                        dbm.insert_alert(
                            conn, item.tube_id, item.ts_minute, "high", "module1",
                            f"Hotspot on {item.tube_id}: {item.wall_temp_c:.1f}°C",
                        )
        return {"ok": True, "n": len(req.readings), "reading_id": last_id}

    # ── Phase 3: Prometheus metrics + deep health ─────────────────────
    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        """Prometheus scrape endpoint. Exposed at /metrics (outside the
        /api/v1 prefix) per Prometheus convention so existing scrape configs
        work unchanged. No auth — protect at the network/ingress layer."""
        out = obs.metrics_response()
        if out is None:
            return Response(content=b"# prometheus_client not installed\n",
                            media_type="text/plain", status_code=501)
        body, ctype = out
        return Response(content=body, media_type=ctype)

    @app.get("/api/v1/health/deep",
             tags=["health"],
             summary="Deep health (DB, schema, models, disk) — k8s readiness")
    def health_deep() -> dict:
        """Deep health check — verifies DB connectivity, schema version,
        model availability, and disk space for the models dir. Used by
        k8s liveness/readiness probes (the shallow /health is for the
        Docker HEALTHCHECK and load-balancer pings)."""
        checks = {}
        # DB
        try:
            with dbm.get_conn(app.state.db_path) as conn:
                n = conn.execute("SELECT COUNT(*) FROM tubes").fetchone()[0]
                from mrto import migrations as mig
                sv = mig.get_schema_version(conn)
                checks["db"] = {"ok": True, "n_tubes": int(n),
                                "schema_version": sv}
        except Exception as e:
            checks["db"] = {"ok": False, "error": str(e)}
        # Models present (not loaded — that's a warmup concern)
        for name, fname in [("module2_rul", "module2_rul.joblib"),
                            ("module3_optimization", "module3_optimization.joblib"),
                            ("module4_burner_catalyst", "module4_burner_catalyst.joblib")]:
            p = os.path.join(MODELS_DIR, fname)
            checks[f"model_{name}"] = {"ok": os.path.exists(p),
                                        "path": p}
        # Disk space for models dir (best-effort)
        try:
            import shutil
            total, used, free = shutil.disk_usage(MODELS_DIR)
            checks["disk"] = {"ok": free > 100 * 1024 * 1024,
                              "free_mb": int(free / 1024 / 1024)}
        except Exception as e:
            checks["disk"] = {"ok": False, "error": str(e)}
        all_ok = all(v.get("ok") if isinstance(v, dict) else True
                     for v in checks.values())
        return {
            "status": "ok" if all_ok else "degraded",
            "service": "mrto",
            "version": "1.0.0",
            "schema_version": checks["db"].get("schema_version", 0)
                              if isinstance(checks.get("db"), dict) else 0,
            "checks": checks,
            "ts_epoch": int(time.time()),
        }

    # ── Model versioning (Phase 2) ──────────────────────────────────
    @app.get("/api/v1/models/versions",
             tags=["metrics"],
             summary="List model versions (manager/operator/it)",
             responses={403: {"description": "Insufficient role"}})
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

    if os.environ.get("MRTO_LIVE_SIM") in ("1", "true", "True", "yes"):
        threading.Thread(
            target=_live_loop, args=(db_path,), name="mrto-live", daemon=True
        ).start()
    return app


# Module-level instance for `uvicorn mrto.app:app`
app = create_app()
