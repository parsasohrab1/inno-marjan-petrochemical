"""Observability: structured JSON logging + Prometheus metrics.

Provides:
- `get_logger(name)` — returns a configured logger that emits one JSON
  object per line (timestamp, level, logger, message, request_id, fields).
  In dev (MRTO_LOG_PRETTY=1) the logger still prints human-readable lines
  so `docker logs` stays readable; the JSON form is for log aggregation
  (ELK / Loki / CloudWatch).
- `prometheus_client` integration — a /metrics endpoint exposes
  request count, latency histogram, in-flight gauge, plus custom
  business counters (login_success, login_failure, refresh_reuse,
  ingest_count, model_loaded).
- `request_id` — every request gets a short UUID; it is attached to the
  log record and returned in the X-Request-ID header for client-side
  correlation.

The metrics registry is module-level so it survives across requests
within one process. For multi-worker deployments, point Prometheus at
a per-worker port (or use prometheus_client multiprocess mode); the
registry surface stays the same.
"""
from __future__ import annotations

import logging
import json
import os
import sys
import time
import uuid
from contextvars import ContextVar

# ContextVar so log records can pick up the current request_id without
# passing it explicitly through every call site.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(rid: str | None = None) -> str:
    rid = rid or uuid.uuid4().hex[:12]
    _request_id_ctx.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_ctx.get()


# ────────────────────────────────────────────────────────────────────
# JSON log formatter
# ────────────────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record. Extra fields passed via
    `logger.info(msg, extra={...})` are merged into the object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        # Merge any extra= fields the caller attached.
        for k, v in record.__dict__.items():
            if k in ("args", "msg", "levelname", "name", "created",
                     "msecs", "relativeCreated", "exc_info", "exc_text",
                     "filename", "funcName", "levelname", "levelno",
                     "lineno", "module", "pathname", "process",
                     "processName", "thread", "threadName", "stack_info",
                     "taskName"):
                continue
            payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_CONFIGURED = False


def get_logger(name: str = "mrto") -> logging.Logger:
    """Return a configured logger. Idempotent — safe to call from any module."""
    global _CONFIGURED
    log = logging.getLogger(name)
    if _CONFIGURED:
        return log
    level = os.environ.get("MRTO_LOG_LEVEL", "INFO").upper()
    log.setLevel(getattr(logging, level, logging.INFO))
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if os.environ.get("MRTO_LOG_PRETTY") == "1":
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
            ))
        else:
            handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    log.propagate = False
    _CONFIGURED = True
    return log


# ────────────────────────────────────────────────────────────────────
# Prometheus metrics (optional dependency — gracefully degrade)
# ────────────────────────────────────────────────────────────────────
# We use prometheus_client if installed; otherwise we expose a tiny
# in-process counter so the /metrics endpoint still returns *something*
# (useful for dev where you don't want the extra dep). The metrics
# module is imported lazily so the app still boots without the package.
try:
    from prometheus_client import (Counter, Histogram, Gauge,
                                     generate_latest, CONTENT_TYPE_LATEST,
                                     CollectorRegistry, REGISTRY)
    _HAS_PROM = True
except Exception:  # pragma: no cover — exercised only when pkg missing
    _HAS_PROM = False

# Use a fresh registry so test runs don't leak collectors across modules.
METRICS_REGISTRY = CollectorRegistry() if _HAS_PROM else None

if _HAS_PROM:
    REQUESTS = Counter("mrto_http_requests_total",
                       "Total HTTP requests",
                       ["method", "path", "status"],
                       registry=METRICS_REGISTRY)
    LATENCY = Histogram("mrto_http_request_duration_seconds",
                        "Request latency in seconds",
                        ["method", "path"],
                        registry=METRICS_REGISTRY)
    INFLIGHT = Gauge("mrto_http_requests_in_flight",
                     "In-flight HTTP requests",
                     registry=METRICS_REGISTRY)
    LOGIN_SUCCESS = Counter("mrto_login_success_total",
                             "Successful logins",
                             registry=METRICS_REGISTRY)
    LOGIN_FAILURE = Counter("mrto_login_failure_total",
                             "Failed logins",
                             registry=METRICS_REGISTRY)
    REFRESH_REUSE = Counter("mrto_refresh_token_reuse_total",
                             "Refresh-token reuse detected (family revoked)",
                             registry=METRICS_REGISTRY)
    INGEST_COUNT = Counter("mrto_ingest_temperature_total",
                            "Temperature readings ingested",
                            registry=METRICS_REGISTRY)
    MODEL_LOADED = Gauge("mrto_model_loaded",
                          "1 if a model is loaded in cache, 0 otherwise",
                          ["module"],
                          registry=METRICS_REGISTRY)
else:  # pragma: no cover
    REQUESTS = LATENCY = INFLIGHT = LOGIN_SUCCESS = LOGIN_FAILURE = None
    REFRESH_REUSE = INGEST_COUNT = MODEL_LOADED = None


def record_request(method: str, path: str, status: int, duration_s: float) -> None:
    if not _HAS_PROM:
        return
    REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    LATENCY.labels(method=method, path=path).observe(duration_s)


def inc_login_success() -> None:
    if _HAS_PROM: LOGIN_SUCCESS.inc()


def inc_login_failure() -> None:
    if _HAS_PROM: LOGIN_FAILURE.inc()


def inc_refresh_reuse() -> None:
    if _HAS_PROM: REFRESH_REUSE.inc()


def inc_ingest() -> None:
    if _HAS_PROM: INGEST_COUNT.inc()


def set_model_loaded(module: str, loaded: bool) -> None:
    if _HAS_PROM: MODEL_LOADED.labels(module=module).set(1 if loaded else 0)


def metrics_response() -> tuple[bytes, str] | None:
    """Return (body_bytes, content_type) for the /metrics endpoint, or
    None if Prometheus is not installed."""
    if not _HAS_PROM:
        return None
    return generate_latest(METRICS_REGISTRY), CONTENT_TYPE_LATEST
