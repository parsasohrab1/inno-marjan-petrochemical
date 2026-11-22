"""In-memory token-bucket rate limiter middleware.

Protects /api/v1/auth/login (brute-force) and the global API surface
(abuse / DoS) without an external dependency. Buckets are keyed by
client IP (X-Forwarded-For first hop or request.client.host).

The implementation is a simple sliding-window counter per (route, ip)
pair, pruned periodically. It is intentionally process-local — for a
multi-worker deployment behind a reverse proxy, swap this for
`slowapi` or a Redis-backed limiter; the interface (`RateLimiter`)
stays the same so app.py need not change.
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse


class RateLimiter:
    """Sliding-window rate limiter with per-route overrides."""

    def __init__(self, default_limit: int = 100, default_window: int = 60):
        # (route_prefix, ip) -> list of hit timestamps.
        self._buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.default_limit = default_limit
        self.default_window = default_window
        # Per-route overrides: prefix -> (limit, window_s).
        self.route_limits = {
            "/api/v1/auth/login": (10, 60),       # 10 logins / minute / IP
            "/api/v1/auth/refresh": (20, 60),     # 20 refreshes / minute / IP
            "/api/v1/ingest/temperature": (60, 60),  # 60 writes / minute / IP
        }
        self._last_prune = time.time()

    def _client_ip(self, request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _limit_for(self, path: str) -> tuple[int, int]:
        for prefix, (lim, win) in self.route_limits.items():
            if path.startswith(prefix):
                return lim, win
        return self.default_limit, self.default_window

    def _prune(self, now: float) -> None:
        # Drop empty buckets every ~5 minutes to bound memory.
        if now - self._last_prune < 300:
            return
        self._last_prune = now
        stale = [k for k, hits in self._buckets.items()
                 if not hits or hits[-1] < now - 3600]
        for k in stale:
            del self._buckets[k]

    def check(self, request: Request) -> tuple[bool, dict]:
        """Returns (allowed, headers). headers has limit/remaining/reset."""
        now = time.time()
        self._prune(now)
        ip = self._client_ip(request)
        limit, window = self._limit_for(request.url.path)
        key = (request.url.path, ip)
        hits = self._buckets[key]
        # Drop hits outside the window.
        cutoff = now - window
        self._buckets[key] = [h for h in hits if h > cutoff]
        hits = self._buckets[key]
        remaining = max(0, limit - len(hits))
        if len(hits) >= limit:
            reset = int(hits[0] + window)
            return False, {"X-RateLimit-Limit": str(limit),
                           "X-RateLimit-Remaining": "0",
                           "X-RateLimit-Reset": str(reset),
                           "Retry-After": str(max(1, reset - int(now)))}
        hits.append(now)
        reset = int(hits[0] + window)
        return True, {"X-RateLimit-Limit": str(limit),
                      "X-RateLimit-Remaining": str(remaining),
                      "X-RateLimit-Reset": str(reset)}

    def as_middleware(self):
        """Return an ASGI/Starlette-compatible middleware callable."""
        limiter = self

        async def middleware(request: Request, call_next):
            # Health check is exempt (used by Docker HEALTHCHECK).
            if request.url.path == "/api/v1/health":
                return await call_next(request)
            allowed, headers = limiter.check(request)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests",
                             "retry_after_s": int(headers.get("Retry-After", 60))},
                    headers=headers,
                )
            response: Response = await call_next(request)
            for k, v in headers.items():
                response.headers[k] = v
            return response

        return middleware
