#!/usr/bin/env python3
"""MRTO synthetic load test — stdlib-only (no locust/k6 dependency).

Logs in once, then hammers the read endpoints at a fixed concurrency for
a fixed duration, and prints throughput + p50/p95/p99 + error rate.

Usage:
    python scripts/loadtest.py --base http://localhost:8080 \
        --duration 60 --concurrency 20 --user admin --pass admin123
    python scripts/loadtest.py --base http://localhost:8080 \
        --duration 120 --concurrency 50 --report loadtest_report.json

The mix (per worker, round-robin):
  GET /api/v1/health          (shallow health, no auth)
  GET /api/v1/tubes           (auth, DB read)
  GET /api/v1/rul            (auth, model or physics fallback)
  GET /api/v1/metrics        (auth, JSON read)
  GET /api/v1/health/deep    (no auth, DB+models check)

Each worker runs requests sequentially in its own thread; concurrency
= number of worker threads. This is a closed-loop model (each worker
waits for a response before sending the next), so the achieved RPS is
bounded by `concurrency / avg_latency`. Use a higher concurrency to
stress more; do not extrapolate to a real user load — the open-loop
shape is different.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from threading import local


_tls = local()


def _login(base: str, user: str, pw: str) -> str:
    body = json.dumps({"username": user, "password": pw}).encode()
    req = urllib.request.Request(
        f"{base}/api/v1/auth/login", data=body,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())["access_token"]


def _get(base: str, path: str, token: str | None) -> tuple[int, float]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base}{path}", headers=headers, method="GET")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read()
            return r.status, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0
    except Exception:
        return 0, time.perf_counter() - t0


def _worker(base: str, token: str, deadline: float,
            path_pool: list[tuple[str, bool]], results: list) -> None:
    """Run requests until deadline; append (path, status, latency_ms) to results."""
    i = 0
    while time.time() < deadline:
        path, needs_auth = path_pool[i % len(path_pool)]
        i += 1
        tok = token if needs_auth else None
        status, lat = _get(base, path, tok)
        results.append((path, status, lat * 1000))


def run(base: str, duration: int, concurrency: int,
        user: str, pw: str) -> dict:
    print(f"logging in as {user} ...")
    token = _login(base, user, pw)
    print(f"token acquired; starting {concurrency} workers for {duration}s")
    path_pool = [
        ("/api/v1/health", False),
        ("/api/v1/tubes", True),
        ("/api/v1/rul", True),
        ("/api/v1/metrics", True),
        ("/api/v1/health/deep", False),
    ]
    results: list = []
    deadline = time.time() + duration
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_worker, base, token, deadline, path_pool, results)
                   for _ in range(concurrency)]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    # Analyze
    n = len(results)
    lats = [r[2] for r in results] if results else [0]
    lats_sorted = sorted(lats)
    statuses = Counter(r[1] for r in results)
    by_path = {}
    for path, status, lat in results:
        by_path.setdefault(path, []).append(lat)
    def pct(p):
        if not lats_sorted: return 0
        i = max(0, min(len(lats_sorted) - 1, int(p * len(lats_sorted))))
        return lats_sorted[i]
    summary = {
        "base": base,
        "duration_s": duration,
        "concurrency": concurrency,
        "total_requests": n,
        "rps": round(n / duration, 1) if duration else 0,
        "latency_ms": {
            "p50": round(pct(0.50), 1),
            "p95": round(pct(0.95), 1),
            "p99": round(pct(0.99), 1),
            "max": round(max(lats) if lats else 0, 1),
        },
        "status_breakdown": dict(statuses),
        "error_rate": round(
            sum(c for s, c in statuses.items() if s == 0 or s >= 500) / max(n, 1),
            4
        ),
        "by_path": {
            p: {
                "n": len(v),
                "p50_ms": round(sorted(v)[len(v) // 2], 1),
                "p95_ms": round(sorted(v)[int(len(v) * 0.95)], 1)
                        if v else 0,
            }
            for p, v in by_path.items()
        },
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="MRTO load test")
    ap.add_argument("--base", default="http://localhost:8080")
    ap.add_argument("--duration", type=int, default=30, help="seconds")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--pass", dest="password", default="admin123")
    ap.add_argument("--report", help="write JSON summary to this file")
    args = ap.parse_args()
    s = run(args.base, args.duration, args.concurrency, args.user, args.password)
    print(json.dumps(s, indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(s, f, indent=2)
        print(f"report written to {args.report}")
    return 0 if s["error_rate"] < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
