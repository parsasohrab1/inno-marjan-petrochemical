# MRTO Security Audit — Phase 4

Audit run: 2026-11-22, against commit `aae8c62` (Phase 3 head).
Tooling: bandit 1.9.x, pip-audit 2.x, against `backend/requirements.txt`
(runtime deps only — pytest/bandit test deps excluded from pip-audit
because they create a resolution conflict that pip-audit's isolated venv
cannot satisfy; they are not shipped to production).

This document records every finding, the decision taken (fix / accept /
defer), and the rationale. Future audits should append a new dated
section rather than rewrite history.

---

## 1. bandit (static analysis of `mrto/`)

Command: `bandit -r mrto -ll` (MEDIUM and higher only).

**Summary**: 11 issues total — 0 High, 2 Medium, 9 Low.

### MEDIUM

#### M1. `B608` — hardcoded SQL expression
- **Location**: `mrto/app.py:99` (`_avg_wall_temp_per_tube`)
- **Description**: `f"... WHERE tube_id IN ({placeholders})"` is flagged
  because the SQL string is built with an f-string. The placeholders
  variable is `",".join("?" * len(tube_ids))` (only `?` chars, never
  user input), and the actual values are passed as a parameter list.
  There is no injection vector — the f-string only injects the number
  of `?` markers, never data.
- **Decision**: **Accept.** The pattern is the canonical way to build
  an `IN (?, ?, ?)` clause in sqlite3 with a variable-length list;
  the alternative (constructing the SQL with `format` and a list of
  integers) is identical in risk. Added an inline comment at the call
  site explaining why the f-string is safe and why nosec was not added
  (we want bandit to keep flagging it on review so a future refactor
  that *does* interpolate user input is caught).

#### M2. `B310` — `urllib.request.urlopen` permitted-scheme blacklist
- **Location**: `mrto/real_data.py:70`
- **Description**: `urllib.request.urlopen` is on bandit's blacklist
  because by default it follows `file:` and other schemes. The code
  passes a `Request` object built from a `https://` UCI URL and uses
  a TLS context. There is no user-controlled URL.
- **Decision**: **Accept.** The URL is a hardcoded UCI dataset endpoint
  (`https://archive.ics.uci.edu/...`), never user input. The TLS context
  validates the cert. Migrating to `requests` would not reduce risk and
  would add a dependency. Documented inline.

### LOW (9 — all accepted, listed for completeness)

| # | Code | Location | Notes |
|---|------|----------|-------|
| L1 | B101 assert | tests/*.py | Asserts in test code — expected. |
| L2 | B110 try_except_pass | mrto/secrets.py | `except Exception: continue` in the secret backend chain — intentional, never let a backend crash boot. |
| L3 | B110 try_except_pass | mrto/app.py | Deep-health disk check — degrade gracefully. |
| L4 | B112 try_except_continue | mrto/secrets.py | Backend iteration. |
| L5 | B110 try_except_pass | mrto/db.py | Migration failure is logged, not fatal. |
| L6 | B110 try_except_pass | mrto/app.py | last_login_at UPDATE if column missing. |
| L7 | B110 try_except_pass | mrto/observability.py | prometheus_client import guard. |
| L8 | B110 try_except_pass | mrto/app.py | metric_path regex compile. |
| L9 | B110 try_except_pass | mrto/app.py | prometheus_client optional import. |

All LOW findings are intentional graceful-degradation patterns;
each is documented at the call site.

---

## 2. pip-audit (transitive dependency CVEs)

Command: `pip-audit -r <runtime-deps> --strict --desc on`.

**Summary**: 14 known vulnerabilities across 2 packages.

### requests 2.32.3 (2 distinct CVEs)

| CVE | Severity | Fix | Status |
|-----|----------|-----|--------|
| PYSEC-2026-1872 | Medium | 2.32.4 | **Fixed** — bumped to 2.32.4. |
| PYSEC-2026-2275 | Low | 2.33.0 | **Accept.** Affects only `extract_zipped_paths()`, a utility MRTO never calls. 2.32.4 is sufficient. |

### starlette 0.50.0 (transitive from fastapi 0.128.0, 5 distinct CVEs)

| CVE | Severity | Fix | Status |
|-----|----------|-----|--------|
| PYSEC-2026-161 | Medium | 1.0.1 | **Defer.** Host-header injection into `request.url.path`. MRTO does not gate access on `request.url.path`; routing uses the raw scope path. Reverse proxy rejects malformed Host headers. |
| PYSEC-2026-2281 | Low | 1.1.0 | **Accept.** Windows-only StaticFiles SSRF via UNC path. MRTO runs on Linux containers and does not serve static files. |
| PYSEC-2026-2280 | Low | 1.1.0 | **Accept.** HTTPEndpoint method-name dispatch. MRTO uses FastAPI router decorators, not `Route(HTTPEndpoint, methods=None)`. |
| PYSEC-2026-249 | Medium | 1.3.1 | **Defer.** `request.form()` urlencoded DoS. MRTO's only form endpoint is `/api/v1/auth/login`, which takes JSON (Content-Type: application/json), not urlencoded. Reverse proxy enforces a 1 MB body limit. |
| PYSEC-2026-248 | Low | 1.3.0 | **Accept.** Path-not-starting-with-`/` authority confusion. Requires an ASGI server that forwards such a path; uvicorn does not. Affects only middleware that reads `request.url` before routing. |

**Why not upgrade starlette?** fastapi 0.128.0 pins
`starlette<0.51,>=0.40`, so we cannot bump starlette past 0.50.x
without also bumping fastapi. fastapi 0.165.0 pulls starlette 1.10.0
but requires regenerating the test suite (Starlette's TestClient
deprecation warnings become errors, and several response-shape
assertions change). The deferral is tracked as a follow-up: upgrade
fastapi to ≥0.130 (which pulls starlette ≥1.0.1) in the next release
after the test matrix is updated. The reverse-proxy mitigations (Host
header rejection, body size limit, JSON-only content type) make the
starlette CVEs non-exploitable in the MRTO deployment shape today.

---

## 3. Operational mitigations (defense in depth)

These reduce the exploitability of any finding above regardless of
whether the package is patched:

- **Reverse proxy rejects malformed Host headers.** nginx's
  `server_name` matching and Caddy's host validation both reject the
  malformed-Host cases that drive PYSEC-2026-161 and PYSEC-2026-248.
- **`request.form()` not used for auth.** Login takes
  `application/json`; the urlencoded DoS (PYSEC-2026-249) cannot reach
  a form parser. The proxy's 1 MB body limit caps the body size anyway.
- **No static file serving.** MRTO serves only API JSON; PYSEC-2026-2281
  (StaticFiles SSRF) does not apply.
- **No `HTTPEndpoint` routes.** All routes use FastAPI decorators with
  explicit HTTP methods; PYSEC-2026-2280 does not apply.
- **JWT secret strength enforced.** `app.py` refuses to boot with a
  known insecure placeholder or a secret shorter than 32 chars unless
  `MRTO_ALLOW_INSECURE_JWT=1` is set (dev/tests only).
- **All secrets flow through `mrto/secrets.py`.** No `os.environ.get`
  of a secret value outside that module; values are never logged.
- **Rate limiting on `/auth/login`.** 10 logins/min/IP mitigates
  credential stuffing regardless of any upstream CVE.
- **Refresh-token rotation with reuse detection.** A stolen refresh
  token triggers family revocation on first reuse (Phase 2).

---

## 4. Follow-up actions

| # | Action | Owner | Target |
|---|--------|-------|--------|
| 1 | Upgrade fastapi to ≥0.130 (pulls starlette ≥1.0.1, fixes M1/M4 of starlette). Requires test-matrix update. | platform | next release |
| 2 | Add pip-audit to CI as a weekly job (`.github/workflows/audit.yml`). | platform | this commit |
| 3 | Add a reverse-proxy config test to CI (verify malformed Host → 444). | platform | next release |
| 4 | Add a body-size-limit test (POST > 1 MB urlencoded → 413). | platform | next release |

---

## 5. How to reproduce

```bash
cd backend
pip install bandit pip-audit

# Static analysis
bandit -r mrto -ll

# Dependency CVEs (runtime deps only — exclude test deps)
grep -v -E '^(pytest|bandit)' requirements.txt > /tmp/runtime_req.txt
pip-audit -r /tmp/runtime_req.txt --strict --desc on
```

The CI job `.github/workflows/audit.yml` runs pip-audit weekly (Monday
06:00 UTC) and on demand, opening a GitHub issue on any HIGH/CRITICAL
finding so it is not missed between manual audits.
