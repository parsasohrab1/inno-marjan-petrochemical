# MRTO Deployment Guide (Phase 4 — production hardening)

This document covers everything an operator needs to run MRTO in
production beyond a single `docker compose up`: TLS termination, secret
management, horizontal scaling, backup/restore, load testing, and the
security-audit pipeline. It assumes you have already followed the
quick-start in the root `README.md`.

---

## 1. TLS termination

MRTO itself speaks plain HTTP. Terminate TLS at a reverse proxy in front
of it. Two equally-supported options are shown below; pick one.

### 1a. Nginx + certbot (Let's Encrypt)

`/etc/nginx/sites-available/mrto.conf`:

```nginx
upstream mrto_app {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name mrto.example.com;
    # ACME challenge location for certbot --webroot
    location /.well-known/acme-challenge/ { root /var/www/certbot; }
    location / { return 301 https://$host$request_uri; }
}

server {
    listen 443 ssl http2;
    server_name mrto.example.com;

    ssl_certificate     /etc/letsencrypt/live/mrto.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mrto.example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_tickets off;

    # Security headers (mirror what the app adds; defense in depth).
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;

    # Promote /metrics to internal IPs only (it has no auth).
    location /metrics {
        allow 10.0.0.0/8;
        allow 192.168.0.0/16;
        deny all;
        proxy_pass http://mrto_app;
    }

    location / {
        proxy_pass http://mrto_app;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Forward the client's request id so server logs correlate.
        proxy_set_header X-Request-ID      $request_id;
        proxy_read_timeout 60s;
    }
}
```

Issue the cert:

```bash
sudo certbot certonly --webroot -w /var/www/certbot \
  -d mrto.example.com --agree-tos -m ops@example.com --no-eff-email
# Auto-renew (certbot installs a systemd timer by default).
sudo systemctl enable --now certbot.timer
```

### 1b. Caddy (automatic HTTPS via Let's Encrypt)

`/etc/caddy/Caddyfile` (one file, automatic issuance + renewal):

```caddyfile
mrto.example.com {
    encode zstd gzip
    # /metrics: no auth at the app — restrict to internal IPs.
    @metrics path /metrics
    @metrics handle {
        @internal remote_ip 10.0.0.0/8 192.168.0.0/16
        handle @internal { reverse_proxy 127.0.0.1:8080 }
        handle { respond 403 }
    }
    reverse_proxy 127.0.0.1:8080 {
        header_up X-Request-ID {http.request.uuid}
    }
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        Referrer-Policy strict-origin-when-cross-origin
    }
}
```

Caddy obtains and rotates the cert on first start; no cron needed.

### Choice

| Concern | Nginx+certbot | Caddy |
| --- | --- | --- |
| Zero-config TLS | ✗ (certbot step) | ✓ |
| Mature in your stack | likely | newer |
| Config verbosity | longer | shorter |
| Internal-IP-only `/metrics` | explicit `allow/deny` | `remote_ip` matcher |

Either is fine. The app's own security headers (added in Phase 1) are
redundant with the proxy's; keep both for defense in depth.

---

## 2. Secret management

All runtime secrets flow through `mrto/secrets.py`, which resolves each
name from a backend chain:

1. **Environment variable** (always first — fastest, least privileged).
2. **File on disk** pointed to by `<NAME>_FILE` (Docker/K8s secret mount).
3. **HashiCorp Vault** when `MRTO_VAULT_ADDR` is set and `hvac` is installed.
4. **SOPS** when `MRTO_SOPS_FILE` is set and `sops` is on PATH.

First hit wins; the result is cached for the process lifetime. The
chain degrades silently: if `hvac` is not installed the Vault backend is
skipped, and if `sops` is not on PATH the SOPS backend is skipped.

### 2a. Docker Swarm secret mount

`docker-compose.yml`:

```yaml
services:
  mrto:
    environment:
      MRTO_JWT_SECRET_FILE: /run/secrets/jwt_secret
      MRTO_ADMIN_PASS_FILE: /run/secrets/admin_pass
    secrets: [jwt_secret, admin_pass]
secrets:
  jwt_secret: { file: ./secrets/jwt_secret.txt }
  admin_pass: { file: ./secrets/admin_pass.txt }
```

The app reads the file content at first access; the secret value never
appears in `docker inspect` environment.

### 2b. Kubernetes

```yaml
env:
  - name: MRTO_JWT_SECRET_FILE
    value: /mnt/secrets/jwt_secret
volumeMounts:
  - name: jwt-secret
    mountPath: /mnt/secrets
    readOnly: true
volumes:
  - name: jwt-secret
    secret:
      secretName: mrto-jwt
```

Create the secret out-of-band:

```bash
kubectl create secret generic mrto-jwt \
  --from-literal=jwt_secret="$(openssl rand -hex 32)"
```

### 2c. Vault

```bash
export MRTO_VAULT_ADDR=https://vault.internal:8200
export MRTO_VAULT_TOKEN_FILE=/mnt/secrets/vault-token
export MRTO_VAULT_MOUNT=secret
export MRTO_VAULT_PATH=mrto
vault kv put secret/mrto jwt_secret=... admin_pass=... database_url=...
pip install hvac   # only on hosts that should use Vault
```

The app reads `secret/mrto/data` (kv v2) and looks up the key
`jwt_secret`, `admin_pass`, etc. (the `mrto_` prefix is stripped).

### 2d. SOPS (age-key encrypted file at rest)

`secrets.sops.yaml` (encrypted with `sops --age <key>`):

```yaml
jwt_secret: ENC[AES256_GCM,...]
admin_pass: ENC[AES256_GCM,...]
```

```bash
export MRTO_SOPS_FILE=/etc/mrto/secrets.sops.yaml
# `sops` must be on PATH; the decryption key lives in SOPS_AGE_KEY_FILE.
```

### Which backend?

| Backend | When to use |
| --- | --- |
| env | dev, single-host, CI |
| `_FILE` mount | Docker Swarm, K8s — recommended for prod |
| Vault | multi-service org with a Vault cluster |
| SOPS | git-tracked encrypted secrets with KMS/age |

For Marjan Petrochemical's single-deployment TRL 5 stage, the `_FILE`
mount is the recommended default: no extra services, no extra libs, and
the secret value is invisible to `docker inspect`.

---

## 3. Horizontal scaling

The app is stateless apart from the DB and the model cache (in-memory).
Scale horizontally by running N replicas behind a load balancer.

### 3.1 Workers per replica

uvicorn with `--workers 4` lets one container saturate ~4 CPU cores. For
more, add containers rather than more workers (the in-memory model cache
and rate-limiter state are per-process).

```bash
uvicorn mrto.app:app --host 0.0.0.0 --port 8080 --workers 4 \
  --proxy-headers --forwarded-allow-ips='*'
```

`--proxy-headers` is required so the app trusts `X-Forwarded-*` from the
reverse proxy (rate limiter uses the real client IP).

### 3.2 Load balancer

Nginx upstream block (round-robin with health checks):

```nginx
upstream mrto_app {
    server mrto-1.internal:8080 max_fails=3 fail_timeout=10s;
    server mrto-2.internal:8080 max_fails=3 fail_timeout=10s;
    server mrto-3.internal:8080 max_fails=3 fail_timeout=10s;
    keepalive 32;
}
```

Use the shallow `/api/v1/health` for LB pings and `/api/v1/health/deep`
for k8s readiness. The deep check verifies DB + schema + models + disk
so a replica that has lost its DB connection is drained, not killed.

### 3.3 Shared database

With >1 replica you **must** switch from SQLite to Postgres (SQLite
handles concurrent readers but a single writer, so writes serialize
across replicas). Set:

```bash
MRTO_DATABASE_URL=postgresql://mrto:...@db.internal:5432/mrto?sslmode=require
MRTO_READ_REPLICA_URL=postgresql://mrto:...@replica.internal:5432/mrto?sslmode=require
```

GET endpoints route to the replica (see `_read_conn` in `app.py`); writes
go to the primary. The replica is optional — if unset, reads hit the
primary too.

### 3.4 Model registry

Each replica loads models from `MRTO_MODELS_DIR` on first request and
caches them in-process. To roll out a retrained model:

1. Nightly retrain (`.github/workflows/retrain.yml`) uploads `.joblib`
   artifacts.
2. Operator copies the new `.joblib` files to the shared models volume
   (or runs `kubectl rollout restart` if using a ConfigMap/emptyDir).
3. Each replica picks up the new version on next request (the loader
   checks file mtime; see `_get_model` in `app.py`).

For zero-downtime rollouts, run two replica sets and shift traffic
(blue/green). The model version is recorded in the `model_versions`
table (Phase 2), so you can audit which version each request was served
from by cross-referencing the request timestamp.

### 3.5 Session affinity

Not required. JWTs are stateless; refresh tokens are in the DB (shared).
Any replica can validate any token.

---

## 4. Backup & restore

`scripts/backup.py` and `scripts/restore.py` are the supported path.

### 4.1 Backup

```bash
# Manual
python scripts/backup.py --out /var/backups/mrto --retain 14

# Cron (daily at 02:00, keep 14 days)
# /etc/cron.d/mrto-backup
0 2 * * * mrto cd /opt/mrto/backend && \
  MRTO_BACKUP_DIR=/var/backups/mrto MRTO_BACKUP_RETAIN=14 \
  python scripts/backup.py >> /var/log/mrto-backup.log 2>&1
```

What it produces in `MRTO_BACKUP_DIR`:

| File | Contents |
| --- | --- |
| `mrto_db_YYYYMMDD_HHMMSS.sqlite3` | SQLite copy (online, via backup API) |
| `mrto_db_YYYYMMDD_HHMMSS.sql.gz` | Postgres dump (when `MRTO_DATABASE_URL` set) |
| `mrto_models_YYYYMMDD_HHMMSS.tar.gz` | models directory |
| `mrto_backup_YYYYMMDD_HHMMSS.manifest` | JSON manifest of the above |

Retention keeps the most recent `MRTO_BACKUP_RETAIN` (default 14) backup
sets; older ones are deleted. A backup set = all files sharing a
timestamp prefix.

SQLite backups use the SQLite backup API (page-by-page under a shared
lock), so they are safe to run while the API is serving. Postgres uses
`pg_dump --no-owner --clean --if-exists`, which is consistent regardless
of concurrent writes.

### 4.2 Restore

```bash
# Restore the newest backup
python scripts/restore.py --latest --yes

# Restore a specific DB + models
python scripts/restore.py \
  --db /var/backups/mrto/mrto_db_20241122_153012.sqlite3 \
  --models /var/backups/mrto/mrto_models_20241122_153012.tar.gz
```

Before overwriting, `restore.py` snapshots the current DB to
`<db>.prerestore.<ts>` so a bad restore can be rolled back by hand. It
asks for confirmation unless `--yes` is passed. For Postgres it applies
the dump with `psql --set ON_ERROR_STOP=on`.

### 4.3 Restore runbook

1. Stop the API: `docker compose stop mrto` (or `kubectl scale --replicas=0`).
2. Snapshot the current DB: `cp mrto.sqlite3 mrto.sqlite3.prerestore.$(date +%s)`.
3. Run `python scripts/restore.py --latest --yes`.
4. Restart the API: `docker compose up -d mrto`.
5. Hit `/api/v1/health/deep` and confirm `status: ok` and the expected
   `schema_version`.
6. Spot-check: log in, query `/api/v1/tubes`, confirm row count matches
   the manifest's expected count.

If step 5 fails, the pre-restore snapshot is at
`<db>.prerestore.<ts>` — copy it back to `MRTO_DB_PATH` and restart.

---

## 5. Load testing

`scripts/loadtest.py` runs a synthetic load profile against a running
MRTO instance using only the stdlib (no locust/k6 dependency). It logs
in once, then hammers the read-heavy endpoints (`/tubes`, `/rul`,
`/metrics`, `/health`) at a configurable concurrency for a fixed
duration, and prints a summary (throughput, p50/p95/p99 latency, error
rate).

```bash
# Against a local instance
python scripts/loadtest.py --base http://localhost:8080 \
  --duration 60 --concurrency 20 --user admin --pass admin123

# Save a report
python scripts/loadtest.py --base http://localhost:8080 \
  --duration 120 --concurrency 50 --report loadtest_report.json
```

Expected single-replica ballpark on a 2-core container: ~800–1200 RPS
on the read endpoints, p95 < 100 ms. If you see p95 > 500 ms or error
rate > 1 %, scale out (Section 3) before scaling up.

Results from a sample run are committed under
`docs/loadtest_sample.json` (see file header for the host config).

---

## 6. Security audit

Two linters run in CI (`.github/workflows/ci.yml`):

- **ruff** — style/import/unused-code checks.
- **bandit** — Python-specific security issues (assert in prod, hard-coded
  passwords, shell injection, weak crypto).

For dependency CVEs, `pip-audit` runs in a separate job
(`.github/workflows/audit.yml`) that opens an issue on any HIGH/CRITICAL
finding. Both are non-blocking on `main` (warning only) so a finding
doesn't freeze the pipeline; the operator is expected to triage and
either upgrade or file an exception.

Manual one-off:

```bash
pip install bandit pip-audit
bandit -r mrto -ll                # -ll = only MEDIUM and higher
pip-audit --strict                # fails on any unfixed CVE
```

Findings from the Phase 4 audit are recorded in
`docs/SECURITY_AUDIT.md` (bandit output, pip-audit output, and the
fix/exception decision for each).
