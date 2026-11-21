# Security Policy — MRTO

## Reporting a Vulnerability

If you discover a security vulnerability in MRTO, please report it responsibly:

- **Email**: oil.ms.sohrabi@gmail.com
- **Subject**: `[SECURITY] MRTO — <brief description>`
- Do **not** open a public issue for security-sensitive findings.

We will acknowledge within 5 business days and target a fix within 30 days for HIGH/CRITICAL issues.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |
| < 1.0   | ❌        |

## Security Features

### Container Image Supply Chain

- **GHCR (GitHub Container Registry)** — image registry integrated with GitHub account
- **Cosign keyless signing** — images signed via OIDC identity (sigstore)
- **Build provenance attestation** — SLSA Level 3 provenance attached to image
- **SBOM (CycloneDX)** — software bill of materials generated on every release
- **Trivy scan** — HIGH/CRITICAL vulnerabilities blocked from merge

### Runtime

- **JWT auth** with HS256 (256-bit secret required)
- **RBAC**: `admin` / `operator` / `viewer` roles enforced per route
- **Audit log**: all mutations recorded with actor, timestamp, action
- **Password hashing**: bcrypt with 12 rounds
- **Rate limiting**: per-IP throttling at the nginx reverse proxy

### Deployment

- **Non-root containers** (UID/GID 1000)
- **Read-only root filesystem** recommended (set `readOnlyRootFilesystem: true` in prod)
- **NetworkPolicy** restricts backend ingress to the frontend namespace
- **PodDisruptionBudget** keeps ≥1 backend pod available during node drains
- **Secrets** never committed; provisioned via `kubectl create secret` or SealedSecrets

## Security Best Practices for Production Deployment

1. **Rotate JWT secret** — generate with `openssl rand -hex 32` and store in a secret manager
2. **Use SealedSecrets / External Secrets** instead of plain K8s secrets
3. **Enable PSA `restricted`** profile on the `mrto` namespace
4. **Add an OPA/Kyverno policy** requiring image signature verification (cosign verify)
5. **Scan container base image** weekly; subscribe to GHSA advisories
6. **Enable mTLS** between pods via a service mesh (Istio/Linkerd) for zero-trust
7. **Backup PVCs** (data + models) on a schedule
8. **Restrict egress** via NetworkPolicy to only required destinations (GHCR, NTP, DNS)

## Token / Credential Hygiene

- **Never commit** `.env`, `*.key`, `*.pem`, or `github_pat_*` tokens
- `.gitignore` excludes all common credential file patterns
- Use `gh auth login` or a credential helper for git; avoid PATs in URLs
- If a token leaks, **revoke immediately** at GitHub → Settings → Developer settings → Personal access tokens

## CI/CD Security

The workflow (`.github/workflows/ci.yml`) follows least-privilege:

- `permissions: contents: read` by default
- `packages: write` only on the `build-and-push` job
- `security-events: write` only on the `security-scan` job
- `id-token: write` only for cosign keyless signing (no long-lived keys)
- `GITHUB_TOKEN` (auto-issued) is used for registry auth — no PATs in CI
