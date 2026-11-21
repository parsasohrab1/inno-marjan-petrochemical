#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MRTO — local / staging deployment helper (docker compose)
# Usage:
#   ./deploy/deploy.sh staging   # pull image, compose up, smoke test
#   ./deploy/deploy.sh down      # tear down stack
#   ./deploy/deploy.sh logs      # tail logs
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$ROOT_DIR/docker-compose.yml"
ENV_FILE="$ROOT_DIR/.env"

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

require_env() {
  if [ ! -f "$ENV_FILE" ]; then
    err ".env not found. Copy .env.example to .env and fill in real values."
    err "  cp .env.example .env"
    exit 1
  fi
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
  : "${MRTO_IMAGE:?MRTO_IMAGE must be set in .env}"
  : "${MRTO_JWT_SECRET:?MRTO_JWT_SECRET must be set in .env}"
}

case "${1:-up}" in
  up|staging)
    require_env
    log "Pulling image: $MRTO_IMAGE"
    docker pull "$MRTO_IMAGE"
    log "Composing stack (replicas: 2 backend + 2 proxy)"
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d --quiet-pull
    log "Waiting for backend health (max 60s)..."
    for i in $(seq 1 12); do
      if curl -fsS "http://localhost:${MRTO_PORT:-8080}/api/v1/health" 2>/dev/null | grep -q '"status":"ok"'; then
        log "Backend healthy ✓"
        log "Dashboard: http://localhost:${MRTO_PORT:-8080}/"
        exit 0
      fi
      sleep 5
    done
    err "Backend did not become healthy. Logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
    ;;
  down)
    log "Tearing down stack..."
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down -v
    log "Stack removed."
    ;;
  logs)
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs -f --tail=100
    ;;
  ps)
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps
    ;;
  restart)
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" restart
    log "Stack restarted."
    ;;
  *)
    cat <<EOF
Usage: $0 {up|staging|down|logs|ps|restart}
  up|staging  Pull image, compose up, smoke-test health endpoint
  down        Stop & remove containers (and volumes)
  logs        Tail container logs
  ps          Show container status
  restart     Restart all services
EOF
    exit 1
    ;;
esac
