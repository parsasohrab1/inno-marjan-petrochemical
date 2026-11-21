#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# MRTO — Kubernetes deployment helper
# Usage:
#   ./deploy/k8s/deploy.sh apply     # create namespace + apply all manifests
#   ./deploy/k8s/deploy.sh status    # show pods, svc, ingress
#   ./deploy/k8s/deploy.sh rollback   # rollout undo
#   ./deploy/k8s/deploy.sh delete     # remove namespace
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$SCRIPT_DIR/manifests.yaml"
NAMESPACE="${MRTO_NAMESPACE:-mrto}"

log() { printf '\033[1;34m[k8s]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

case "${1:-apply}" in
  apply)
    log "Creating JWT secret (if not present)..."
    if ! kubectl -n "$NAMESPACE" get secret mrto-secrets >/dev/null 2>&1; then
      kubectl -n "$NAMESPACE" create secret generic mrto-secrets \
        --from-literal=MRTO_JWT_SECRET="$(openssl rand -hex 32)" \
        --from-literal=MRTO_ADMIN_USER=admin \
        --from-literal=MRTO_ADMIN_PASS="$(openssl rand -base64 18)" \
        --dry-run=client -o yaml | kubectl apply -f -
      log "Secret created. Retrieve admin pass with:"
      log "  kubectl -n $NAMESPACE get secret mrto-secrets -o jsonpath='{.data.MRTO_ADMIN_PASS}' | base64 -d"
    fi
    log "Applying manifests..."
    kubectl apply -f "$MANIFEST"
    log "Waiting for rollout (max 180s)..."
    kubectl -n "$NAMESPACE" rollout status deployment/mrto-backend --timeout=180s
    kubectl -n "$NAMESPACE" rollout status deployment/mrto-frontend --timeout=180s
    log "✓ Deployed. Ingress host: $(kubectl -n $NAMESPACE get ingress mrto -o jsonpath='{.spec.rules[0].host}')"
    ;;
  status)
    kubectl -n "$NAMESPACE" get pods,services,ingress,hpa,pdb -o wide
    ;;
  rollback)
    kubectl -n "$NAMESPACE" rollout undo deployment/mrto-backend
    kubectl -n "$NAMESPACE" rollout undo deployment/mrto-frontend
    log "Rollback initiated."
    ;;
  delete)
    err "This will delete the $NAMESPACE namespace and all its resources."
    read -rp "Type the namespace name to confirm: " confirm
    [ "$confirm" = "$NAMESPACE" ] || { err "Aborted."; exit 1; }
    kubectl delete namespace "$NAMESPACE"
    log "Namespace $NAMESPACE deleted."
    ;;
  *)
    cat <<EOF
Usage: $0 {apply|status|rollback|delete}
  apply     Create secret, apply manifests, wait for rollout
  status    Show pods, services, ingress, HPA, PDB
  rollback  Rollout undo both deployments
  delete    Remove the namespace (irreversible)
EOF
    exit 1
    ;;
esac
