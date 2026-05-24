#!/usr/bin/env bash
# Label K3s nodes for the Guitar Detect workload split.
#
# DESIGN.md §2.2: two roles —
#   workload=io       runs gateway + redis (light I/O work, mobile chip)
#   workload=compute  runs inference worker (heavy CPU, Ryzen 7)
#
# Usage:
#   MOBILE_NODE=k3s-mobile COMPUTE_NODE=k3s-ryzen ./label-nodes.sh
#
# Or pass node names as args:
#   ./label-nodes.sh k3s-mobile k3s-ryzen
#
# Idempotent — `kubectl label --overwrite` is safe to re-run.

set -euo pipefail

MOBILE_NODE="${1:-${MOBILE_NODE:-}}"
COMPUTE_NODE="${2:-${COMPUTE_NODE:-}}"

if [[ -z "${MOBILE_NODE}" || -z "${COMPUTE_NODE}" ]]; then
    echo "ERROR: set MOBILE_NODE and COMPUTE_NODE (env vars or positional args)." >&2
    echo "" >&2
    echo "Currently-known nodes:" >&2
    kubectl get nodes -o name >&2
    exit 1
fi

command -v kubectl >/dev/null 2>&1 || {
    echo "ERROR: kubectl not installed or not on PATH." >&2
    exit 1
}

echo "==> Labeling ${MOBILE_NODE} as workload=io"
kubectl label node "${MOBILE_NODE}" workload=io --overwrite

echo "==> Labeling ${COMPUTE_NODE} as workload=compute"
kubectl label node "${COMPUTE_NODE}" workload=compute --overwrite

echo ""
echo "Current labeling:"
kubectl get nodes -L workload
