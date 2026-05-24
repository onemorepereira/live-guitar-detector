#!/usr/bin/env bash
# Generate a mkcert-signed TLS cert for guitars.home.lan and install it
# as a Kubernetes TLS secret in the guitar-detect namespace.
#
# Run on your developer/admin host (the one with mkcert installed AND
# kubectl context pointed at the target cluster). Idempotent — re-running
# regenerates the secret in place via `kubectl apply`.
#
# Requirements:
#   - mkcert installed locally (https://github.com/FiloSottile/mkcert)
#   - kubectl context pointing at the target cluster
#
# Post-install reminders printed at the end:
#   - Install the mkcert root CA on every viewing device (see README).
#   - Add `<ingress-ip>  guitars.home.lan` to router DNS or device /etc/hosts.

set -euo pipefail

HOST="${HOST:-guitars.home.lan}"
NAMESPACE="${NAMESPACE:-guitar-detect}"
SECRET_NAME="${SECRET_NAME:-guitars-tls}"
WORK_DIR="${WORK_DIR:-$(mktemp -d)}"

command -v mkcert >/dev/null 2>&1 || {
  echo "ERROR: mkcert not installed. Install from https://github.com/FiloSottile/mkcert" >&2
  exit 1
}
command -v kubectl >/dev/null 2>&1 || {
  echo "ERROR: kubectl not installed." >&2
  exit 1
}

echo "==> Ensuring mkcert root CA is installed in the local trust store"
mkcert -install

echo "==> Generating cert for ${HOST} into ${WORK_DIR}"
cd "${WORK_DIR}"
mkcert "${HOST}"
# mkcert outputs files like guitars.home.lan.pem + guitars.home.lan-key.pem
CERT_FILE="${WORK_DIR}/${HOST}.pem"
KEY_FILE="${WORK_DIR}/${HOST}-key.pem"
[[ -f "${CERT_FILE}" && -f "${KEY_FILE}" ]] || {
  echo "ERROR: expected ${CERT_FILE} / ${KEY_FILE}, not found." >&2
  exit 1
}

echo "==> Creating namespace ${NAMESPACE} if missing"
kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

echo "==> Applying TLS secret ${SECRET_NAME} in ${NAMESPACE}"
kubectl create secret tls "${SECRET_NAME}" \
  --cert="${CERT_FILE}" \
  --key="${KEY_FILE}" \
  -n "${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

CA_ROOT=$(mkcert -CAROOT)

cat <<EOF

============================================================================
TLS secret '${SECRET_NAME}' installed in namespace '${NAMESPACE}'.

Cert and key files: ${WORK_DIR}/${HOST}.pem and ${HOST}-key.pem
mkcert root CA dir: ${CA_ROOT}

Next steps:

1. DNS — make ${HOST} resolve to your K3s ingress IP.
   Get the ingress IP:
     kubectl -n kube-system get svc traefik -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
   Then either:
     a) Add an A record to your router/LAN DNS, OR
     b) Add to /etc/hosts on every viewing device:
        <ingress-ip>  ${HOST}

2. Trust — install the mkcert root CA on every viewing device. See
   deploy/k3s/README.md for per-platform instructions (rootCA.pem lives
   at ${CA_ROOT}/rootCA.pem).

3. Verify:
     curl -v https://${HOST}/healthz
   Expected: HTTP 200 with a green padlock from any device that trusts
   the mkcert CA.
============================================================================
EOF
