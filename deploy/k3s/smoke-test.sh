#!/usr/bin/env bash
# Post-deploy smoke test for the Guitar Detect K3s install.
#
# Exercises the gateway HTTP + WS surface from a host that can reach the
# ingress (typically the operator dev host). Exits 0 if all checks pass,
# non-zero on the first failure.
#
# Requirements on the calling host:
#   - curl
#   - websocat (https://github.com/vi/websocat)
#   - jq
#   - DNS / /etc/hosts entry for the configured HOST
#   - mkcert root CA installed (or pass -k to curl via INSECURE=1)
#
# Usage:
#   ./smoke-test.sh                                 # default host
#   HOST=guitars.home.lan ./smoke-test.sh           # explicit override
#   INSECURE=1 ./smoke-test.sh                      # skip cert verification

set -euo pipefail

HOST="${HOST:-guitars.home.lan}"
INSECURE="${INSECURE:-0}"
SESSION_ID="${SESSION_ID:-smoke-$(date +%s)-$$}"

CURL_ARGS=(--silent --show-error --fail --max-time 10)
WEBSOCAT_ARGS=()
if [[ "${INSECURE}" == "1" ]]; then
    CURL_ARGS+=(--insecure)
    WEBSOCAT_ARGS+=(--insecure)
fi

for cmd in curl websocat jq; do
    command -v "${cmd}" >/dev/null 2>&1 || {
        echo "ERROR: ${cmd} not on PATH." >&2
        exit 1
    }
done

trap 'echo "==> Cleaning up session ${SESSION_ID}"; \
      curl "${CURL_ARGS[@]}" -X DELETE "https://${HOST}/api/session/${SESSION_ID}" >/dev/null || true' EXIT

echo "==> 1/4  GET https://${HOST}/healthz"
curl "${CURL_ARGS[@]}" "https://${HOST}/healthz" | jq -e '.ok == true' >/dev/null
echo "    PASS"

echo "==> 2/4  GET https://${HOST}/readyz"
curl "${CURL_ARGS[@]}" "https://${HOST}/readyz" | jq -e '.ok == true' >/dev/null
echo "    PASS"

echo "==> 3/4  POST https://${HOST}/api/session  session_id=${SESSION_ID}"
curl "${CURL_ARGS[@]}" -X POST "https://${HOST}/api/session" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\": \"${SESSION_ID}\"}" \
    | jq -e '.ok == true' >/dev/null
echo "    PASS"

echo "==> 4/4  WS upgrade wss://${HOST}/ws?session_id=${SESSION_ID} (ping/pong)"
WS_OUTPUT=$(printf '{"type":"ping"}\n' | \
    timeout 5 websocat "${WEBSOCAT_ARGS[@]}" --one-message \
        "wss://${HOST}/ws?session_id=${SESSION_ID}" || true)
echo "${WS_OUTPUT}" | jq -e '.type == "pong"' >/dev/null
echo "    PASS  (received ${WS_OUTPUT})"

echo ""
echo "All checks PASSED for ${HOST}."
