#!/usr/bin/env bash
set -euo pipefail

# Blocking helper that waits for the host llama-server to report ready
# on /health before the sandbox container starts. Resolves the silent-hang
# problem described in issue #209: when the host llama-server is down,
# the sandbox container hangs on its first model_request for ~10 minutes
# because infra/docker/docker-compose.yml:99 routes LLAMACPP_HOST to the
# host loopback via the llamacpp:8080 alias. This script surfaces the
# failure in seconds instead.
#
# Usage:
#   wait_for_llamacpp.sh [--host HOST] [--port PORT] [--timeout SECS] [--quiet]
#
# Exit codes:
#   0  server responded with HTTP 200 on /health
#   1  timed out waiting
#   2  bad usage (unknown flag or missing value)

HOST="${LLAMACPP_WAIT_HOST:-127.0.0.1}"
PORT="${LLAMACPP_WAIT_PORT:-8080}"
TIMEOUT="${LLAMACPP_WAIT_TIMEOUT:-60}"
QUIET=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            if [[ $# -lt 2 ]]; then
                echo "error: --host requires a value" >&2
                exit 2
            fi
            HOST="$2"
            shift 2
            ;;
        --port)
            if [[ $# -lt 2 ]]; then
                echo "error: --port requires a value" >&2
                exit 2
            fi
            PORT="$2"
            shift 2
            ;;
        --timeout)
            if [[ $# -lt 2 ]]; then
                echo "error: --timeout requires a value" >&2
                               exit 2
            fi
            TIMEOUT="$2"
            shift 2
            ;;
        --quiet|-q)
            QUIET=1
            shift
            ;;
        -h|--help)
            cat <<'USAGE'
usage: wait_for_llamacpp.sh [--host HOST] [--port PORT] [--timeout SECS] [--quiet]

Poll http://HOST:PORT/health until the endpoint returns HTTP 200 or
the deadline expires. Use before starting the sandbox so a missing or
unready llama-server is reported immediately instead of causing a
~10-minute silent hang inside the container (issue #209).

  --host HOST       Host to poll     (default: 127.0.0.1, or LLAMACPP_WAIT_HOST)
  --port PORT       Port to poll     (default: 8080,     or LLAMACPP_WAIT_PORT)
  --timeout SECS    Deadline seconds (default: 60,       or LLAMACPP_WAIT_TIMEOUT)
  --quiet, -q       Suppress progress messages on stderr

Exit codes:
  0  server responded 200 OK on /health
  1  timed out
  2  bad usage
USAGE
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2
            exit 2
            ;;
    esac
done

if ! command -v curl >/dev/null 2>&1; then
    echo "error: 'curl' is required but was not found on PATH." >&2
    exit 2
fi

URL="http://${HOST}:${PORT}/health"

if [[ $QUIET -ne 1 ]]; then
    echo "Waiting for ${URL} (timeout: ${TIMEOUT}s)..." >&2
fi

waited=0
while [[ $waited -lt $TIMEOUT ]]; do
    code="$(curl -s -o /dev/null -w '%{http_code}' \
        --connect-timeout 2 --max-time 5 "$URL" 2>/dev/null || echo "000")"
    if [[ "$code" == "200" ]]; then
        if [[ $QUIET -ne 1 ]]; then
            echo "llama-server is ready (${URL} → 200, waited ${waited}s)." >&2
        fi
        exit 0
    fi
    sleep 1
    waited=$((waited + 1))
done

if [[ $QUIET -ne 1 ]]; then
    echo "error: timed out after ${TIMEOUT}s waiting for ${URL}" >&2
    echo "       Is the host llama-server running?" >&2
    echo "       Start it with: infra/llama-cpp/rocm_setup.sh" >&2
fi
exit 1
