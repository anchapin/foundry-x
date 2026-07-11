#!/usr/bin/env bash
set -euo pipefail

# Operator-facing wrapper that gates the sandboxed benchmark run behind
# a readiness check on the host llama-server (issue #209). Without this
# gate the sandbox container starts, tries its first model_request, and
# hangs silently for ~10 minutes when the host server is down.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/infra/docker/docker-compose.yml"

# Pass through wait-related flags to wait_for_llamacpp.sh, everything
# else becomes the runner's --task arguments.
WAIT_ARGS=()
RUNNER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host|--port|--timeout)
            if [[ $# -ge 2 ]]; then
                WAIT_ARGS+=("$1" "$2")
                shift 2
            else
                WAIT_ARGS+=("$1")
                shift
            fi
            ;;
        --quiet|-q)
            WAIT_ARGS+=("$1")
            shift
            ;;
        *)
            RUNNER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Block until llama-server is up so the operator gets the error *before*
# the sandbox container starts (and hangs silently for ~10 min).
if ! "${SCRIPT_DIR}/wait_for_llamacpp.sh" "${WAIT_ARGS[@]+"${WAIT_ARGS[@]}"}"; then
    exit 1
fi

exec docker compose -f "${COMPOSE_FILE}" run --rm foundryx "${RUNNER_ARGS[@]+"${RUNNER_ARGS[@]}"}"
