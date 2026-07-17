#!/usr/bin/env bash
#
# End-to-end benchmark orchestrator (issue #207).
#
# Ties together the three Phase-3 steps that the operator previously had to
# run by hand:
#
#   1. Launch llama-server (infra/llama-cpp) if it is not already serving.
#   2. Wait for /health to report ready.
#   3. Run the sandboxed agent (infra/docker/docker-compose.yml).
#   4. Assert logs/traces.db grew (evidence the run produced a trace).
#   5. Tear llama-server down on exit (unless --keep-server).
#
# Usage:
#   run_benchmark.sh --model /srv/models/foo.Q5_K_M.gguf --task "..."
#   run_benchmark.sh --task "..." --keep-server           # server already up
#   run_benchmark.sh --task "..." --dry-run               # print compose argv
#   run_benchmark.sh --task "..." --compose-extra -f,infra/docker/docker-compose.rocm.yml
#
# Flags:
#   --model <path>        GGUF model for auto-launched llama-server.
#                         If omitted the script assumes the server is already
#                         running and only waits for /health.
#   --task <prompt>       Task prompt passed to the sandbox runner. Required.
#   --compose-extra <a>   Extra docker compose arguments, comma-separated.
#                         Example: --compose-extra -f,infra/docker/docker-compose.rocm.yml
#   --keep-server         Do not tear down an auto-launched llama-server on exit.
#   --dry-run             Print the resolved docker compose invocation and exit.
#
# Env vars (all optional):
#   LLAMACPP_HOST            Host for /health probe (default http://127.0.0.1:8080)
#   LLAMACPP_SERVER_BIN      Path to llama-server binary (default: LLAMACPP_DIR/build/bin/llama-server)
#   LLAMACPP_DIR             llama.cpp checkout (default $HOME/llama.cpp)
#   LLAMACPP_NGL             GPU layers to offload (default 0)
#   LLAMACPP_HEALTH_TIMEOUT  /health deadline in seconds (default 60)
#   LLAMACPP_MODEL_OVERRIDE  Allow override when running model differs from --model (for sweeps)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

COMPOSE_FILE="$REPO_ROOT/infra/docker/docker-compose.yml"
TRACES_DB="$REPO_ROOT/logs/traces.db"

LLAMACPP_HOST_URL="${LLAMACPP_HOST:-http://127.0.0.1:8080}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMACPP_SERVER_BIN="${LLAMACPP_SERVER_BIN:-$LLAMACPP_DIR/build/bin/llama-server}"
LLAMACPP_NGL="${LLAMACPP_NGL:-0}"
HEALTH_TIMEOUT="${LLAMACPP_HEALTH_TIMEOUT:-60}"

MODEL=""
TASK=""
COMPOSE_EXTRA=()
KEEP_SERVER=0
DRY_RUN=0

usage() {
    cat <<'USAGE'
usage: run_benchmark.sh --task <prompt> [--model <gguf>]
                        [--compose-extra <args>] [--keep-server] [--dry-run]

Required:
  --task <prompt>            Task prompt for the sandboxed agent run.

Optional:
  --model <gguf>             GGUF path. Auto-launches llama-server if not running.
  --compose-extra <a,b,...>  Extra docker compose args (comma-separated).
                             Example: -f,infra/docker/docker-compose.rocm.yml
  --keep-server              Do not tear down auto-launched llama-server on exit.
  --dry-run                  Print the resolved docker compose argv and exit.

Env: LLAMACPP_HOST, LLAMACPP_SERVER_BIN, LLAMACPP_DIR, LLAMACPP_NGL,
     LLAMACPP_HEALTH_TIMEOUT, LLAMACPP_MODEL_OVERRIDE
USAGE
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            [[ $# -ge 2 ]] || { echo "error: --model requires a path argument" >&2; exit 2; }
            MODEL="$2"; shift 2 ;;
        --task)
            [[ $# -ge 2 ]] || { echo "error: --task requires a prompt argument" >&2; exit 2; }
            TASK="$2"; shift 2 ;;
        --compose-extra)
            [[ $# -ge 2 ]] || { echo "error: --compose-extra requires an argument" >&2; exit 2; }
            IFS=',' read -r -a _extra <<< "$2"
            COMPOSE_EXTRA+=("${_extra[@]}")
            shift 2 ;;
        --keep-server)
            KEEP_SERVER=1; shift ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "error: --task is required" >&2
    usage >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Build the resolved docker compose argv (used by --dry-run and the real run)
# ---------------------------------------------------------------------------
compose_argv=(
    docker compose
    -f "$COMPOSE_FILE"
    "${COMPOSE_EXTRA[@]}"
    run --rm foundryx --task "$TASK"
)

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry-run resolved docker compose invocation:"
    printf '  %q' "${compose_argv[@]}"
    echo
    exit 0
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

llamacpp_is_healthy() {
    local resp
    resp="$(curl -fsS "${LLAMACPP_HOST_URL}/health" 2>/dev/null || true)"
    [[ "$resp" == *'"ok"'* ]]
}

get_running_model() {
    curl -fsS "${LLAMACPP_HOST_URL}/v1/models" 2>/dev/null \
        | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        | head -n1 || true
}

wait_for_health() {
    echo "==> Waiting for llama-server /health at ${LLAMACPP_HOST_URL} ..."
    local waited=0
    while [[ $waited -lt $HEALTH_TIMEOUT ]]; do
        if llamacpp_is_healthy; then
            echo "    ready after ${waited}s"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "error: llama-server did not report /health within ${HEALTH_TIMEOUT}s" >&2
    return 1
}

SERVER_PID=""
SERVER_LAUNCHED=0
SERVER_LOG=""

cleanup_server() {
    if [[ $KEEP_SERVER -eq 1 || $SERVER_LAUNCHED -eq 0 ]]; then
        return
    fi
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "==> Tearing down auto-launched llama-server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    [[ -n "$SERVER_LOG" ]] && rm -f "$SERVER_LOG"
}
trap cleanup_server EXIT

# ---------------------------------------------------------------------------
# 1. Ensure llama-server is running
# ---------------------------------------------------------------------------
if ! llamacpp_is_healthy; then
    if [[ -z "$MODEL" ]]; then
        echo "error: llama-server is not reachable at ${LLAMACPP_HOST_URL}" >&2
        echo "       and no --model was given to auto-launch it." >&2
        echo "       Start llama-server manually or pass --model <gguf>." >&2
        exit 1
    fi

    if [[ ! -x "$LLAMACPP_SERVER_BIN" ]]; then
        echo "error: llama-server binary not found or not executable: ${LLAMACPP_SERVER_BIN}" >&2
        echo "       Set LLAMACPP_SERVER_BIN or build llama.cpp (infra/llama-cpp/README.md)." >&2
        exit 1
    fi

    if [[ ! -r "$MODEL" ]]; then
        echo "error: model not found or not readable: ${MODEL}" >&2
        exit 1
    fi

    echo "==> Launching llama-server"
    echo "    binary: $LLAMACPP_SERVER_BIN"
    echo "    model:  $MODEL"
    echo "    ngl:    $LLAMACPP_NGL"

    SERVER_LOG="$(mktemp -t llamacpp-bench.XXXXXX.log)"
    "$LLAMACPP_SERVER_BIN" \
        --model "$MODEL" \
        --host 127.0.0.1 --port 8080 \
        --n-gpu-layers "$LLAMACPP_NGL" \
        >"$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    SERVER_LAUNCHED=1
elif [[ -n "$MODEL" ]]; then
    running_model="$(get_running_model)"
    if [[ -z "$running_model" ]]; then
        echo "error: could not determine model name from running server at ${LLAMACPP_HOST_URL}" >&2
        echo "       The /v1/models endpoint did not return a model id." >&2
        exit 1
    fi
    expected_model="$(basename "$MODEL")"
    if [[ "$running_model" != "$expected_model" ]]; then
        if [[ "${LLAMACPP_MODEL_OVERRIDE:-0}" != "1" ]]; then
            echo "error: running server is serving '${running_model}' but benchmark targets '${expected_model}'" >&2
            echo "       Either stop the other server or set LLAMACPP_MODEL_OVERRIDE=1 to override." >&2
            exit 1
        else
            echo "==> WARNING: LLAMACPP_MODEL_OVERRIDE is set; proceeding despite model mismatch" >&2
            echo "    running: ${running_model}" >&2
            echo "    expected: ${expected_model}" >&2
        fi
    else
        echo "==> Verified running server is serving model: ${running_model}"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Wait for /health
# ---------------------------------------------------------------------------
wait_for_health || {
    if [[ $SERVER_LAUNCHED -eq 1 && -n "$SERVER_LOG" ]]; then
        echo "       --- server log (last 20 lines) ---" >&2
        tail -n 20 "$SERVER_LOG" >&2 || true
    fi
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Record traces.db size, run the sandbox, assert it grew
# ---------------------------------------------------------------------------
before=0
if [[ -f "$TRACES_DB" ]]; then
    before=$(stat -c %s "$TRACES_DB" 2>/dev/null || echo 0)
fi

echo "==> Running sandbox"
printf '  %q\n' "${compose_argv[@]}"
"${compose_argv[@]}"
RUN_RC=$?

if [[ $RUN_RC -ne 0 ]]; then
    echo "error: sandbox exited with code ${RUN_RC}" >&2
    exit "$RUN_RC"
fi

after=0
if [[ -f "$TRACES_DB" ]]; then
    after=$(stat -c %s "$TRACES_DB" 2>/dev/null || echo 0)
fi

if [[ "$after" -le "$before" ]]; then
    echo "error: logs/traces.db did not grow (before=${before}, after=${after})" >&2
    echo "       The run produced no trace events; check the sandbox logs." >&2
    exit 1
fi

echo "==> Benchmark complete"
echo "    traces.db: ${before} -> ${after} bytes"
