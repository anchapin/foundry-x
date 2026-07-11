#!/usr/bin/env bash
#
# launch_llamacpp.sh -- single env-driven entry point for llama-server.
#
# Wraps the llama-server invocation documented in
# infra/llama-cpp/README.md so Phase-3 quantization sweeps can flip
# --n-gpu-layers and the model path without copy-pasting the full flag
# set each run. Resolves the flags, prints the PID and the resolved
# /health URL on stdout, then exec's llama-server in the foreground.
set -euo pipefail

# Path to the llama-server binary. Override with LLAMACPP_SERVER_BIN;
# otherwise fall back to whatever is on PATH.
LLAMACPP_SERVER_BIN="${LLAMACPP_SERVER_BIN:-llama-server}"

MODEL=""
HOST="0.0.0.0"
PORT="8080"
N_GPU_LAYERS="0"
CTX_SIZE="8192"
LOG_FILE=""
PID_FILE=""

print_usage() {
    cat <<'USAGE'
usage: launch_llamacpp.sh --model <gguf> [options]

Launch llama-server with a resolved, reviewable flag set. The script
prints the server PID and the resolved /health URL on stdout, then
exec's llama-server in the foreground (Ctrl-C stops the server).

Required:
  --model <path>            Path to the GGUF model file.

Options (all have defaults):
  --host <addr>             Bind address (default 0.0.0.0).
  --port <port>             Bind port (default 8080).
  --n-gpu-layers <n>        Layers to offload to GPU (default 0).
  --ctx-size <n>            Context window size (default 8192).
  --log-file <path>         Redirect llama-server stdout/stderr here.
                            The PID/health lines still go to the
                            script's own stdout.
  --pid-file <path>         Write the server PID to this file.
  -h, --help                Show this help and exit.

Env vars:
  LLAMACPP_SERVER_BIN       Path to llama-server binary
                            (default: llama-server on PATH).
USAGE
}

need_arg() {
    if [[ $# -lt 2 ]]; then
        echo "error: $1 requires an argument (see --help)" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            need_arg "$@"
            MODEL="$2"; shift 2 ;;
        --host)
            need_arg "$@"
            HOST="$2"; shift 2 ;;
        --port)
            need_arg "$@"
            PORT="$2"; shift 2 ;;
        --n-gpu-layers)
            need_arg "$@"
            N_GPU_LAYERS="$2"; shift 2 ;;
        --ctx-size)
            need_arg "$@"
            CTX_SIZE="$2"; shift 2 ;;
        --log-file)
            need_arg "$@"
            LOG_FILE="$2"; shift 2 ;;
        --pid-file)
            need_arg "$@"
            PID_FILE="$2"; shift 2 ;;
        -h|--help)
            print_usage; exit 0 ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2
            exit 2 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "error: --model is required (see --help)" >&2
    exit 2
fi

# Resolved argv, in a fixed order so sweeps and tests can diff it.
ARGS=(
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --n-gpu-layers "$N_GPU_LAYERS"
    --ctx-size "$CTX_SIZE"
)

# PID/health are emitted before exec; exec preserves this PID, so $$
# is the PID llama-server will run as.
if [[ -n "$PID_FILE" ]]; then
    printf '%s\n' "$$" >"$PID_FILE"
fi

echo "pid: $$"
echo "health: http://${HOST}:${PORT}/health"

# Redirect only the exec'd process's fds when a log file is requested;
# the PID/health lines above already reached the real stdout.
if [[ -n "$LOG_FILE" ]]; then
    exec "$LLAMACPP_SERVER_BIN" "${ARGS[@]}" >"$LOG_FILE" 2>&1
else
    exec "$LLAMACPP_SERVER_BIN" "${ARGS[@]}"
fi
