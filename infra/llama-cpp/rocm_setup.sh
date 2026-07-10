#!/usr/bin/env bash
set -euo pipefail

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMACPP_REF="${LLAMACPP_REF:-b9957}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/llvm/bin/clang++"

# --- Smoke-test configuration (opt-in) ----------------------------
# After a successful build the script can optionally verify the built
# llama-server actually serves inference. Enable with either:
#   ./rocm_setup.sh --smoke-test /path/to/model.gguf
#   LLAMACPP_SMOKE_MODEL=/path/to/model.gguf ./rocm_setup.sh
# Additional knobs (all optional):
#   LLAMACPP_SMOKE_PORT     port for the ephemeral server (default 8765)
#   LLAMACPP_SMOKE_NGL      GPU layers to offload        (default 0)
#   LLAMACPP_SMOKE_TIMEOUT  readiness deadline, seconds  (default 60)
# When unset the script behaves exactly as before (build + echo).
LLAMACPP_SMOKE_MODEL="${LLAMACPP_SMOKE_MODEL:-}"
SMOKE_TEST_MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            if [[ $# -lt 2 ]]; then
                echo "error: --smoke-test requires a model path argument" >&2
                exit 2
            fi
            SMOKE_TEST_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'USAGE'
usage: rocm_setup.sh [--smoke-test <gguf>]

Builds llama.cpp with ROCm for the RX 6600 XT. With no arguments the
script builds and prints the run hint, exactly as before.

  --smoke-test <gguf>   After building, launch llama-server with the given
                        GGUF and verify it responds (HTTP 200 + non-empty
                        completion). Also enabled via LLAMACPP_SMOKE_MODEL.

Env vars: LLAMACPP_REF (b9957), LLAMACPP_SMOKE_MODEL,
          LLAMACPP_SMOKE_PORT (8765), LLAMACPP_SMOKE_NGL (0),
          LLAMACPP_SMOKE_TIMEOUT (60)
USAGE
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2
            exit 2
            ;;
    esac
done
# CLI flag takes precedence over the env var.
if [[ -z "$SMOKE_TEST_MODEL" && -n "$LLAMACPP_SMOKE_MODEL" ]]; then
    SMOKE_TEST_MODEL="$LLAMACPP_SMOKE_MODEL"
fi
# -------------------------------------------------------------------

if [[ ! -d "$LLAMACPP_DIR" ]]; then
    git clone --no-checkout --filter=blob:none https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR"
fi
# -------------------------------------------------------------------

# --- Pre-build ROCm sanity gate (issue #210) ----------------------
# Without ROCm the build either fails deep inside cmake with a cryptic
# HIP error or produces a binary that silently falls back to CPU at
# runtime. Run the four checks and refuse to build unless all hold.
# --check-rocm runs ONLY the checks and never builds.
if [[ "$CHECK_ROCM_ONLY" -eq 1 ]]; then
    if check_rocm; then
        exit 0
    fi
    exit 1
fi

if ! check_rocm; then
    echo "error: refusing to build; ROCm preconditions not met (run with --check-rocm for details)" >&2
    exit 1
fi
# -------------------------------------------------------------------

if [[ ! -d "$LLAMACPP_DIR" ]]; then
    git clone --no-checkout --filter=blob:none https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR"
fi

echo "Building llama.cpp @ $LLAMACPP_REF"
git -C "$LLAMACPP_DIR" fetch --depth 1 origin "$LLAMACPP_REF"
git -C "$LLAMACPP_DIR" checkout --detach FETCH_HEAD

echo "Building llama.cpp @ $LLAMACPP_REF"
git -C "$LLAMACPP_DIR" fetch --depth 1 origin "$LLAMACPP_REF"
git -C "$LLAMACPP_DIR" checkout --detach FETCH_HEAD

cd "$LLAMACPP_DIR"

HIPCFLAGS="-march=native -mtune=native" \
CMAKE_HIP_COMPILER="$HIPCC" \
    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1032

cmake --build build --config Release -j"$(nproc)"

SERVER_BIN="$LLAMACPP_DIR/build/bin/llama-server"

echo
echo "Built llama.cpp ROCm binary at $SERVER_BIN"
echo "Start it with: $SERVER_BIN --model <gguf>"

# No smoke test requested: preserve the historical zero-arg behavior.
if [[ -z "$SMOKE_TEST_MODEL" ]]; then
    exit 0
fi

# --- Post-build smoke test -----------------------------------------
SMOKE_PORT="${LLAMACPP_SMOKE_PORT:-8765}"
SMOKE_NGL="${LLAMACPP_SMOKE_NGL:-0}"
SMOKE_TIMEOUT="${LLAMACPP_SMOKE_TIMEOUT:-60}"
SMOKE_SERVER_PID=""
SMOKE_BASE="http://127.0.0.1:${SMOKE_PORT}"

smoke_cleanup() {
    if [[ -n "${SMOKE_SERVER_PID}" ]] && kill -0 "$SMOKE_SERVER_PID" 2>/dev/null; then
        kill "$SMOKE_SERVER_PID" 2>/dev/null || true
        wait "$SMOKE_SERVER_PID" 2>/dev/null || true
    fi
}
trap smoke_cleanup EXIT

smoke_fail() {
    echo "error: $1" >&2
    cat >&2 <<'HINT'

       The ROCm-built llama-server did not serve inference. Common RX 6600
       XT pitfalls (see infra/llama-cpp/README.md "ROCm pitfalls"):
         * HSA_OVERRIDE_GFX_VERSION=10.3.0 may be required on older kernels.
         * --n-gpu-layers is VRAM-bound (8 GB); lower LLAMACPP_SMOKE_NGL.
         * Large --ctx-size competes with model weights for VRAM.
HINT
    exit 1
}

echo
echo "==> Smoke test: launching llama-server"
echo "    model:   ${SMOKE_TEST_MODEL}"
echo "    ngl:     ${SMOKE_NGL}"
echo "    port:    ${SMOKE_PORT}"

if [[ ! -r "$SMOKE_TEST_MODEL" ]]; then
    echo "error: smoke-test model not found or not readable: ${SMOKE_TEST_MODEL}" >&2
    echo "       pass --smoke-test <path-to-gguf> or set LLAMACPP_SMOKE_MODEL." >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "error: 'curl' is required for the smoke test but was not found." >&2
    exit 1
fi

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "error: expected server binary missing or not executable: ${SERVER_BIN}" >&2
    exit 1
fi

SERVER_LOG="$(mktemp -t llama-smoke.XXXXXX.log)"

"$SERVER_BIN" \
    --model "$SMOKE_TEST_MODEL" \
    --host 127.0.0.1 \
    --port "$SMOKE_PORT" \
    --n-gpu-layers "$SMOKE_NGL" \
    >"$SERVER_LOG" 2>&1 &
SMOKE_SERVER_PID=$!

# Poll /health until the server reports ready or we time out.
waited=0
healthy=0
while [[ $waited -lt $SMOKE_TIMEOUT ]]; do
    if ! kill -0 "$SMOKE_SERVER_PID" 2>/dev/null; then
        echo "error: llama-server exited before becoming ready." >&2
        echo "       --- server log (last 20 lines) ---" >&2
        tail -n 20 "$SERVER_LOG" >&2 || true
        rm -f "$SERVER_LOG"
        smoke_fail "server process terminated during startup"
    fi
    if health="$(curl -fsS "${SMOKE_BASE}/health" 2>/dev/null || true)"; then
        if [[ "$health" == *'"ok"'* ]]; then
            healthy=1
            break
        fi
    fi
    sleep 1
    waited=$((waited + 1))
done

if [[ $healthy -ne 1 ]]; then
    echo "error: llama-server did not report ready within ${SMOKE_TIMEOUT}s." >&2
    echo "       --- server log (last 20 lines) ---" >&2
    tail -n 20 "$SERVER_LOG" >&2 || true
    rm -f "$SERVER_LOG"
    smoke_fail "timed out waiting for /health"
fi

# Assert a completion actually returns non-empty content.
completion="$(curl -fsS -X POST "${SMOKE_BASE}/completion" \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Say hello in one word.","n_predict":4,"temperature":0}' \
    2>/dev/null || true)"
content="$(printf '%s' "$completion" \
    | sed -n 's/.*"content"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)"
if [[ -z "$content" ]]; then
    echo "error: /completion returned empty content on a healthy server." >&2
    echo "       response: ${completion:-<empty>}" >&2
    rm -f "$SERVER_LOG"
    smoke_fail "no completion content"
fi

# Report the model name via the OpenAI-compatible endpoint (best-effort).
model_name="$(curl -fsS "${SMOKE_BASE}/v1/models" 2>/dev/null \
    | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1 || true)"
[[ -z "$model_name" ]] && model_name="(unreported)"

rm -f "$SERVER_LOG"

echo "==> Smoke test PASSED"
echo "    model:    ${model_name}"
echo "    sample:   ${content}"
echo "    binary:   ${SERVER_BIN}"
