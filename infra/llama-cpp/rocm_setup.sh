#!/usr/bin/env bash
set -euo pipefail

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
LLAMACPP_REF="${LLAMACPP_REF:-b9957}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/llvm/bin/clang++"

# Minimum ROCm version that lists Navi 22 (RX 6600 XT) in the supported
# matrix out of the box (infra/llama-cpp/README.md "ROCm pitfalls").
ROCM_MIN_VERSION="${ROCM_MIN_VERSION:-5.7}"

# Optional SHA256 checksum for model integrity verification (issue #284).
# When set, the smoke test verifies the model file matches this digest
# before launching the server. Leave empty to skip verification.
LLAMACPP_MODEL_SHA256="${LLAMACPP_MODEL_SHA256:-}"

# Overrideable probes so the four pre-flight checks are hermetic under
# test (tests/infra/test_rocm_sanity.py). On a real host the defaults
# match the kernel/runtime layout documented in infra/llama-cpp/README.md.
AMDGPU_PROBE="${AMDGPU_PROBE:-/sys/module/amdgpu}"
KFD_PROBE="${KFD_PROBE:-/dev/kfd}"

# --- ROCm pre-flight helpers (issue #210) -------------------------
# _version_ge <actual> <minimum>: return 0 when actual >= minimum.
# Compares dotted numeric versions left-to-right; non-numeric suffixes
# (e.g. a git-describe in /opt/rocm/.info/version) are stripped.
_version_ge() {
    local lhs="$1" rhs="$2"
    local IFS=.
    # shellcheck disable=SC2206
    local a=($lhs) b=($rhs)
    local n=$(( ${#a[@]} > ${#b[@]} ? ${#a[@]} : ${#b[@]} ))
    local i la lb
    for (( i = 0; i < n; i++ )); do
        la=${a[i]:-0}
        lb=${b[i]:-0}
        la=${la%%[!0-9]*}
        lb=${lb%%[!0-9]*}
        la=${la:-0}
        lb=${lb:-0}
        if (( la > lb )); then
            return 0
        elif (( la < lb )); then
            return 1
        fi
    done
    return 0
}

# Resolve the rocminfo binary (prefer $ROCM_PATH/bin/rocminfo, fall back
# to PATH) and print the unique gfx* agents it reports. Empty on failure.
_rocminfo_gfx_agents() {
    local cmd=""
    if [[ -x "$ROCM_PATH/bin/rocminfo" ]]; then
        cmd="$ROCM_PATH/bin/rocminfo"
    elif command -v rocminfo >/dev/null 2>&1; then
        cmd="rocminfo"
    else
        return 0
    fi
    "$cmd" 2>/dev/null | grep -io 'gfx[0-9][0-9][0-9][0-9]*' | sort -u | tr '\n' ' '
}

# Run the four ROCm pre-flight checks, printing one line per check.
# Returns 0 only if all four hold; otherwise prints a one-line hint to
# stderr and returns 1. Used by --check-rocm (standalone) and as the
# build gate below.
check_rocm() {
    local version_file="$ROCM_PATH/.info/version"
    local rocm_version=""
    local version_ok=0 amdgpu_ok=0 kfd_ok=0 gfx_ok=0
    local gfx_agents=""

    # 1. ROCm installed + version >= ROCM_MIN_VERSION
    if [[ -r "$version_file" ]]; then
        rocm_version="$(tr -d '[:space:]' < "$version_file")"
    fi
    if [[ -n "$rocm_version" ]]; then
        if _version_ge "$rocm_version" "$ROCM_MIN_VERSION"; then
            version_ok=1
        fi
        printf 'ROCm version:        %s (>= %s %s)\n' \
            "$rocm_version" "$ROCM_MIN_VERSION" \
            "$([[ $version_ok -eq 1 ]] && echo OK || echo FAIL)"
    else
        printf 'ROCm version:        MISSING (%s unreadable)\n' "$version_file"
    fi

    # 2. amdgpu kernel module loaded
    if [[ -d "$AMDGPU_PROBE" ]]; then
        amdgpu_ok=1
        printf 'amdgpu module:       loaded\n'
    else
        printf 'amdgpu module:       MISSING (no %s)\n' "$AMDGPU_PROBE"
    fi

    # 3. /dev/kfd present
    if [[ -e "$KFD_PROBE" ]]; then
        kfd_ok=1
        printf '/dev/kfd:            present\n'
    else
        printf '/dev/kfd:            MISSING (no %s)\n' "$KFD_PROBE"
    fi

    # 4. rocminfo lists a gfx agent
    gfx_agents="$(_rocminfo_gfx_agents || true)"
    if [[ -n "$gfx_agents" ]]; then
        gfx_ok=1
        printf 'gfx agents:          %s\n' "${gfx_agents% }"
    else
        printf 'gfx agents:          NONE (rocminfo reported no gfx* agent)\n'
    fi

    if [[ $version_ok -eq 1 && $amdgpu_ok -eq 1 \
          && $kfd_ok -eq 1 && $gfx_ok -eq 1 ]]; then
        return 0
    fi

    echo "hint: ROCm preconditions not met; see infra/llama-cpp/README.md 'ROCm pitfalls'" >&2
    return 1
}
# -------------------------------------------------------------------

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
CHECK_ROCM_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-rocm)
            CHECK_ROCM_ONLY=1
            shift
            ;;
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
usage: rocm_setup.sh [--check-rocm] [--smoke-test <gguf>]

Builds llama.cpp with ROCm for the RX 6600 XT. With no arguments the
script runs the ROCm pre-flight checks, then builds and prints the run
hint, exactly as before.

  --check-rocm          Run the four ROCm pre-flight checks (ROCm
                        version, amdgpu module, /dev/kfd, gfx agent)
                        and exit. Does not build. Exits 0 if all
                        preconditions hold, 1 otherwise.
  --smoke-test <gguf>   After building, launch llama-server with the given
                        GGUF and verify it responds (HTTP 200 + non-empty
                        completion). Also enabled via LLAMACPP_SMOKE_MODEL.

Env vars: LLAMACPP_REF (b9957), LLAMACPP_SMOKE_MODEL,
          LLAMACPP_SMOKE_PORT (8765), LLAMACPP_SMOKE_NGL (0),
          LLAMACPP_SMOKE_TIMEOUT (60), LLAMACPP_MODEL_SHA256 (empty),
          ROCM_PATH (/opt/rocm), ROCM_MIN_VERSION (5.7),
          AMDGPU_PROBE (/sys/module/amdgpu), KFD_PROBE (/dev/kfd)
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
current_ref="$(git -C "$LLAMACPP_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$current_ref" && "$current_ref" == "$LLAMACPP_REF" ]]; then
    echo "llama.cpp @ $LLAMACPP_REF already checked out; skipping fetch+checkout"
else
    git -C "$LLAMACPP_DIR" fetch --depth 1 origin "$LLAMACPP_REF"
    git -C "$LLAMACPP_DIR" checkout --detach FETCH_HEAD
fi

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

# --- Model integrity verification (issue #284) ----------------------
# When LLAMACPP_MODEL_SHA256 is set, compute the sha256sum of the model
# file and refuse to proceed if it does not match. This ensures Phase-3
# quantization sweeps use a known-good GGUF (ADR-0007 provenance).
if [[ -n "$LLAMACPP_MODEL_SHA256" ]]; then
    if ! command -v sha256sum >/dev/null 2>&1; then
        echo "error: 'sha256sum' is required for model verification but was not found." >&2
        exit 1
    fi
    actual_sha="$(sha256sum "$SMOKE_TEST_MODEL" | awk '{print $1}')"
    if [[ "$actual_sha" != "$LLAMACPP_MODEL_SHA256" ]]; then
        echo "error: model SHA256 mismatch" >&2
        echo "       expected: $LLAMACPP_MODEL_SHA256" >&2
        echo "       actual:   $actual_sha" >&2
        echo "       file:     $SMOKE_TEST_MODEL" >&2
        exit 1
    fi
    echo "    sha256:   $actual_sha (verified)"
fi
# -------------------------------------------------------------------

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
