#!/bin/bash
#
# Context window sweep execution script for AMD RX 6600 XT (issue #1079).
#
# Executes the ADR-0020 context window sweep at 8192, 16384, and 32768 tokens
# to determine whether the intelligence floor changes at larger context windows.
#
# Prerequisites:
#   1. FOUNDRY_MODEL_PATH pointing to directory with Q5_K_M, Q8_0 quantizations
#   2. OPENCODE_SERVER_URL pointing to llama.cpp server or OpenAI-compatible endpoint
#   3. llama.cpp server running with --rocm flag
#
# Usage:
#   chmod +x logs/run-ctx-sweep.sh
#   FOUNDRY_MODEL_PATH=/srv/models ./logs/run-ctx-sweep.sh
#
# The sweep is executed at each context window size (8192, 16384, 32768) using
# the --context-tokens flag added in PR #1076 (issue #1050). Results are written
# to logs/sweep_ctx_${CONTEXT_TOKENS}.json for each context window.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration — adjust these for your setup
MODEL_PATH="${FOUNDRY_MODEL_PATH:-/srv/models}"
HARNESS_DIR="${HARNESS_DIR:-$REPO_ROOT/harness}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/logs}"

# Context window sizes to sweep (per ADR-0020)
CONTEXT_WINDOWS="8192 16384 32768"

# Quantizations to test — recommended floor (Q5_K_M) and baseline (Q8_0)
QUANTIZATIONS="Q5_K_M,Q8_0"
BASELINE="Q8_0"

# Regression threshold in percentage points
REGRESSION_THRESHOLD="2.0"

echo "=============================================="
echo "Context Window Sweep — AMD RX 6600 XT"
echo "Issue: #1079"
echo "=============================================="
echo "Model path:       ${MODEL_PATH}"
echo "Harness dir:      ${HARNESS_DIR}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Context windows:  ${CONTEXT_WINDOWS}"
echo "Quantizations:    ${QUANTIZATIONS}"
echo "Baseline:         ${BASELINE}"
echo "=============================================="

# Verify model path exists
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH '${MODEL_PATH}' does not exist"
    exit 1
fi

# Verify each quantization exists
for quant in Q5_K_M Q8_0; do
    quant_path="${MODEL_PATH}/${quant}"
    if [[ ! -d "${quant_path}" ]] && [[ ! -f "${quant_path}.gguf" ]]; then
        echo "WARNING: ${quant} not found at ${quant_path}"
    else
        echo "OK: ${quant} found"
    fi
done

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
echo ""
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Run the sweep at each context window size
for CTX in $CONTEXT_WINDOWS; do
    OUTPUT_FILE="${OUTPUT_DIR}/sweep_ctx_${CTX}-${TIMESTAMP}.json"

    echo "=============================================="
    echo "Running sweep at context_tokens=${CTX}"
    echo "Output: ${OUTPUT_FILE}"
    echo "=============================================="

    # Set FOUNDRY_CONTEXT_TOKENS for this sweep via the --context-tokens flag
    FOUNDRY_MODEL_PATH="${MODEL_PATH}" \
        uv run foundry-sweep sweep \
            --quantizations "${QUANTIZATIONS}" \
            --harness-dir "${HARNESS_DIR}" \
            --baseline "${BASELINE}" \
            --regression-threshold "${REGRESSION_THRESHOLD}" \
            --context-tokens "${CTX}" \
            --output "${OUTPUT_FILE}"

    echo ""
    echo "Sweep complete for context_tokens=${CTX}"
    echo "Results written to: ${OUTPUT_FILE}"
    echo ""
done

echo "=============================================="
echo "Context window sweep complete"
echo "=============================================="
echo ""
echo "After running on GPU hardware, update docs/adr/0020-phase-3-findings.md"
echo "with the empirical pass rates from each output file:"
echo ""
for CTX in $CONTEXT_WINDOWS; do
    echo "  logs/sweep_ctx_${CTX}-${TIMESTAMP}.json"
done
echo ""
echo "The Context Window Intelligence Floor Table in ADR-0020 §Context Window Sweep"
echo "tracks pass rates at 8192, 16384, and 32768 tokens."
