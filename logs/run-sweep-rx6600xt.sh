#!/bin/bash
# foundry-sweep run script for AMD RX 6600 XT
# Issue: #541
# Status: PLANNED — run this on hardware with RX 6600 XT
#
# Prerequisites:
#   1. FOUNDRY_MODEL_PATH pointing to directory with Q4_K_S, Q5_K_M, Q6_K, Q8_0 quantizations
#   2. OPENCODE_SERVER_URL pointing to llama.cpp server or OpenAI-compatible endpoint
#   3. llama.cpp server running with --rocm flag
#
# Usage:
#   chmod +x logs/run-sweep-rx6600xt.sh
#   FOUNDRY_MODEL_PATH=/srv/models ./logs/run-sweep-rx6600xt.sh

set -euo pipefail

# Configuration — adjust these for your setup
MODEL_PATH="${FOUNDRY_MODEL_PATH:-/srv/models}"
HARNESS_DIR="${HARNESS_DIR:-./harness}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs}"

# Generate timestamped output path
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/sweep-rx6600xt-${TIMESTAMP}.json"

# Quantizations to test (in order of increasing VRAM requirement)
QUANTIZATIONS="Q4_K_S,Q5_K_M,Q6_K,Q8_0"

echo "=========================================="
echo "foundry-sweep — AMD RX 6600 XT"
echo "=========================================="
echo "Timestamp:    ${TIMESTAMP}"
echo "Model path:   ${MODEL_PATH}"
echo "Harness dir:  ${HARNESS_DIR}"
echo "Output:       ${OUTPUT_FILE}"
echo "Quantizations: ${QUANTIZATIONS}"
echo "=========================================="

# Verify model path exists
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH '${MODEL_PATH}' does not exist"
    exit 1
fi

# Verify each quantization exists
for quant in Q4_K_S Q5_K_M Q6_K Q8_0; do
    quant_path="${MODEL_PATH}/${quant}"
    if [[ ! -d "${quant_path}" ]] && [[ ! -f "${quant_path}.gguf" ]]; then
        echo "WARNING: ${quant} not found at ${quant_path}"
    else
        echo "OK: ${quant} found"
    fi
done

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Run the sweep
# Note: foundry-sweep CLI is implemented in PRs #473, #527 (on fix/issue-495-benchmarks-run-suite branch)
# This script documents the expected invocation once the infrastructure is merged.
echo ""
echo "Ready to run:"
echo "  FOUNDRY_MODEL_PATH=${MODEL_PATH} \\"
echo "    foundry-sweep \\"
echo "      --quantizations ${QUANTIZATIONS} \\"
echo "      --harness-dir ${HARNESS_DIR} \\"
echo "      --output ${OUTPUT_FILE}"
echo ""
echo "After sweep completes, update docs/PHASE3-FINDINGS.md with results."
