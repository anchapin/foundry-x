#!/usr/bin/env bash
set -euo pipefail

LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/llama.cpp}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIPCC="${ROCM_PATH}/llvm/bin/clang++"

if [[ ! -d "$LLAMACPP_DIR" ]]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR"
fi

cd "$LLAMACPP_DIR"

HIPCFLAGS="-march=native -mtune=native" \
CMAKE_HIP_COMPILER="$HIPCC" \
    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1032

cmake --build build --config Release -j"$(nproc)"

echo
echo "Built llama.cpp ROCm binary at $LLAMACPP_DIR/build/bin/llama-server"
echo "Start it with: $LLAMACPP_DIR/build/bin/llama-server --model <gguf>"
