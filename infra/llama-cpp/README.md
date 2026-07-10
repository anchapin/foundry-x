# llama.cpp ROCm setup

Target: AMD RX 6600 XT on Linux Mint with ROCm 5.7+.

## Build

```bash
LLAMACPP_REF="${LLAMACPP_REF:-b9957}"
git clone --no-checkout --filter=blob:none https://github.com/ggerganov/llama.cpp
git -C llama.cpp fetch --depth 1 origin "$LLAMACPP_REF"
git -C llama.cpp checkout --detach FETCH_HEAD
cd llama.cpp
HIPCFLAGS="-march=native -mtune=native" \
CMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1032
cmake --build build --config Release -j"$(nproc)"
```

`gfx1032` is the Navi 22 / RX 6600 XT target. Adjust for other cards.

## Run

```bash
./build/bin/llama-server \
  --model /srv/models/your-model.Q5_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 35 \
  --ctx-size 8192
```

Then point `LLAMACPP_HOST` in `.env` at it.

## Smoke test

`rocm_setup.sh` can verify the built server actually serves inference
right after building — opt-in, no model needed in CI:

```bash
./rocm_setup.sh --smoke-test /srv/models/your-model.Q5_K_M.gguf
# or via env var:
LLAMACPP_SMOKE_MODEL=/srv/models/your-model.Q5_K_M.gguf ./rocm_setup.sh
```

It launches `llama-server` in the background, polls `/health` for up to
60 s, asserts a non-empty `/completion`, reports the model name, then
tears the server down. On failure it prints the server log and points
at the pitfalls below.

| Env var | Default | Purpose |
| --- | --- | --- |
| `LLAMACPP_SMOKE_NGL` | `0` | GPU layers to offload. Raise (e.g. `35`) to exercise the ROCm path. |
| `LLAMACPP_SMOKE_PORT` | `8765` | Ephemeral port for the test server. |
| `LLAMACPP_SMOKE_TIMEOUT` | `60` | Readiness deadline in seconds. |

## ROCm pitfalls on RX 6600 XT

- ROCm 5.7+ lists Navi 22 in the supported matrix out of the box.
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` may be needed if you hit `agent refused` errors on older kernels.
- `n-gpu-layers` is VRAM-bound; the 6600 XT has 8 GB. Offload the full model for Q5 quantizations, partial for Q6/Q8.
- Watch VRAM headroom when bumping `--ctx-size`: context caching competes with model weights.

## Phase 3 automation

`src/foundry_x/evolution/critic.py` is the natural place to add a "quantization sweep" that re-evaluates the same harness against Q4, Q5, and Q6 builds to find the intelligence floor for the available VRAM.
