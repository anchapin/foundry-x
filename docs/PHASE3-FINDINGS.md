# Phase 3 Quantization Sweep Findings

## Hardware Profile

| Property | Value |
|----------|-------|
| GPU | AMD RX 6600 XT |
| VRAM | 8 GB |
| OS | Linux (target) |
| ROCm Version | 6.x (target) |

## Planned Sweep

**Status: Pending** — awaiting hardware access and model quantizations.

### Prerequisites

- [ ] `FOUNDRY_MODEL_PATH` pointing to directory containing:
  - `Q4_K_S` quantization
  - `Q5_K_M` quantization
  - `Q6_K` quantization
  - `Q8_0` quantization
- [ ] `OPENCODE_SERVER_URL` pointing to llama.cpp server or OpenAI-compatible endpoint
- [ ] llama.cpp server running with `--rocm` flag for RX 6600 XT support
- [ ] Sufficient VRAM: Q8_0 may require partial offload

### Quantizations Under Test

| Quantization | Memory Est. | Notes |
|--------------|-------------|-------|
| Q4_K_S | ~4.5 GB | Expected minimum viable floor |
| Q5_K_M | ~5.5 GB | Balanced quality/vram |
| Q6_K | ~6.5 GB | Near-fp16 quality |
| Q8_0 | ~8 GB | May need partial offload on 8 GB card |

### Sweep Command

```bash
FOUNDRY_MODEL_PATH=/srv/models \
  foundry-sweep \
  --quantizations Q4_K_S,Q5_K_M,Q6_K,Q8_0 \
  --harness-dir ./harness \
  --output ./logs/sweep-YYYYMMDD.json
```

### Expected Output Format

```json
{
  "sweep_id": "...",
  "timestamp": "...",
  "hardware": {
    "gpu": "AMD RX 6600 XT",
    "vram_gb": 8
  },
  "quantizations": [
    {
      "name": "Q4_K_S",
      "model_path": "/srv/models/Q4_K_S/",
      "result": {
        "passed": 18,
        "failed": 2,
        "pass_rate": 0.90,
        "task_results": [
          {"task": "bash_skill_real_exec", "status": "PASSED"},
          {"task": "grep_search_fix", "status": "PASSED"},
          ...
        ]
      }
    },
    ...
  ],
  "recommended_floor": "Q4_K_S"
}
```

## Intelligence Floor Findings

### Per-Quantization Pass Rates

| Quantization | Pass Rate | Notes |
|--------------|-----------|-------|
| Q4_K_S | TBD | Pending sweep |
| Q5_K_M | TBD | Pending sweep |
| Q6_K | TBD | Pending sweep |
| Q8_0 | TBD | Pending sweep |

### Task-Level Failures

TBD — pending sweep execution.

### Recommended Floor

**TBD** — pending sweep execution.

The recommended floor will be the lowest quantization that maintains >85% benchmark pass rate
while fitting in RX 6600 XT VRAM without offload.

## Related Issues

- Issue #495: Run suite across Q4/Q5/Q8 quantizations
- Issue #541: Run foundry-sweep on RX 6600 XT and record intelligence floor
