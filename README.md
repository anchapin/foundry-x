# FoundryX

A self-improving agent harness foundry. FoundryX turns agent development from manual prompt engineering into an automated evolution loop: it wraps your coding agent (for example OpenCode), records every prompt, tool call, and outcome to a structured trace store, and uses a meta-agent to propose edits to the agent's "DNA" -- its system prompt, hooks, and skill catalog -- based on observed failures.

## What is in this repo

- `docs/PRD.md` -- product requirements
- `docs/ROADMAP.md` -- three-phase delivery plan
- `harness/` -- version-controlled agent DNA: `system_prompt.txt`, `hooks/`, `skills/`
- `src/foundry_x/` -- runtime code
  - `trace/` -- `TraceLogger`: wraps the agent and persists execution traces
  - `execution/` -- runner that drives an OpenCode-style agent session
  - `evolution/` -- `Digester`, `Evolver`, `Critic`: failure analysis -> edit proposal -> gatekeeping loop
- `infra/` -- Docker and llama.cpp ROCm setup helpers
- `tests/` -- pytest suite
- `logs/` -- runtime trace storage (gitignored)
- `benchmarks/` -- standardized coding benchmarks consumed by the `Critic`

## Quick start

```bash
uv sync
cp .env.example .env       # then edit
uv run python -m foundry_x.execution.runner --task "hello world"
uv run pytest              # smoke test for TraceLogger
```

## Hardware targets

Tested against an AMD Radeon RX 6600 XT (ROCm) host with a Linux Mint / Ryzen 5 5600G control plane. Model-agnostic: runs against any llama.cpp-compatible server or any remote OpenAI-compatible endpoint configured via `OPENCODE_SERVER_URL`.

## Where to look first

- Want to **observe** your agent? Start in `src/foundry_x/trace/logger.py` and `src/foundry_x/execution/runner.py`.
- Want to **analyze failures**? Start in `src/foundry_x/evolution/digester.py`.
- Want to **evolve the harness**? Start in `src/foundry_x/evolution/evolver.py` and `critic.py`.
- Want to **run a local model**? Start in `infra/llama-cpp/README.md`.

## License

See `LICENSE` (add your preferred license here).
