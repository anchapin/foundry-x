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

**New to the project?** Walk through the
[getting-started tutorial](docs/TUTORIAL.md) — a guided end-to-end tour
(clone → plant a trace → inspect events → read KPIs) that takes under 30
minutes and runs fully offline.

## CLI entrypoints

Three console scripts ship with `foundry-x` (registered in
`pyproject.toml` §`[project.scripts]`). Each runs offline against the
trace store under `logs/` — no live model required.

| Command | Module | Purpose |
| --- | --- | --- |
| `fx-trace` | [`observability/cli.py`](src/foundry_x/observability/cli.py) | Trace-driven inspection: regression reports, timeline rendering, session roll-ups, tool-latency percentiles |
| `foundry-kpis` | [`observability/kpis.py`](src/foundry_x/observability/kpis.py) | Compute the three PRD success-metric KPIs (cycle time, regression rate, improvement rate) from trace data |
| `foundry-x-trace` (alias `foundry-trace`) | [`trace/cli.py`](src/foundry_x/trace/cli.py) | Inspect and render trace data per ADR-0007: list/show/export sessions, render failure reports, grep events, redact secrets, seed sample data |

Each example below exits 0 against a fresh clone (no trace store yet):

```bash
# --- offline smoke: plant a deterministic session, then explore it ---
uv run foundry-x-trace seed-sample-trace
uv run foundry-x-trace sessions                         # list recorded sessions
uv run foundry-x-trace show <session_id>                # print the event timeline
uv run foundry-kpis                                     # print the three PRD KPIs
uv run fx-trace regression-report                       # aggregate Critic verdicts
uv run fx-trace session-summary                         # one-row-per-session roll-up
uv run fx-trace tool-latency                            # per-tool latency percentiles
```

Without a trace store every command degrades gracefully (prints "No
sessions found." / `N/A` / empty report). Pass `--help` to any command
for the full flag surface.

## Hardware targets

Tested against an AMD Radeon RX 6600 XT (ROCm) host with a Linux Mint / Ryzen 5 5600G control plane. Model-agnostic: runs against any llama.cpp-compatible server or any remote OpenAI-compatible endpoint configured via `OPENCODE_SERVER_URL`.

## Where to look first

- Want to **observe** your agent? Start in `src/foundry_x/trace/logger.py` and `src/foundry_x/execution/runner.py`.
- Want to **analyze failures**? Start in `src/foundry_x/evolution/digester.py`.
- Want to **evolve the harness**? Start in `src/foundry_x/evolution/evolver.py` and `critic.py`.
- Want to **run a local model**? Start in `infra/llama-cpp/README.md`.

## License

See `LICENSE` (add your preferred license here).
