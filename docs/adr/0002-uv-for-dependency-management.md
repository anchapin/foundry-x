# ADR-0002: Use `uv` for dependency management

## Status

Accepted. 2026-07-10.

## Context

The PRD specifies Python 3.11+ and a fast, reproducible local dev
loop. Options considered:

- `pip` + `requirements.txt`
- `poetry`
- `pdm`
- `uv`

## Decision

We use [`uv`](https://docs.astral.sh/uv/) for:

- dependency resolution and locking (`uv.lock` is committed)
- virtual environment management (`uv sync`, `uv run`)
- running tools (`uv run pytest`, `uv run ruff`)

`pyproject.toml` is the source of truth for dependencies; the
lockfile is the source of truth for reproducible installs.

## Consequences

- All commands in `README.md`, `CONTRIBUTING.md`, and CI use
  `uv run`. We do not document `pip install` paths.
- New contributors need `uv` installed. We document the one-liner
  in `CONTRIBUTING.md`.
- We accept a soft lock-in to the Astral toolchain. If `uv` is
  abandoned, we can migrate because `pyproject.toml` is portable.
- `uv pip audit` runs in CI for supply-chain checks (see
  `docs/SECURITY.md`).
