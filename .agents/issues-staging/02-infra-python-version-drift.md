## Motivation

The Dockerfile base image is `python:3.14-slim` (bumped by dependabot PR #922), but every CI workflow runs `uv python install 3.11`. The entire test suite — lint, unit, benchmark, critic gate — exercises Python 3.11, while the production sandbox container runs Python 3.14. A regression that only manifests on 3.14 (e.g., a stdlib deprecation, a typing change, an asyncio behavior shift, a removed deprecated API) would pass every CI check and ship undetected until the Docker container runs in production.

No existing test asserts that the Dockerfile Python minor version matches the CI Python minor version. Additionally, stale references to `python:3.11-slim` persist in `.github/dependabot.yml:10`, `tests/infra/test_dependabot_config.py:87`, and `tests/test_uv_pinning.py:9,45`.

## Evidence

- `infra/docker/Dockerfile:38` — `FROM python:3.14-slim@sha256:cea0e604...` AS builder
- `infra/docker/Dockerfile:94` — `FROM python:3.14-slim@sha256:cea0e604...` AS runtime
- `.github/workflows/*.yml` — all 9 workflow files use `uv python install 3.11` (12 occurrences total)
- `pyproject.toml:5` — `requires-python = ">=3.11"` allows both versions, so the drift is silent
- `.github/dependabot.yml:10` — stale comment says "digest-pinned python:3.11-slim@sha256"
- PR #922 (merged) — bumped Dockerfile 3.11->3.14; CI workflows not updated

## Risk

Low. Test-only addition + CI workflow version alignment. Bumping CI to 3.14 may surface latent 3.14-incompatible code — but those failures are exactly the regressions this guard is designed to catch.

## Acceptance Criteria

1. `tests/infra/test_python_version_consistency.py` parses the Dockerfile `FROM python:X.Y-slim` line and asserts the minor version matches `uv python install X.Y` across all CI workflows
2. The test fails (not skips) if any workflow's Python minor version differs from the Dockerfile
3. All CI workflows updated to 3.14; `uv run pytest` is green
4. No stale `python:3.11-slim` string literal remains in `.github/dependabot.yml` or test docstrings

## ADR(s)

ADR-0002 — advances (uv-managed Python version consistency across build and CI environments)
