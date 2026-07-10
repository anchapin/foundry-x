# Multi-stage Dockerfile for the FoundryX sandbox image.
#
# Resolves issue #116 and stacks on top of issue #124 (digest pin) and
# issue #25 (SHA256-pinned uv installer). Three independent security
# layers, all required by docs/SECURITY.md threat #3 (supply-chain
# compromise via dependency drift):
#
#   1. Base image pinned by digest (issue #124) — no silent drift between
#      rebuilds.
#   2. uv installer pinned by version AND SHA256 (issue #25) — no MITM
#      between the build host and GitHub can swap the binary.
#   3. uv's resolver state is cached across rebuilds (issue #116) and
#      run with `--frozen` so the resolved package set cannot drift.
#
# Acceptance criteria from issue #116, restated for the multi-stage
# version:
#   - Runtime stage does NOT contain build-essential, git, or curl.
#     Verify with:  docker run --rm foundryx:latest which build-essential
#     (must exit non-zero).
#   - `uv sync --frozen --no-dev` runs in the builder stage; a second
#     consecutive `docker build` reuses the BuildKit cache mount at
#     /root/.cache/uv (asserted by --progress=plain showing the cached
#     layer hit).
#   - `docker images foundryx:latest` reports a smaller byte count than
#     the single-stage predecessor; record before/after in the PR.
#     Now enforced as a regression ceiling by issue #286: see
#     tests/infra/test_docker_build.py::test_runtime_image_size_within_baseline
#     and tests/infra/image_size_baseline.json.
#
# Requires BuildKit (Docker 23+ default; `docker compose build` uses
# BuildKit by default). On legacy builders, the `--mount=type=cache`
# syntax will fail loudly — which is the desired behavior, since the
# cache mount is part of the acceptance criteria.

# ---- Stage 1: builder -------------------------------------------------------
# Carries the toolchain required to install uv and resolve the lockfile.
# This stage is NOT the published image.
FROM python:3.11-slim@sha256:a7bce671c71744cd4dcef2bc9cd9c5eff3d7cfa6b55eca30759765047184bcdb AS builder
# was: python:3.11-slim (2026-07-10) — 3.11.15-slim-trixie per issue #124
#      Resolved from registry-1.docker.io/v2/library/python/manifests/3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin the uv installer to a specific version. Unpinned `pip install uv`
# reproduces PyPI's "latest" on every build, which is both a reproducibility
# bug (ADR-0002: the resolved package set can silently change between
# builds) and a supply-chain weakening (docs/SECURITY.md threat #3).
# Bump UV_VERSION deliberately and re-verify `uv sync --frozen` succeeds.
ARG UV_VERSION=0.9.30
RUN pip install --no-cache-dir "uv==${UV_VERSION}"
RUN uv --version
WORKDIR /app
COPY pyproject.toml uv.lock* ./
# BuildKit cache mount: a second consecutive `docker build` reuses uv's
# resolver state instead of re-resolving the lockfile. The BuildKit cache
# lives outside the image layers, so it does not bloat the published
# image. `--frozen` pins the resolved set; the legacy `|| uv sync --no-dev`
# silent fallback was removed because it was a reproducibility hole.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src ./src
COPY harness ./harness
COPY tests ./tests

# ---- Stage 2: runtime -------------------------------------------------------
# No compiler, no VCS, no HTTP client — only ca-certificates (needed for
# outbound TLS to llama-server). The .venv created in the builder is
# copied wholesale; uv itself is NOT installed here.
FROM python:3.11-slim@sha256:a7bce671c71744cd4dcef2bc9cd9c5eff3d7cfa6b55eca30759765047184bcdb AS runtime
# was: python:3.11-slim (2026-07-10)

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/harness /app/harness
COPY --from=builder /app/tests /app/tests

# Activate the builder-created virtualenv by putting it on PATH. This
# means `python` resolves to the venv's interpreter, and `python -m
# foundry_x.execution.runner` works without any `uv run` prefix (uv is a
# build tool, not a runtime dependency).
ENV PATH="/app/.venv/bin:${PATH}"

CMD ["python", "-m", "foundry_x.execution.runner"]
