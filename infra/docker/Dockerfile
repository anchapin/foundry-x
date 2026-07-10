# Pin the base image to a digest (issue #124).
# Tag: python:3.11-slim -> 3.11.15-slim-trixie
# Resolved: 2026-07-10 from registry-1.docker.io/v2/library/python/manifests/3.11-slim
FROM python:3.11-slim@sha256:a7bce671c71744cd4dcef2bc9cd9c5eff3d7cfa6b55eca30759765047184bcdb

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin the uv installer to a specific version AND content hash (issue #124).
# UV_VERSION is the GitHub release tag (issue #25). UV_SHA256 is the SHA256
# of the matching x86_64-unknown-linux-gnu release tarball, embedded so a
# compromised release or MITM between the build host and GitHub cannot
# silently swap the binary (docs/SECURITY.md threat #3).
# To bump UV_VERSION: download
#   https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz
# record `sha256sum` output, update UV_SHA256, and re-run
# `uv sync --frozen` to confirm reproducibility.
ARG UV_VERSION=0.9.30
ARG UV_SHA256=8b3762374972daa7a74bbc6896cc73229ca69a07403dd9f9ea3805a51ffd7582
ARG UV_TARGET=uv-x86_64-unknown-linux-gnu

RUN set -eux; \
    curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_TARGET}.tar.gz" -o /tmp/uv.tar.gz; \
    echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum -c -; \
    tar -xzf /tmp/uv.tar.gz -C /usr/local --strip-components=1; \
    rm -f /tmp/uv.tar.gz; \
    uv --version

WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev || uv sync --no-dev
COPY src ./src
COPY harness ./harness
COPY tests ./tests

CMD ["uv", "run", "python", "-m", "foundry_x.execution.runner"]
