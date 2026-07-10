FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
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
RUN uv sync --frozen --no-dev || uv sync --no-dev
COPY src ./src
COPY harness ./harness
COPY tests ./tests

CMD ["uv", "run", "python", "-m", "foundry_x.execution.runner"]
