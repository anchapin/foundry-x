"""Dockerfile build validation for infra/docker/Dockerfile.

Also enforces the runtime image size regression guard for issue #286:
the acceptance criterion in Dockerfile:23-25 (``docker images
foundryx:latest`` must report a smaller byte count than the
single-stage predecessor) was a one-time manual check from #116 with
no ongoing enforcement. ``test_runtime_image_size_within_baseline``
converts that contract into a CI ceiling so a future dependency
addition cannot silently inflate the runtime image.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "infra" / "docker" / "Dockerfile"
IMAGE_SIZE_BASELINE = ROOT / "tests" / "infra" / "image_size_baseline.json"

#: The image ref produced by ``docker compose build`` / ``docker build``
#: (see infra/docker/docker-compose.yml:84). Used both for the size
#: inspection and for the "is it built yet?" skip gate.
FOUNDRYX_IMAGE = "foundryx:latest"


def _docker_binary() -> str:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker binary is not on PATH; skipping Dockerfile build check")
    return docker


def _docker_supports_build_check(docker: str) -> bool:
    result = subprocess.run(
        [docker, "build", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0 and "--check" in result.stdout


def _build_context(tmp_path: Path) -> Path:
    context = tmp_path / "context"
    dockerfile = context / "infra" / "docker" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    shutil.copy2(DOCKERFILE, dockerfile)
    shutil.copy2(ROOT / "pyproject.toml", context / "pyproject.toml")
    if (ROOT / "uv.lock").exists():
        shutil.copy2(ROOT / "uv.lock", context / "uv.lock")
    for name in ("src", "harness", "tests"):
        directory = context / name
        directory.mkdir(parents=True)
        (directory / ".keep").write_text("", encoding="utf-8")
    return context


def test_dockerfile_build_configuration_is_valid(tmp_path: Path) -> None:
    docker = _docker_binary()
    if not _docker_supports_build_check(docker):
        pytest.skip("docker build --check is unavailable; skipping full image build to avoid pulls")

    context = _build_context(tmp_path)
    result = subprocess.run(
        [docker, "build", "--check", "-f", "infra/docker/Dockerfile", "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Runtime image size regression guard (issue #286)
# ---------------------------------------------------------------------------
#
# Restates the size acceptance criterion from infra/docker/Dockerfile:23-25
# (itself part of issue #116) as an enforced ceiling. The image is NOT
# built by these tests: building is the job of .github/workflows/docker.yml.
# If foundryx:latest is absent the size test skips, which keeps the default
# ``uv run pytest`` in ci.yml fast and free of network pulls.


def _load_baseline() -> dict[str, Any]:
    """Parse and validate the size baseline contract file.

    Kept strict on purpose: a malformed baseline must fail loudly and
    early (AGENTS.md §2 — never silently swallow) rather than producing
    a misleading pass.
    """
    assert IMAGE_SIZE_BASELINE.exists(), (
        f"baseline file missing: {IMAGE_SIZE_BASELINE.relative_to(ROOT)} (required by issue #286)"
    )
    try:
        data = json.loads(IMAGE_SIZE_BASELINE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"{IMAGE_SIZE_BASELINE.relative_to(ROOT)} is not valid JSON: {exc}"
        ) from exc

    for key in ("baseline_bytes", "margin_percent"):
        assert key in data, (
            f"{IMAGE_SIZE_BASELINE.relative_to(ROOT)} missing required key "
            f"'{key}' (issue #286 contract)"
        )
    assert isinstance(data["baseline_bytes"], int) and data["baseline_bytes"] > 0, (
        "baseline_bytes must be a positive integer"
    )
    assert (
        isinstance(data["margin_percent"], (int, float)) and 0 <= data["margin_percent"] <= 100
    ), "margin_percent must be a number in [0, 100]"
    return data


def _ceiling_bytes(baseline: dict[str, Any]) -> int:
    """baseline_bytes grown by margin_percent, rounded up to the byte."""
    return math.ceil(baseline["baseline_bytes"] * (1 + baseline["margin_percent"] / 100))


def _human(size_bytes: int) -> str:
    """1024-based, two-decimal human rendering for failure messages."""
    size = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"  # pragma: no cover - unreachable fallback


def _image_size(docker: str) -> int | None:
    """Return the size of foundryx:latest in bytes, or None if not built."""
    result = subprocess.run(
        [docker, "image", "inspect", FOUNDRYX_IMAGE, "--format", "{{.Size}}"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    return int(result.stdout.strip())


def test_image_size_baseline_file_is_well_formed() -> None:
    """The baseline file parses and carries the issue #286 contract fields.

    Runs without docker so a botched edit to the JSON is caught in the
    default ``uv run pytest`` run, not only inside the docker workflow.
    """
    baseline = _load_baseline()
    # Sanity: the ceiling must be strictly larger than the baseline so a
    # size-equal build still passes (layer jitter should not flake).
    assert _ceiling_bytes(baseline) > baseline["baseline_bytes"]


# ---------------------------------------------------------------------------
# BuildKit cache mount regression guard (issue #640)
# ---------------------------------------------------------------------------


def _builder_stage_lines() -> list[str]:
    """Return lines belonging to the builder stage, excluding the runtime stage."""
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    builder_lines: list[str] = []
    in_builder = False
    for line in lines:
        if line.startswith("# ---- Stage 1: builder"):
            in_builder = True
            continue
        if line.startswith("# ---- Stage 2: runtime"):
            break
        if in_builder:
            builder_lines.append(line)
    return builder_lines


def test_buildkit_cache_mount_present_in_builder_stage() -> None:
    """The builder stage declares a BuildKit cache mount for uv's resolver state.

    Issue #640: the Dockerfile comment (lines 78-82) and the acceptance
    criteria from issue #116 require that a second consecutive ``docker build``
    reuses uv's resolver state via a BuildKit cache mount rather than
    re-resolving the lockfile on every rebuild. This test asserts the
    ``--mount=type=cache,target=/root/.cache/uv`` syntax is present in the
    builder stage so a future Dockerfile edit cannot accidentally drop it.
    """
    builder_lines = _builder_stage_lines()
    cache_mount_lines = [
        line.strip()
        for line in builder_lines
        if "--mount=type=cache" in line and "target=/root/.cache/uv" in line
    ]
    assert cache_mount_lines, (
        "No BuildKit cache mount targeting /root/.cache/uv found in the "
        "builder stage of infra/docker/Dockerfile. Issue #640 requires this "
        "mount to persist uv's resolver state across rebuilds. Add:\n"
        "  RUN --mount=type=cache,target=/root/.cache/uv \\\n"
        "      uv sync --frozen --no-dev\n"
        "to the builder stage."
    )


def test_uv_sync_uses_frozen_flag() -> None:
    """The uv sync invocation in the builder stage uses --frozen.

    Issue #640: the Dockerfile comment (line 82) and the acceptance criteria
    from issue #116 require that ``uv sync`` runs with ``--frozen`` so the
    resolved package set cannot drift between rebuilds. This test asserts
    the flag is present in the builder stage so a future edit cannot silently
    remove it and reintroduce the reproducibility hole.
    """
    builder_lines = _builder_stage_lines()
    sync_lines = [line.strip() for line in builder_lines if "uv sync" in line]
    assert sync_lines, (
        "No 'uv sync' invocation found in the builder stage of "
        "infra/docker/Dockerfile. The builder stage must run 'uv sync' to "
        "install dependencies."
    )
    frozen_lines = [line for line in sync_lines if "--frozen" in line]
    assert frozen_lines, (
        "The 'uv sync' invocation in the builder stage of "
        "infra/docker/Dockerfile does not use --frozen. Issue #640 requires "
        "--frozen to pin the resolved package set and prevent silent drift "
        "between rebuilds. Add --frozen to the 'uv sync' command."
    )


def test_runtime_image_size_within_baseline() -> None:
    """foundryx:latest must stay under baseline + documented margin.

    Issue #286: the documented size contract (Dockerfile:23-25) becomes
    a regression guard. On breach the assertion names the baseline, the
    observed size, and points the PR author at the baseline file to bump
    with evidence — exactly acceptance criterion 3.
    """
    docker = _docker_binary()

    observed = _image_size(docker)
    if observed is None:
        pytest.skip(
            f"{FOUNDRYX_IMAGE} is not built; the image-size guard runs in "
            f".github/workflows/docker.yml. To run locally, build first: "
            f"`docker build -f infra/docker/Dockerfile -t {FOUNDRYX_IMAGE} .`"
        )

    baseline = _load_baseline()
    ceiling = _ceiling_bytes(baseline)

    assert observed <= ceiling, (
        f"{FOUNDRYX_IMAGE} is {observed} bytes ({_human(observed)}), which "
        f"exceeds the size ceiling of {ceiling} bytes ({_human(ceiling)}) = "
        f"baseline {baseline['baseline_bytes']} bytes "
        f"({_human(baseline['baseline_bytes'])}) + {baseline['margin_percent']}% "
        f"margin. A growing runtime image slows every `docker compose` run "
        f"and widens the supply-chain surface (docs/SECURITY.md threat #3). "
        f"If this growth is legitimate, rebuild locally, record the new "
        f"`docker image inspect {FOUNDRYX_IMAGE} --format '{{{{.Size}}}}'`, "
        f"and bump baseline_bytes in "
        f"{IMAGE_SIZE_BASELINE.relative_to(ROOT)} in THIS PR with that "
        f"evidence."
    )


def test_runtime_stage_has_healthcheck() -> None:
    """Runtime stage must carry a HEALTHCHECK instruction (issue #813).

    The runtime container should self-test so orchestrators can distinguish a
    genuine startup failure from a process that reports ready prematurely.
    """
    dockerfile_text = DOCKERFILE.read_text(encoding="utf-8")

    lines = dockerfile_text.splitlines()
    runtime_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("FROM python") and "AS runtime" in stripped:
            runtime_start = i
            break

    assert runtime_start is not None, "runtime stage (FROM ... AS runtime) not found in Dockerfile"

    runtime_section = "\n".join(lines[runtime_start:])
    assert "HEALTHCHECK" in runtime_section, (
        "HEALTHCHECK not found in runtime stage. "
        "Add HEALTHCHECK to the runtime stage of infra/docker/Dockerfile (issue #813)."
    )
