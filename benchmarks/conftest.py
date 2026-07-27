"""Shared pytest fixtures for the benchmark suite.

This conftest makes the ``benchmark_workspace`` fixture available to every
test under ``benchmarks/`` (and its subdirectories, e.g. ``tasks/``). It is
the isolation boundary the local development path relies on until full
Docker sandboxing lands (SECURITY.md, "Sandbox").

The ``benchmark_workspace`` fixture
-----------------------------------
``benchmark_workspace`` yields a :class:`pathlib.Path` to an empty, private
temporary directory that is removed automatically when the test finishes
(it is layered on pytest's built-in ``tmp_path``, so teardown is handled by
pytest itself). Benchmark tasks MUST treat this directory as the agent's
entire filesystem view: write outputs here, read seeded inputs here, and
never touch the repository tree.

Seeding static fixtures (optional)
----------------------------------
A task that needs static inputs from ``benchmarks/fixtures/<name>/`` asks
the fixture to copy them in via indirect parametrization::

    @pytest.mark.parametrize(
        "benchmark_workspace", ["sort_a_list"], indirect=True
    )
    @pytest.mark.benchmark
    def test_sort_a_list(benchmark_workspace: Path) -> None:
        assert (benchmark_workspace / "input.txt").exists()

When the parameter is omitted (the common case) the workspace is empty. If
the named fixture directory does not exist, the fixture raises
``FileNotFoundError`` at setup time so the mistake surfaces loudly rather
than silently producing a false pass (PHILOSOPHY.md, "Evidence over
opinion").

Skip patterns
-------------
:func:`_seed_workspace` accepts a ``skip`` keyword — a set of directory/file
names to exclude while copying. It defaults to
:data:`DEFAULT_SKIP` (``{"__pycache__", ".git", ".venv"}``) so tasks no
longer reimplement the same boilerplate::

    _seed_workspace(workspace, "my_fixture", skip={"node_modules", "build"})

An explicit ``skip`` **replaces** the default set (it does not merge), giving
the caller full control.

Template expansion (optional)
-----------------------------
If the fixture directory contains a ``template.env`` file, each non-empty,
non-comment line is treated as a relative path to a file already copied into
the workspace. ``{{ENV_VAR}}`` placeholders inside those files are replaced
with the value of the corresponding environment variable::

    # template.env
    config/settings.toml

    # config/settings.toml (inside the fixture, before expansion)
    api_key = "{{BENCHMARK_API_KEY}}"

If a referenced environment variable is unset, ``_seed_workspace`` raises
``KeyError`` with a clear message. When no ``template.env`` exists the
fixture behaves exactly as before (backwards compatible).

Fixture manifest (optional)
---------------------------
A ``fixture.toml`` or ``fixture.json`` placed in the fixture root describes
how to seed the workspace. Recognised top-level keys:

``skip`` (list[str])
    Directory/file names to exclude. Used only when no explicit ``skip``
    argument is passed to :func:`_seed_workspace`; it **replaces** the
    default skip set.

Example ``fixture.toml``::

    skip = ["__pycache__", ".git", ".venv", "node_modules"]

The manifest file (``fixture.toml`` / ``fixture.json``) and ``template.env``
are metadata: they are never copied into the workspace.

Isolation guarantee
-------------------
The yielded path is unique per test invocation; no two tests share it and
nothing escapes into the repository tree. That is what makes the local
development path safe in lieu of full Docker sandboxing.

See also ``benchmarks/README.md`` ("The benchmark_workspace fixture") and
ADR-0004 / ADR-0005.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

#: Root directory for static benchmark fixture data (``benchmarks/fixtures/``).
FIXTURES_ROOT = Path(__file__).parent / "fixtures"

#: Directory/file names excluded by default when seeding a workspace.
DEFAULT_SKIP: frozenset[str] = frozenset({"__pycache__", ".git", ".venv"})

#: Seed-control metadata files that must never leak into the workspace.
_SEED_METADATA_FILES: frozenset[str] = frozenset({"template.env", "fixture.toml", "fixture.json"})

#: Matches ``{{ENV_VAR}}`` placeholders inside templated fixture files.
_TEMPLATE_RE = re.compile(r"\{\{(\w+)\}\}")


def _read_manifest(source: Path) -> dict[str, object]:
    """Parse ``fixture.toml`` / ``fixture.json`` from *source* if present.

    Returns an empty dict when neither file exists.
    """
    toml_path = source / "fixture.toml"
    if toml_path.is_file():
        with open(toml_path, "rb") as fh:
            return tomllib.load(fh)  # type: ignore[return-value]
    json_path = source / "fixture.json"
    if json_path.is_file():
        return json.loads(json_path.read_text(encoding="utf-8"))
    return {}


def _expand_templates(workspace: Path, source: Path) -> None:
    """Expand ``{{ENV_VAR}}`` placeholders in files listed by ``template.env``.

    ``template.env`` (if present in *source*) contains one relative file path
    per line (``#`` comments and blank lines are ignored). Each referenced
    file is read from *workspace* and every ``{{VAR}}`` token replaced with
    ``os.environ[VAR]``.

    Raises:
        KeyError: if a placeholder references an unset environment variable.
    """
    template_manifest = source / "template.env"
    if not template_manifest.is_file():
        return
    for line in template_manifest.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel or rel.startswith("#"):
            continue
        target = workspace / rel
        if not target.is_file():
            continue
        content = target.read_text(encoding="utf-8")
        missing = [var for var in _TEMPLATE_RE.findall(content) if var not in os.environ]
        if missing:
            raise KeyError(
                f"template expansion failed: environment variable(s) "
                f"{missing!r} not set (required by {rel} in template.env)"
            )
        expanded = _TEMPLATE_RE.sub(lambda m: os.environ[m.group(1)], content)
        target.write_text(expanded, encoding="utf-8")


def _seed_workspace(
    workspace: Path,
    seed_name: str,
    *,
    skip: set[str] | None = None,
) -> None:
    """Copy ``benchmarks/fixtures/<seed_name>/`` into ``workspace``.

    Supports optional skip patterns, ``template.env`` expansion, and a
    ``fixture.toml``/``fixture.json`` manifest. See the module docstring for
    full details.

    Args:
        workspace: destination directory (created by ``shutil.copytree``).
        seed_name: subdirectory of ``benchmarks/fixtures/`` to copy.
        skip: directory/file names to exclude. When ``None`` the skip set is
            read from the fixture manifest, falling back to
            :data:`DEFAULT_SKIP`. An explicit value **replaces** both.

    Raises:
        FileNotFoundError: if the named fixture directory is missing.
        KeyError: if a ``template.env`` placeholder references an unset env var.
    """
    source = FIXTURES_ROOT / seed_name
    if not source.is_dir():
        raise FileNotFoundError(
            f"benchmark fixture directory not found: {source} "
            "(requested via @pytest.mark.parametrize(..., indirect=True))"
        )

    # Resolve the effective skip set: explicit param > manifest > defaults.
    if skip is not None:
        effective_skip = set(skip)
    else:
        manifest = _read_manifest(source)
        manifest_skip = manifest.get("skip")
        if isinstance(manifest_skip, list):
            effective_skip = set(manifest_skip)
        else:
            effective_skip = set(DEFAULT_SKIP)

    # Metadata files must never leak into the agent's workspace.
    effective_skip |= set(_SEED_METADATA_FILES)

    def _ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in effective_skip}

    shutil.copytree(source, workspace, dirs_exist_ok=True, ignore=_ignore)
    _expand_templates(workspace, source)


@pytest.fixture
def benchmark_workspace(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path]:
    """Yield an isolated, per-test working directory for a benchmark task.

    Args:
        request: pytest request object. ``request.param`` may name a
            subdirectory of ``benchmarks/fixtures/`` whose contents are copied
            into the workspace before it is yielded (optional; the workspace
            is empty when unset).
        tmp_path: pytest's per-test temporary directory, used as the parent so
            pytest owns cleanup.

    Yields:
        An empty (or seeded) :class:`~pathlib.Path` to a temp directory.
    """
    workspace = tmp_path / "benchmark_workspace"
    workspace.mkdir()

    seed_name: str | None = getattr(request, "param", None)
    if not seed_name:
        seed_name = os.environ.get("PYTEST_BENCHMARK_FIXTURE")
    if seed_name:
        _seed_workspace(workspace, seed_name)

    yield workspace
    # ``tmp_path`` (and everything beneath it) is removed by pytest at the end
    # of the session, so explicit cleanup here is unnecessary.
