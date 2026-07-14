"""Smoke-test for ``harness/scripts/load_check.py`` (issue #107).

We invoke the script as a subprocess against a fixture harness tree and
against the real ``harness/`` directory. Subprocess invocation (rather than
direct import + call) ensures the test exercises the public CLI surface and
that ``sys.path`` manipulation inside the script is sound.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOAD_CHECK = REPO_ROOT / "harness" / "scripts" / "load_check.py"


def _make_fixture_harness(
    tmp_path: Path,
    skills: dict[str, dict | str],
    system_prompt: str = "persona directive line 1\n",
    include_hooks: bool = False,
    hooks_init: str = "",
    hooks_base: str = "",
    manifest: dict | None = None,
) -> Path:
    """Build a temporary harness tree under ``tmp_path``.

    Each ``skills`` key is a filename; the value is either a parsed ``dict``
    (written as JSON) or a raw ``str`` (written verbatim — used to inject
    deliberately-broken JSON). ``include_hooks`` controls whether the
    fixture also contains a minimal ``harness/hooks`` package so the script
    can complete its import check. ``manifest`` (issue #277), when given,
    is written to ``harness/manifest.json`` so cross-ref validation can be
    exercised.
    """
    harness = tmp_path / "harness"
    skills_dir = harness / "skills"
    skills_dir.mkdir(parents=True)
    for fname, payload in skills.items():
        body = json.dumps(payload) if isinstance(payload, dict) else payload
        (skills_dir / fname).write_text(body, encoding="utf-8")
    (harness / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    if include_hooks:
        hooks_dir = harness / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "__init__.py").write_text(hooks_init, encoding="utf-8")
        (hooks_dir / "base.py").write_text(hooks_base, encoding="utf-8")
    if manifest is not None:
        (harness / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return harness


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
@pytest.mark.xfail(
    reason=(
        "Issue #278: harness/skills/example_skill.json declares name='read_file', "
        "which violates the new filename-to-name invariant enforced by load_check. "
        "Renaming the skill file is a harness DNA edit (AGENTS.md \u00a72) and must "
        "go through the Evolver -> Critic loop (ADR-0004). This strict xfail tracks "
        "that Evolver target: once the rename lands, the test xpasses and the strict "
        "marker flips it red to remind us to remove the marker."
    ),
    strict=True,
)
def test_load_check_passes_against_real_harness_dir() -> None:
    """Against the canonical ``harness/`` directory the script must exit 0."""
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", "harness"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"load_check failed against real harness; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "load-check OK" in proc.stdout


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_reports_broken_skill(tmp_path: Path) -> None:
    """A harness with deliberately-broken JSON in one skill must exit non-zero
    and name the broken file on stderr. Issue #107 acceptance."""
    _make_fixture_harness(
        tmp_path,
        skills={
            # Real-shaped valid skill so we know only the broken one trips us.
            "good.json": {
                "name": "good",
                "version": "0.1.0",
                "description": "valid",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            # Deliberately malformed.
            "broken.json": "{ not json",
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=textwrap.dedent(
            """\
            # Minimal hooks package: provides HookRegistry for load_check.
            class HookRegistry:
                def __init__(self) -> None:
                    self._hooks = []
                def register(self, hook: object) -> None:
                    self._hooks.append(hook)

            def get_registry() -> HookRegistry:
                return HookRegistry()
            """
        ),
    )

    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        # The fixture tree has its own package layout that may shadow the
        # project-level one; isolate PYTHONPATH so we exercise only the
        # script's own sys.path handling (parent of --harness-dir).
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode != 0, (
        f"load_check should have failed; stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "broken.json" in proc.stderr, (
        f"stderr must name the broken file (issue #107); got {proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_reports_skill_name_filename_mismatch(tmp_path: Path) -> None:
    """Issue #278: when ``skills/<stem>.json`` has ``doc['name'] != <stem>``
    the script must exit non-zero and name both the filename and the internal
    name on stderr. The runner globs by filename but exposes ``doc['name']``
    as the tool name (runner.py ``_load_tool_definitions``), so a divergence
    must surface at the Critic gate, not at runtime."""
    _make_fixture_harness(
        tmp_path,
        skills={
            "good.json": {
                "name": "good",
                "version": "0.1.0",
                "description": "valid",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            "mismatch.json": {
                "name": "other",
                "version": "0.1.0",
                "description": "name does not match filename stem",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=textwrap.dedent(
            """\
            # Minimal hooks package: provides HookRegistry for load_check.
            class HookRegistry:
                def __init__(self) -> None:
                    self._hooks = []
                def register(self, hook: object) -> None:
                    self._hooks.append(hook)

            def get_registry() -> HookRegistry:
                return HookRegistry()
            """
        ),
    )

    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        # The fixture tree has its own package layout that may shadow the
        # project-level one; isolate PYTHONPATH so we exercise only the
        # script's own sys.path handling (parent of --harness-dir).
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode != 0, (
        f"load_check should have failed on name/filename mismatch; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "mismatch.json" in proc.stderr, (
        f"stderr must name the offending file (issue #278); got {proc.stderr!r}"
    )
    assert "'other'" in proc.stderr, (
        f"stderr must name the internal name (issue #278); got {proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_reports_skill_name_filename_mismatch(tmp_path: Path) -> None:
    """Issue #278: when ``skills/<stem>.json`` has ``doc['name'] != <stem>``
    the script must exit non-zero and name both the filename and the internal
    name on stderr. The runner globs by filename but exposes ``doc['name']``
    as the tool name (runner.py ``_load_tool_definitions``), so a divergence
    must surface at the Critic gate, not at runtime."""
    _make_fixture_harness(
        tmp_path,
        skills={
            "good.json": {
                "name": "good",
                "version": "0.1.0",
                "description": "valid",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            "mismatch.json": {
                "name": "other",
                "version": "0.1.0",
                "description": "name does not match filename stem",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=textwrap.dedent(
            """\
            # Minimal hooks package: provides HookRegistry for load_check.
            class HookRegistry:
                def __init__(self) -> None:
                    self._hooks = []
                def register(self, hook: object) -> None:
                    self._hooks.append(hook)

            def get_registry() -> HookRegistry:
                return HookRegistry()
            """
        ),
    )

    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        # The fixture tree has its own package layout that may shadow the
        # project-level one; isolate PYTHONPATH so we exercise only the
        # script's own sys.path handling (parent of --harness-dir).
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode != 0, (
        f"load_check should have failed on name/filename mismatch; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "mismatch.json" in proc.stderr, (
        f"stderr must name the offending file (issue #278); got {proc.stderr!r}"
    )
    assert "'other'" in proc.stderr, (
        f"stderr must name the internal name (issue #278); got {proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_exits_2_for_missing_dir() -> None:
    """When ``--harness-dir`` points at a non-existent path the script exits 2
    (usage-level error, distinct from invariant failure)."""
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", "/nonexistent/harness"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 2
    assert "does not exist" in proc.stderr


_MINIMAL_HOOKS_BASE = textwrap.dedent(
    """\
    # Minimal hooks package: provides HookRegistry for load_check.
    class HookRegistry:
        def __init__(self) -> None:
            self._hooks = []
        def register(self, hook: object) -> None:
            self._hooks.append(hook)

    def get_registry() -> HookRegistry:
        return HookRegistry()
    """
)


_VALID_SKILL = {
    "name": "real",
    "version": "0.1.0",
    "description": "valid",
    "input_schema": {"type": "object"},
    "output_schema": {"type": "object"},
}


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_manifest_references_missing_skill(tmp_path: Path) -> None:
    """A manifest naming a skill absent from disk must trip the Critic gate
    (issue #277). The fixture is otherwise valid so that only the manifest
    cross-ref check fails, proving the gate catches manifest↔disk drift."""
    _make_fixture_harness(
        tmp_path,
        skills={"real.json": _VALID_SKILL},
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base"],
            "skills": ["real.json", "ghost.json"],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode != 0, (
        f"load_check should fail on manifest/disk drift; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ghost.json" in proc.stderr, (
        f"stderr must name the missing skill (issue #277); got {proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_manifest_references_missing_hook(tmp_path: Path) -> None:
    """A manifest naming a hook absent from disk must trip the Critic gate
    (issue #277). Symmetric to the missing-skill case."""
    _make_fixture_harness(
        tmp_path,
        skills={"real.json": _VALID_SKILL},
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base", "phantom"],
            "skills": ["real.json"],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert proc.returncode != 0, (
        f"load_check should fail on manifest/disk drift; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "phantom" in proc.stderr, (
        f"stderr must name the missing hook (issue #277); got {proc.stderr!r}"
    )
