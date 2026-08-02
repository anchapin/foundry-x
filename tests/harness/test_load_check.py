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
def test_load_check_passes_against_real_harness_dir() -> None:
    """Against the canonical ``harness/`` directory the script must exit 0."""
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", "harness"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
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
        check=False,
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
        check=False,
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
        check=False,
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
        check=False,
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
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on manifest/disk drift; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "phantom" in proc.stderr, (
        f"stderr must name the missing hook (issue #277); got {proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_skill_file_on_disk_not_in_manifest(tmp_path: Path) -> None:
    """A skill file on disk that is not declared in manifest.json must trip the
    Critic gate (issue #1010). This is the reverse of issue #277: instead of
    manifest→disk drift we catch disk→manifest drift."""
    harness = _make_fixture_harness(
        tmp_path,
        skills={
            "real.json": _VALID_SKILL,
            "undeclared_skill.json": _VALID_SKILL,
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base"],
            "skills": ["real.json"],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on disk→manifest drift; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "undeclared_skill.json" in proc.stderr, (
        f"stderr must name the undeclared skill file (issue #1010); got {proc.stderr!r}"
    )
    assert "issue #1010" in proc.stderr, f"stderr must reference issue #1010; got {proc.stderr!r}"


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_hook_file_on_disk_not_in_manifest(tmp_path: Path) -> None:
    """A hook file on disk that is not declared in manifest.json must trip the
    Critic gate (issue #1010). This is the reverse of issue #277 for hooks."""
    harness = _make_fixture_harness(
        tmp_path,
        skills={"real.json": _VALID_SKILL},
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base"],
            "skills": ["real.json"],
        },
    )
    extra_hook = tmp_path / "harness" / "hooks" / "undeclared_hook.py"
    extra_hook.write_text("class UndeclaredHook:\n    pass\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on disk→manifest drift; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "undeclared_hook.py" in proc.stderr, (
        f"stderr must name the undeclared hook file (issue #1010); got {proc.stderr!r}"
    )
    assert "issue #1010" in proc.stderr, f"stderr must reference issue #1010; got {proc.stderr!r}"


# ---------------------------------------------------------------------------
# Hook-order validation tests (issue #567)
# ---------------------------------------------------------------------------


def _make_fixture_harness_with_hooks(
    tmp_path: Path,
    skills: dict[str, dict | str],
    manifest: dict,
    hook_modules: dict[str, str],
    system_prompt: str = "persona directive line 1\n",
) -> Path:
    """Build a temporary harness tree with custom hook modules.

    ``hook_modules`` maps filename to module body text. Each module can define
    a class with ``_phase = <int>`` to declare its execution order.
    """
    harness = tmp_path / "harness"
    skills_dir = harness / "skills"
    skills_dir.mkdir(parents=True)
    for fname, payload in skills.items():
        body = json.dumps(payload) if isinstance(payload, dict) else payload
        (skills_dir / fname).write_text(body, encoding="utf-8")
    (harness / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    hooks_dir = harness / "hooks"
    hooks_dir.mkdir()
    (hooks_dir / "__init__.py").write_text(
        "from .base import HookRegistry, get_registry, register_hook\n"
        "__all__ = ['HookRegistry', 'get_registry', 'register_hook']\n",
        encoding="utf-8",
    )
    (hooks_dir / "base.py").write_text(
        textwrap.dedent(
            """\
            class HookRegistry:
                def __init__(self) -> None:
                    self._hooks = []
                def register(self, hook: object) -> None:
                    self._hooks.append(hook)
            def get_registry() -> HookRegistry:
                return HookRegistry()
            def register_hook(hook: object) -> None:
                pass
            """
        ),
        encoding="utf-8",
    )
    for fname, body in hook_modules.items():
        (hooks_dir / fname).write_text(body, encoding="utf-8")
    (harness / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return harness


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_hook_order_validation_passes_when_order_matches(tmp_path: Path) -> None:
    """When manifest order and _phase values agree, validation succeeds (issue #567)."""
    harness = _make_fixture_harness_with_hooks(
        tmp_path,
        skills={"real.json": _VALID_SKILL},
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["injection_firewall", "context_pruning", "rate_limit"],
            "skills": ["real.json"],
        },
        hook_modules={
            "injection_firewall.py": textwrap.dedent(
                """\
                class InjectionFirewallHook:
                    _phase = 0
                """
            ),
            "context_pruning.py": textwrap.dedent(
                """\
                class ContextPruningHook:
                    _phase = 1
                """
            ),
            "rate_limit.py": textwrap.dedent(
                """\
                class RateLimitHook:
                    _phase = 2
                """
            ),
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode == 0, (
        f"load_check should pass when hook order matches; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "hook-order validated" in proc.stdout


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_hook_order_validation_fails_when_order_mismatches(tmp_path: Path) -> None:
    """When manifest order disagrees with _phase values, validation fails (issue #567)."""
    harness = _make_fixture_harness_with_hooks(
        tmp_path,
        skills={"real.json": _VALID_SKILL},
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["injection_firewall", "context_pruning"],
            "skills": ["real.json"],
        },
        hook_modules={
            "injection_firewall.py": textwrap.dedent(
                """\
                class InjectionFirewallHook:
                    _phase = 1
                """
            ),
            "context_pruning.py": textwrap.dedent(
                """\
                class ContextPruningHook:
                    _phase = 0
                """
            ),
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail when hook order mismatches; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "injection_firewall" in proc.stderr
    assert "_phase=1" in proc.stderr


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_hook_order_validation_skips_without_phase(tmp_path: Path) -> None:
    """When a hook has no _phase, validation skips with a warning (ADR-0019)."""
    harness = _make_fixture_harness_with_hooks(
        tmp_path,
        skills={"real.json": _VALID_SKILL},
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["injection_firewall", "rate_limit"],
            "skills": ["real.json"],
        },
        hook_modules={
            "injection_firewall.py": textwrap.dedent(
                """\
                class InjectionFirewallHook:
                    _phase = 0
                """
            ),
            "rate_limit.py": textwrap.dedent(
                """\
                class RateLimitHook:
                    pass
                """
            ),
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode == 0, (
        f"load_check should pass (skip) when hook has no _phase; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "WARN" in proc.stderr
    assert "rate_limit" in proc.stderr
    assert "skipping order validation" in proc.stderr


# ---------------------------------------------------------------------------
# skill_inventory validation tests (issue #1463)
# ---------------------------------------------------------------------------

_MANIFEST_SKILL_INV_OK = {
    "version": "0.1.0",
    "model_target": "test/model",
    "hooks": ["base"],
    "skills": ["alpha.json", "beta.json"],
    "skill_inventory": [
        {"name": "alpha", "description": "Alpha skill"},
        {"name": "beta", "description": "Beta skill"},
    ],
}


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_passes_when_skill_inventory_matches_skills(tmp_path: Path) -> None:
    """Issue #1463: when skill_inventory names == skills[] names, check passes."""
    harness = _make_fixture_harness_with_hooks(
        tmp_path,
        skills={
            "alpha.json": _VALID_SKILL | {"name": "alpha"},
            "beta.json": _VALID_SKILL | {"name": "beta"},
        },
        manifest=_MANIFEST_SKILL_INV_OK,
        hook_modules={},
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode == 0, (
        f"load_check should pass when skill_inventory is consistent; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_skill_inventory_names_diverge_from_skills(
    tmp_path: Path,
) -> None:
    """Issue #1463: skill_inventory names set != skills[] names set must fail."""
    harness = _make_fixture_harness(
        tmp_path,
        skills={
            "alpha.json": _VALID_SKILL | {"name": "alpha"},
            "beta.json": _VALID_SKILL | {"name": "beta"},
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base"],
            "skills": ["alpha.json", "beta.json"],
            "skill_inventory": [
                {"name": "alpha", "description": "Alpha"},
                {"name": "gamma", "description": "not in skills[]"},
            ],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on skill_inventory/skills[] desync; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "gamma" in proc.stderr
    assert "beta" in proc.stderr
    assert "issue #1463" in proc.stderr


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_skill_inventory_entry_missing_name(
    tmp_path: Path,
) -> None:
    """Issue #1463: a skill_inventory entry without 'name' must fail."""
    harness = _make_fixture_harness(
        tmp_path,
        skills={
            "alpha.json": _VALID_SKILL | {"name": "alpha"},
            "beta.json": _VALID_SKILL | {"name": "beta"},
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base"],
            "skills": ["alpha.json", "beta.json"],
            "skill_inventory": [
                {"name": "alpha", "description": "Alpha"},
                {"description": "missing name key"},
            ],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on missing name in skill_inventory; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "missing required key 'name'" in proc.stderr


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_skill_inventory_has_duplicate_names(
    tmp_path: Path,
) -> None:
    """Issue #1463: duplicate names in skill_inventory must fail."""
    harness = _make_fixture_harness(
        tmp_path,
        skills={"alpha.json": _VALID_SKILL | {"name": "alpha"}},
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
        manifest={
            "version": "0.1.0",
            "model_target": "test/model",
            "hooks": ["base"],
            "skills": ["alpha.json"],
            "skill_inventory": [
                {"name": "alpha", "description": "first"},
                {"name": "alpha", "description": "duplicate"},
            ],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(harness)],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on duplicate skill_inventory names; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "duplicate" in proc.stderr


# ---------------------------------------------------------------------------
# Duplicate skill-name detection tests (issue #1460)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_two_skill_files_share_name(tmp_path: Path) -> None:
    """Issue #1460: two skill files with the same ``doc["name"]`` must fail.
    The runner globs ``skills/*.json`` and registers each ``doc["name"]`` as
    a tool name; duplicates cause ambiguous tool dispatch."""
    _make_fixture_harness(
        tmp_path,
        skills={
            "dup_a.json": _VALID_SKILL | {"name": "dup"},
            "dup_b.json": _VALID_SKILL | {"name": "dup"},
        },
        include_hooks=True,
        hooks_init="",
        hooks_base=_MINIMAL_HOOKS_BASE,
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on duplicate skill names; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "duplicate skill name" in proc.stderr, (
        f"stderr must mention duplicate skill name (issue #1460); got {proc.stderr!r}"
    )
    assert "'dup'" in proc.stderr, (
        f"stderr must name the duplicated name (issue #1460); got {proc.stderr!r}"
    )


@pytest.mark.skipif(not LOAD_CHECK.exists(), reason="harness/scripts/load_check.py missing")
def test_load_check_fails_when_manifest_skills_has_duplicate_entry(tmp_path: Path) -> None:
    """Issue #1460: manifest ``skills[]`` listing the same filename twice must
    fail. The runner would register the same tool twice, causing ambiguous
    dispatch."""
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
            "skills": ["real.json", "real.json"],
        },
    )
    proc = subprocess.run(
        [sys.executable, str(LOAD_CHECK), "--harness-dir", str(tmp_path / "harness")],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": ""},
        check=False,
    )
    assert proc.returncode != 0, (
        f"load_check should fail on duplicate manifest skills[] entry; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "duplicate" in proc.stderr, (
        f"stderr must mention duplicate (issue #1460); got {proc.stderr!r}"
    )
    assert "real.json" in proc.stderr, (
        f"stderr must name the duplicated entry (issue #1460); got {proc.stderr!r}"
    )
