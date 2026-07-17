#!/usr/bin/env python3
"""Smoke-test that the harness tree is loadable. Used by the Critic (ADR-0004)
to gate ``ProposedEdit`` proposals before they are marked active.

Validates seven invariants:

* the harness directory itself exists
* every ``harness/skills/*.json`` parses and carries the five required keys
  (``name``, ``version``, ``description``, ``input_schema``, ``output_schema``)
* every ``harness/skills/<stem>.json`` has ``doc['name'] == <stem>`` (issue
  #278): the runner globs by filename but exposes the internal ``name`` as
  the tool name, so a filename/name divergence would let the model see one
  tool name while debugging references point at another
* ``harness/system_prompt.txt`` exists and is non-empty
* ``import harness.hooks`` succeeds and the registry instantiates
* ``harness/manifest.json`` cross-refs resolve on disk (issue #277)
* hook execution order matches manifest declaration (issue #567)

Stdlib-only by design. The script adds the parent of ``--harness-dir`` to
``sys.path`` so that ``import harness.hooks`` resolves the same way the
rest of the foundry does under pytest's ``pythonpath = ["."]``.

Exit codes:
    0  — all invariants pass
    1  — at least one invariant fails (per-failure message on stderr)
    2  — usage error (e.g. missing --harness-dir or non-existent dir)

Issue #107; advances SECURITY.md threat #1 (harness degradation).
Issue #277; closes the manifest↔disk drift gap in the Critic gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any


# Required keys on every harness/skills/*.json document. Names match the
# convention in harness/skills/example_skill.json plus the original #3 layer.
_REQUIRED_SKILL_KEYS: tuple[str, ...] = (
    "name",
    "version",
    "description",
    "input_schema",
    "output_schema",
)

# Required keys on every harness/manifest.json document (issue #277).
# Mirrors the contract in tests/harness/test_manifest.py:REQUIRED_KEYS so
# the Critic gate and the dev test suite agree on the manifest schema.
_REQUIRED_MANIFEST_KEYS: tuple[str, ...] = (
    "version",
    "model_target",
    "hooks",
    "skills",
)


def _check_dir_exists(harness_dir: Path) -> list[str]:
    if not harness_dir.is_dir():
        return [f"harness directory does not exist: {harness_dir}"]
    return []


def _check_skills(harness_dir: Path) -> list[str]:
    skills_dir = harness_dir / "skills"
    if not skills_dir.is_dir():
        return [f"harness/skills directory does not exist: {skills_dir}"]
    failures: list[str] = []
    for path in sorted(skills_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})")
            continue
        if not isinstance(doc, dict):
            failures.append(f"{path}: top-level must be a JSON object, got {type(doc).__name__}")
            continue
        missing = [k for k in _REQUIRED_SKILL_KEYS if k not in doc]
        if missing:
            failures.append(f"{path}: missing required keys {missing!r}")
        # Issue #278: the runner discovers skills by globbing skills/*.json
        # (filename) but exposes doc['name'] as the tool name (runner.py
        # _load_tool_definitions). A filename/name divergence means the model
        # sees one tool name while debugging references point at another. Only
        # check when 'name' is present -- a missing 'name' is already reported
        # above and comparing an absent field against the stem would be noise.
        if "name" not in missing:
            name = doc.get("name")
            stem = path.stem
            if name != stem:
                failures.append(
                    f"{path}: skill name {name!r} must match filename stem "
                    f"{stem!r} (issue #278; runner.py exposes doc['name'] as "
                    f"the tool name while debugging references point at the file)"
                )
    return failures


def _check_system_prompt(harness_dir: Path) -> list[str]:
    path = harness_dir / "system_prompt.txt"
    if not path.exists():
        return [f"{path}: missing"]
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"{path}: read failed ({exc.strerror or exc})"]
    if not text:
        return [f"{path}: empty"]
    return []


def _check_manifest(harness_dir: Path) -> list[str]:
    """Validate ``harness/manifest.json`` cross-refs against disk (issue #277).

    The manifest is the Evolver's declaration of what the harness ships.
    Without this check a ``ProposedEdit`` that adds or removes a skill or
    hook file can silently desync the manifest from disk -- the Critic
    gate never catches the drift. Validates:

    * the manifest exists and parses as a JSON object
    * required keys (``version``, ``model_target``, ``hooks``, ``skills``)
    * every ``skills`` entry resolves under ``harness/skills/``
    * every ``hooks`` entry resolves to ``harness/hooks/<entry>.py``
    * ``version`` matches ``harness/VERSION`` when that file exists

    Returns early when structural keys are missing -- cross-ref checks are
    meaningless without a well-formed ``hooks``/``skills`` shape.
    """
    manifest = harness_dir / "manifest.json"
    if not manifest.exists():
        return [f"{manifest}: missing (manifest.json is required by the Critic gate)"]
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{manifest}: invalid JSON ({exc.msg} at line {exc.lineno})"]
    if not isinstance(doc, dict):
        return [f"{manifest}: top-level must be a JSON object, got {type(doc).__name__}"]

    failures: list[str] = []
    missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in doc]
    if missing:
        failures.append(f"{manifest}: missing required keys {missing!r}")
        return failures

    skills_dir = harness_dir / "skills"
    skills = doc["skills"]
    if isinstance(skills, list):
        for entry in skills:
            skill_path = skills_dir / str(entry)
            if not skill_path.exists():
                failures.append(
                    f"{manifest}: skill entry {entry!r} not found on disk at {skill_path}"
                )
    else:
        failures.append(f"{manifest}: 'skills' must be a list, got {type(skills).__name__}")

    hooks_dir = harness_dir / "hooks"
    hooks = doc["hooks"]
    if isinstance(hooks, list):
        for entry in hooks:
            hook_path = hooks_dir / f"{str(entry)}.py"
            if not hook_path.exists():
                failures.append(
                    f"{manifest}: hook entry {entry!r} not found on disk at {hook_path}"
                )
    else:
        failures.append(f"{manifest}: 'hooks' must be a list, got {type(hooks).__name__}")

    version_file = harness_dir / "VERSION"
    if version_file.exists():
        try:
            disk_version = version_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            failures.append(f"{version_file}: read failed ({exc.strerror or exc})")
        else:
            manifest_version = str(doc["version"])
            if manifest_version != disk_version:
                failures.append(
                    f"{manifest}: version {manifest_version!r} disagrees with "
                    f"harness/VERSION ({disk_version!r})"
                )

    return failures


def _read_manifest_hooks(harness_dir: Path) -> list[str]:
    """Return the ``hooks`` array from ``harness/manifest.json`` (issue #206).

    SECURITY.md:50-52 promises that rate-limit defaults "live in
    ``harness/hooks/``," so the load-check success message should name
    every wired hook. Returns an empty list when the manifest is absent
    or malformed -- those are not load-check failures (the manifest has
    its own dedicated test suite in ``tests/harness/test_manifest.py``).
    """
    manifest = harness_dir / "manifest.json"
    if not manifest.exists():
        return []
    try:
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    hooks = doc.get("hooks", [])
    if not isinstance(hooks, list):
        return []
    return [str(h) for h in hooks]


class HookManifestValidator:
    """Validate that hook execution order matches manifest declaration (issue #567).

    Parses ``harness/manifest.json``, reads the declared ``hooks`` order, imports
    each hook module, and inspects its ``_phase`` class attribute.  Fails with a
    descriptive error if the actual execution order diverges from the manifest
    declaration.  Skips with a warning if a hook does not declare a ``_phase``
    (per ADR-0019).
    """

    _HOOK_CLASSES: dict[str, type] = {
        "injection_firewall": None,  # filled in lazily
        "context_pruning": None,
        "rate_limit": None,
    }

    def __init__(self, harness_dir: Path) -> None:
        self._harness_dir = harness_dir
        self._parent: Path = harness_dir.resolve().parent
        self._sys_path_inserted = False

    def _ensure_in_sys_path(self) -> None:
        parent_str = str(self._parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
            self._sys_path_inserted = True
        else:
            self._sys_path_inserted = False

    def _remove_sys_path(self) -> None:
        if self._sys_path_inserted:
            parent_str = str(self._parent)
            if parent_str in sys.path:
                sys.path.remove(parent_str)

    def _load_hook_class(self, name: str) -> type | None:
        if name == "base":
            return None
        module_name = f"harness.hooks.{name}"
        try:
            importlib.invalidate_caches()
            module = importlib.import_module(module_name)
        except ImportError:
            return None
        hook_classes = [
            getattr(module, attr)
            for attr in dir(module)
            if not attr.startswith("_")
            and isinstance(getattr(module, attr, None), type)
            and hasattr(getattr(module, attr), "_phase")
        ]
        if not hook_classes:
            return None
        if len(hook_classes) > 1:
            warnings.warn(
                f"hook module {module_name!r} defines multiple classes with "
                f"_phase: {hook_classes!r}; using first: {hook_classes[0].__name__}",
                UserWarning,
            )
        return hook_classes[0]

    def validate(self) -> tuple[bool, list[str]]:
        """Validate hook order against manifest.

        Returns
        -------
        (ok, errors_or_warnings)
            * (True, []) if order is valid
            * (True, [warnings]) if order is valid but some hooks were skipped
            * (False, [errors]) if order is invalid
        """
        manifest = self._harness_dir / "manifest.json"
        if not manifest.exists():
            return True, []

        try:
            doc = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True, []

        hooks_declared: list[str] = doc.get("hooks", [])
        if not isinstance(hooks_declared, list):
            return True, []

        self._ensure_in_sys_path()
        errors: list[str] = []
        warnings_list: list[str] = []
        try:
            for index, name in enumerate(hooks_declared):
                hook_class = self._load_hook_class(name)
                if hook_class is None:
                    if name != "base":
                        warnings_list.append(
                            f"hook {name!r}: no class with _phase found; "
                            f"skipping order validation (ADR-0019)"
                        )
                    continue
                actual_phase: int | Any = hook_class._phase
                if not isinstance(actual_phase, int):
                    warnings_list.append(
                        f"hook {name!r}: _phase is {type(actual_phase).__name__} "
                        f"(expected int); skipping order validation (ADR-0019)"
                    )
                    continue
                if actual_phase != index:
                    errors.append(
                        f"hook {name!r}: declared at position {index} in manifest "
                        f"but _phase={actual_phase}; execution order would be "
                        f"incorrect (ADR-0019 requires manifest order == execution order)"
                    )
        finally:
            self._remove_sys_path()

        if errors:
            return False, errors
        return True, warnings_list


def _check_hooks_importable(harness_dir: Path) -> list[str]:
    """Add the parent of ``harness_dir`` to ``sys.path`` and try to
    ``import harness.hooks``. Catches ImportError + post-import registry
    problems; surfaces the traceback fragments for diagnostics."""
    parent = harness_dir.resolve().parent
    parent_str = str(parent)
    inserted = False
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
        inserted = True
    try:
        try:
            importlib.invalidate_caches()
            module = importlib.import_module("harness.hooks")
        except Exception as exc:  # noqa: BLE001 — surface the actual failure
            return [f"import harness.hooks: FAILED ({type(exc).__name__}: {exc})"]
        # The harness self-registers the firewall on import (see
        # harness/hooks/__init__.py). Verify a registry exists and either
        # get_registry() or module-level access yields one.
        get_registry = getattr(module, "get_registry", None)
        registry = None
        if callable(get_registry):
            try:
                registry = get_registry()
            except Exception as exc:  # noqa: BLE001
                return [f"harness.hooks.get_registry(): FAILED ({type(exc).__name__}: {exc})"]
        if registry is None:
            return ["harness.hooks did not expose a registry (get_registry is None)"]
    finally:
        if inserted and sys.path and sys.path[0] == parent_str:
            sys.path.pop(0)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="load_check.py",
        description="Smoke-test the harness tree (issue #107; ADR-0004 gate).",
    )
    parser.add_argument(
        "--harness-dir",
        default="harness",
        type=Path,
        help="Path to the harness directory (default: ./harness)",
    )
    args = parser.parse_args(argv)
    harness_dir: Path = args.harness_dir

    failures: list[str] = []
    failures.extend(_check_dir_exists(harness_dir))
    if failures:
        # If the dir itself doesn't exist, every other check is meaningless.
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        return 2

    failures.extend(_check_skills(harness_dir))
    failures.extend(_check_system_prompt(harness_dir))
    failures.extend(_check_hooks_importable(harness_dir))
    failures.extend(_check_manifest(harness_dir))

    ok, validator_msgs = HookManifestValidator(harness_dir).validate()
    if not ok:
        failures.extend(validator_msgs)
    else:
        for msg in validator_msgs:
            print(f"WARN: {msg}", file=sys.stderr)

    if failures:
        print(f"harness load-check FAILED: {len(failures)} issue(s)", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    hooks_wired = _read_manifest_hooks(harness_dir)
    hooks_str = ", ".join(hooks_wired) if hooks_wired else "(none)"
    print(
        f"harness load-check OK: {harness_dir} "
        f"(skills OK, system_prompt.txt non-empty, registry instantiates, "
        f"manifest cross-refs OK, hook-order validated, hooks wired: {hooks_str})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
