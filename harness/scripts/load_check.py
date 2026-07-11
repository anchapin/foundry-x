#!/usr/bin/env python3
"""Smoke-test that the harness tree is loadable. Used by the Critic (ADR-0004)
to gate ``ProposedEdit`` proposals before they are marked active.

Validates four invariants:

* every ``harness/skills/*.json`` parses and carries the five required keys
  (``name``, ``version``, ``description``, ``input_schema``, ``output_schema``)
* ``import harness.hooks`` succeeds and the registry instantiates
* ``harness/system_prompt.txt`` exists and is non-empty
* the harness directory itself exists

Stdlib-only by design. The script adds the parent of ``--harness-dir`` to
``sys.path`` so that ``import harness.hooks`` resolves the same way the
rest of the foundry does under pytest's ``pythonpath = ["."]``.

Exit codes:
    0  — all invariants pass
    1  — at least one invariant fails (per-failure message on stderr)
    2  — usage error (e.g. missing --harness-dir or non-existent dir)

Issue #107; advances SECURITY.md threat #1 (harness degradation).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


# Required keys on every harness/skills/*.json document. Names match the
# convention in harness/skills/example_skill.json plus the original #3 layer.
_REQUIRED_SKILL_KEYS: tuple[str, ...] = (
    "name",
    "version",
    "description",
    "input_schema",
    "output_schema",
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
        f"hooks wired: {hooks_str})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
