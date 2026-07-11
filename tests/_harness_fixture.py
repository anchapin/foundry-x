"""Shared harness-fixture helpers for Critic tests (issue #187).

The Critic gate now runs ``harness/scripts/load_check.py`` against the
sandbox copy as a precondition (issue #187; ADR-0004). Every fixture
harness exercised by ``Critic.evaluate`` must therefore contain the four
artefacts load_check validates:

* a ``skills/`` directory (empty is fine -- ``glob("*.json")`` yields none)
* a non-empty ``system_prompt.txt``
* an importable ``hooks`` package that exposes ``get_registry()``
* the ``scripts/load_check.py`` script itself

``install_load_check_prerequisites`` adds the load_check-only artefacts to
an existing harness fixture so each test module only has to specify what is
unique to its scenario (the pytest test file, extra files, or deliberately
broken invariants).
"""

from __future__ import annotations

import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_LOAD_CHECK = _REPO_ROOT / "harness" / "scripts" / "load_check.py"

# Minimal ``harness/hooks/__init__.py``: load_check imports ``harness.hooks``
# and calls ``get_registry()``; a non-None return satisfies the invariant
# (see harness/scripts/load_check.py::_check_hooks_importable).
_MINIMAL_HOOKS_INIT = (
    "class HookRegistry:\n"
    "    def __init__(self):\n"
    "        self._hooks = []\n"
    "\n"
    "    def register(self, hook):\n"
    "        self._hooks.append(hook)\n"
    "\n"
    "\n"
    "def get_registry():\n"
    "    return HookRegistry()\n"
)


def install_load_check_prerequisites(harness_dir: Path) -> None:
    """Add load_check-only artefacts to an existing harness fixture.

    Idempotent: safe to call after the test module has already created
    ``tests/`` and ``system_prompt.txt``. Creates an empty ``skills/``
    dir, a minimal ``hooks`` package exposing ``get_registry()``, and
    copies the real ``scripts/load_check.py`` into the fixture so the
    Critic can spawn it against the sandbox copy.
    """
    scripts_dir = harness_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REAL_LOAD_CHECK, scripts_dir / "load_check.py")
    (harness_dir / "skills").mkdir(exist_ok=True)
    hooks_dir = harness_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "__init__.py").write_text(_MINIMAL_HOOKS_INIT, encoding="utf-8")
