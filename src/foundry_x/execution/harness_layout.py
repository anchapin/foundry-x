"""Pre-flight validation of the harness directory layout (issue #90).

The FoundryX execution runner inserts ``harness_dir`` into ``sys.path``
and then relies on the harness exposing ``system_prompt.txt``, ``hooks/``,
and ``skills/``. A wrong path or incomplete checkout used to surface deep
inside ``harness.hooks.__init__`` as an :class:`ImportError`, or inside
``run_task`` as a malformed-skill :class:`ValueError` -- never as a clear,
actionable CLI message. ``validate()`` makes that misconfiguration fail
fast at the entry point with the missing entries named verbatim, the
evidence-first failure mode ``docs/PHILOSOPHY.md`` §1 demands.

With ``strict=True`` (issue #1468), ``validate()`` also runs the same
content-level invariants the Critic gate relies on: skill-JSON parse,
manifest cross-refs, hook importability, and hook execution order. These
checks are imported from ``harness/scripts/load_check.py`` -- the single
source of truth -- so the runner and the Critic gate always agree.

Out of scope (tracked separately):

* Auto-creating a missing harness skeleton -- the operator owns the
  harness (``docs/SECURITY.md``, ADR-0004).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SYSTEM_PROMPT: str = "system_prompt.txt"
HOOKS_DIR: str = "hooks"
SKILLS_DIR: str = "skills"

REQUIRED_ENTRIES: tuple[str, ...] = (SYSTEM_PROMPT, HOOKS_DIR, SKILLS_DIR)


class HarnessValidationError(Exception):
    """Raised when ``harness_dir`` is missing one or more required entries.

    ``missing`` carries each entry name verbatim -- relative to
    ``harness_dir`` when it exists, or ``[str(harness_dir)]`` when the
    directory itself is not present. The list lets the caller format its
    own user-facing message (typically one line per missing entry on
    ``stderr``) without re-walking the directory and re-deriving the same
    conclusion.
    """

    def __init__(self, harness_dir: Path, missing: list[str]) -> None:
        self.harness_dir = Path(harness_dir)
        self.missing = list(missing)
        super().__init__(
            f"harness directory {self.harness_dir} is missing required "
            f"entries: {', '.join(self.missing) if self.missing else '<none>'}"
        )


class HarnessContentError(Exception):
    """Raised when ``harness_dir`` passes path checks but fails content checks.

    Carries ``failures`` — a list of human-readable failure messages
    produced by the ``load_check`` invariants (skill-JSON parse errors,
    manifest cross-ref drift, hook-import failures, hook-order mismatches).
    The caller formats each message on ``stderr`` so the operator sees
    every gap in one shot (issue #1468).
    """

    def __init__(self, harness_dir: Path, failures: list[str]) -> None:
        self.harness_dir = Path(harness_dir)
        self.failures = list(failures)
        joined = "; ".join(self.failures) if self.failures else "<none>"
        super().__init__(f"harness content validation failed for {self.harness_dir}: {joined}")


def _missing_entries(harness_dir: Path) -> list[str]:
    """Return the entries in :data:`REQUIRED_ENTRIES` that ``harness_dir`` lacks.

    Pure helper -- does not raise -- so :func:`validate` can report all
    gaps in one shot and the operator can fix them in one pass instead
    of cycling one error at a time.

    If ``harness_dir`` does not exist or is not a directory, the path
    itself is returned as the single missing entry; downstream access
    (``is_file`` / ``is_dir``) on a non-existent path would otherwise
    triple-report every required entry without ever saying the real
    problem is that the directory is gone.
    """
    if not harness_dir.is_dir():
        return [str(harness_dir)]
    missing: list[str] = []
    if not (harness_dir / SYSTEM_PROMPT).is_file():
        missing.append(SYSTEM_PROMPT)
    if not (harness_dir / HOOKS_DIR).is_dir():
        missing.append(HOOKS_DIR)
    if not (harness_dir / SKILLS_DIR).is_dir():
        missing.append(SKILLS_DIR)
    return missing


def _run_content_checks(harness_dir: Path) -> list[str]:
    """Run the content-level invariants from ``harness/scripts/load_check.py``.

    Imports the load_check module dynamically (it is a standalone script,
    not a package module) and calls its ``_check_*`` functions plus
    ``HookManifestValidator``. Returns a flat list of failure strings;
    an empty list means all content invariants passed (issue #1468).
    """
    load_check_path = harness_dir / "scripts" / "load_check.py"
    if not load_check_path.is_file():
        msg = (
            f"{load_check_path}: missing (required for strict validation; "
            "copy harness/scripts/load_check.py into the harness tree)"
        )
        return [msg]
    spec = importlib.util.spec_from_file_location("_load_check_runtime", load_check_path)
    if spec is None or spec.loader is None:
        return [f"{load_check_path}: cannot create import spec"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    failures: list[str] = []
    failures.extend(module._check_skills(harness_dir))
    failures.extend(module._check_system_prompt(harness_dir))
    failures.extend(module._check_hooks_importable(harness_dir))
    failures.extend(module._check_manifest(harness_dir))

    ok, msgs = module.HookManifestValidator(harness_dir).validate()
    if not ok:
        failures.extend(msgs)
    return failures


def validate(harness_dir: Path, strict: bool = False) -> None:
    """Validate that ``harness_dir`` exposes the harness layout the runner expects.

    Required entries (relative to ``harness_dir``):

    * ``system_prompt.txt`` -- file the agent loop reads as the system
      message (``runner.run_task``).
    * ``hooks/`` -- directory; ``harness/hooks/__init__.py`` self-
      registers the prompt-input firewall
      (``docs/SECURITY.md`` "Prompt-input firewall") on import.
    * ``skills/`` -- directory; ``runner._load_tool_definitions`` globs
      ``*.json`` here to assemble the OpenAI-compatible ``tools=``
      surface the model sees.

    When ``strict=True`` (issue #1468), also runs the content-level
    invariants from ``harness/scripts/load_check.py``: skill-JSON parse,
    manifest cross-refs, hook importability, and hook execution order.
    Content failures raise :class:`HarnessContentError` carrying every
    failure message in a single exception so the CLI can print the full
    list at once.

    Raises :class:`HarnessValidationError` carrying every missing entry
    in a single exception so the CLI can print the full list at once.
    Returns ``None`` on a valid layout; the function is intentionally
    side-effect-free so tests can call it without touching ``sys.path``
    or the trace store.
    """
    missing = _missing_entries(harness_dir)
    if missing:
        raise HarnessValidationError(harness_dir, missing)
    if strict:
        failures = _run_content_checks(harness_dir)
        if failures:
            raise HarnessContentError(harness_dir, failures)
