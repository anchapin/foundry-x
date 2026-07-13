"""Pre-flight validation of the harness directory layout (issue #90).

The FoundryX execution runner inserts ``harness_dir`` into ``sys.path``
and then relies on the harness exposing ``system_prompt.txt``, ``hooks/``,
and ``skills/``. A wrong path or incomplete checkout used to surface deep
inside ``harness.hooks.__init__`` as an :class:`ImportError`, or inside
``run_task`` as a malformed-skill :class:`ValueError` -- never as a clear,
actionable CLI message. ``validate()`` makes that misconfiguration fail
fast at the entry point with the missing entries named verbatim, the
evidence-first failure mode ``docs/PHILOSOPHY.md`` §1 demands.

Validation is path-only: parsing JSON / booting hooks / running the
prompt-input firewall is the harness's job at startup. This module
exists so the operator sees the gap, not a downstream traceback.

Out of scope (tracked separately):

* Auto-creating a missing harness skeleton -- the operator owns the
  harness (``docs/SECURITY.md``, ADR-0004).
* Validating the *contents* of ``system_prompt.txt`` or
  ``skills/*.json`` -- those are Critic concerns.
"""

from __future__ import annotations

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


def validate(harness_dir: Path) -> None:
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

    Raises :class:`HarnessValidationError` carrying every missing entry
    in a single exception so the CLI can print the full list at once.
    Returns ``None`` on a valid layout; the function is intentionally
    side-effect-free so tests can call it without touching ``sys.path``
    or the trace store.
    """
    missing = _missing_entries(harness_dir)
    if missing:
        raise HarnessValidationError(harness_dir, missing)
