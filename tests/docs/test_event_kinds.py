"""Guard test: trace event kind vocabulary in CONTEXT.md is the source of truth.

Issue #194 closed the gap where the trace ``kind`` vocabulary was scattered
across :mod:`src/foundry_x/execution/runner`, :mod:`src/foundry_x/observability/regression_report`,
:mod:`harness/hooks/injection_firewall`, :mod:`harness/hooks/context_pruning`,
and the JSONL marker paths in :mod:`src/foundry_x/trace.logger`. A
Phase-3 contributor wiring a new producer used to grep the codebase to
discover the closed set. :mod:`docs.CONTEXT.md` is now the source of
truth: every currently-emitted kind is enumerated in the ``Event
kinds`` table, and :data:`foundry_x.evolution.digester.FAILURE_KINDS`
plus :data:`foundry_x.evolution.digester.FAILURE_PAYLOAD_KEYS` are
cross-referenced under the ``Failure-signalling subset`` subsection.

This test pins the contract:

1. The closed set of currently-emitted kinds is hardcoded in
   :data:`KNOWN_KINDS` below; the table in CONTEXT.md must enumerate
   every one of them and nothing else. Adding an emitted kind without
   updating both sides fails this test, which is the desired fail-closed
   behaviour for a vocabulary change (CONTEXT.md:8-9, ADR-0004).
2. The ``Failure-signalling subset`` subsection must name every value
   in :data:`foundry_x.evolution.digester.FAILURE_KINDS`,
   :data:`foundry_x.evolution.digester.FAILURE_PAYLOAD_KEYS`, and the
   :data:`foundry_x.evolution.digester.INJECTION_BLOCKED_KIND`
   constant, so the failure vocabulary is reachable from one place.
3. Every kind name in the table must be valid snake_case — the issue
   explicitly forbids renaming kinds in a documentation change.

Pure string matching and a hardcoded constant — no new dependency,
mirrors :mod:`tests.docs.test_context_glossary`. See issue #194 and
ADR-0007.
"""

from __future__ import annotations

import re
from pathlib import Path

from foundry_x.evolution.digester import (
    FAILURE_KINDS,
    FAILURE_PAYLOAD_KEYS,
    INJECTION_BLOCKED_KIND,
)

# Repo root: tests/docs/test_event_kinds.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_MD = REPO_ROOT / "docs" / "CONTEXT.md"

# Closed set of ``kind`` values currently emitted by FoundryX. Update in
# the same PR that introduces a new producer (CONTEXT.md:8-9) — this
# test fails until both sides of the contract are updated.
#
# Emitters (kept here as a comment so the source-of-truth is greppable):
#   - TraceLogger.session / _end_session: ``session_start``, ``session_end``
#     (JSONL marker dict in src/foundry_x/trace/logger.py)
#   - Runner.main: ``task_received``, ``task_completed``, ``task_failed``
#   - Runner.run_with_limits: ``task_aborted``
#   - Runner.run_task: ``user_prompt``, ``model_request``, ``model_response``,
#     ``model_error``, ``tool_call``, ``tool_result``, ``outcome``,
#     ``hook_registry_error`` (issue #260: get_registry() raised)
#   - InjectionFirewallHook: ``injection_blocked`` (via ``tracer`` callback)
#   - InjectionFirewallHook: ``firewall_exception`` (via ``tracer`` callback, issue #823)
#   - ContextPruningHook: ``context_pruned`` (via ``tracer`` callback)
#   - record_verdict: ``critic_verdict`` (constant ``VERDICT_KIND``)
KNOWN_KINDS: frozenset[str] = frozenset(
    {
        "context_pruned",
        "critic_verdict",
        "firewall_exception",
        "hook_registry_error",
        "injection_blocked",
        "model_error",
        "model_request",
        "model_response",
        "outcome",
        "session_end",
        "session_start",
        "task_aborted",
        "task_completed",
        "task_failed",
        "task_received",
        "tool_call",
        "tool_result",
        "user_prompt",
    }
)

# A kind cell in the markdown table looks like:
#   | **`session_start`** | ...
# We capture the bolded, backticked kind name.
_TABLE_KIND_CELL_RE = re.compile(r"^\|\s*\*\*`([a-z][a-z0-9_]*)`\*\*\s*\|")

# A bullet list entry that names a kind with backticks:
#   - **`tool_error`**, `task_failed`, ...
_BULLET_KIND_RE = re.compile(r"`([a-z][a-z0-9_]*)`")

# The section we want to scan starts at "## Event kinds" and ends at the
# next "## " heading.
_SECTION_HEADING_RE = re.compile(r"^##\s+", re.MULTILINE)


def _read_event_kinds_section() -> str:
    text = CONTEXT_MD.read_text(encoding="utf-8")
    match = re.search(r"^##\s+Event kinds\s*$", text, re.MULTILINE)
    assert match is not None, "docs/CONTEXT.md is missing the '## Event kinds' section"
    start = match.end()
    next_heading = _SECTION_HEADING_RE.search(text, pos=start)
    end = next_heading.start() if next_heading else len(text)
    return text[start:end]


def _extract_table_kinds(section: str) -> list[str]:
    kinds: list[str] = []
    for line in section.splitlines():
        m = _TABLE_KIND_CELL_RE.match(line)
        if m:
            kinds.append(m.group(1))
    return kinds


def _extract_failure_subset_kinds(section: str) -> set[str]:
    """Kinds named in the 'Failure-signalling subset' subsection.

    Restricted to that subsection so unrelated backticked literals
    elsewhere in the section (e.g. column-name mentions like
    ``error_type``) are not picked up.
    """
    subset_match = re.search(
        r"^###\s+Failure-signalling subset\s*(.+?)(?=^###\s+|\Z)",
        section,
        re.MULTILINE | re.DOTALL,
    )
    if subset_match is None:
        return set()
    body = subset_match.group(1)
    return set(_BULLET_KIND_RE.findall(body))


def test_event_kinds_section_exists():
    """CONTEXT.md must carry an '## Event kinds' section so the table can live."""
    assert CONTEXT_MD.is_file(), f"missing glossary: {CONTEXT_MD}"
    section = _read_event_kinds_section()
    assert section.strip(), "Event kinds section in CONTEXT.md is empty"


def test_table_enumerates_every_known_kind():
    """Every currently-emitted kind must have a row in the Event kinds table.

    The check is symmetric: the table must enumerate every value in
    :data:`KNOWN_KINDS`, and it must not introduce orphan rows that are
    not in :data:`KNOWN_KINDS`. Both directions are protected so the
    source of truth cannot drift in either direction.
    """
    section = _read_event_kinds_section()
    documented = set(_extract_table_kinds(section))

    missing = sorted(KNOWN_KINDS - documented)
    assert not missing, (
        "Trace event kinds are emitted in src/ or harness/ but not "
        f"documented in docs/CONTEXT.md 'Event kinds' table: {missing}. "
        "Add a row for each and keep KNOWN_KINDS in sync (issue #194)."
    )

    extra = sorted(documented - KNOWN_KINDS)
    assert not extra, (
        "CONTEXT.md 'Event kinds' table lists kinds that are not in "
        f"KNOWN_KINDS: {extra}. Either remove the row or add the kind "
        "to KNOWN_KINDS in tests/docs/test_event_kinds.py (issue #194)."
    )


def test_table_kind_names_are_snake_case():
    """Every kind cell must be a valid snake_case Python-identifier-like string.

    The issue explicitly forbids renaming kinds as part of this work;
    a non-snake_case entry would suggest an accidental rename via
    documentation drift.
    """
    section = _read_event_kinds_section()
    kinds = _extract_table_kinds(section)
    invalid = [k for k in kinds if not re.fullmatch(r"[a-z][a-z0-9_]*", k)]
    assert not invalid, (
        f"Invalid kind names in CONTEXT.md Event kinds table: {invalid}. "
        "Kinds must be snake_case and unchanged from the emitter (issue #194)."
    )


def test_failure_subset_cross_references_digester_constants():
    """The Failure-signalling subset subsection must name every value in the Digester's failure vocabulary.

    Closed set: ``FAILURE_KINDS`` ∪ {``INJECTION_BLOCKED_KIND``} for
    failure-signalling kinds, plus ``FAILURE_PAYLOAD_KEYS`` for
    payload-key signals. The digester exposes them as module constants
    so this test can pin the cross-reference without duplicating the
    lists.
    """
    section = _read_event_kinds_section()
    subset_kinds = _extract_failure_subset_kinds(section)

    expected_failure_kinds = set(FAILURE_KINDS) | {INJECTION_BLOCKED_KIND}
    missing_failure_kinds = sorted(expected_failure_kinds - subset_kinds)
    assert not missing_failure_kinds, (
        "CONTEXT.md 'Failure-signalling subset' must name every value in "
        "FAILURE_KINDS plus INJECTION_BLOCKED_KIND. Missing: "
        f"{missing_failure_kinds}."
    )

    expected_payload_keys = set(FAILURE_PAYLOAD_KEYS)
    missing_payload_keys = sorted(expected_payload_keys - subset_kinds)
    assert not missing_payload_keys, (
        "CONTEXT.md 'Failure-signalling subset' must name every key in "
        f"FAILURE_PAYLOAD_KEYS. Missing: {missing_payload_keys}."
    )
