"""Deterministic trace-walking tests for the Digester (issue #15).

Acceptance per issue #15 / ADR-0007: feed synthetic ``TraceEvent`` sequences
(one clean run, one tool-error run, one traceback run) and assert the correct
``proposed_class`` and first-failed-step identification. The failure report is
derived from trace content, not speculation.
"""

from __future__ import annotations

import pytest

from foundry_x.evolution.digester import (
    FAILURE_KINDS,
    FAILURE_PAYLOAD_KEYS,
    INJECTION_ATTEMPT_CLASS,
    INJECTION_BLOCKED_KIND,
    Digester,
    FailureReport,
)
from foundry_x.trace.logger import TraceEvent

_SESSION = "sess-1"


def _ev(
    kind: str,
    payload: dict,
    *,
    event_id: str = "e0",
    seq: int = 0,
) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        session_id=_SESSION,
        timestamp=f"2026-07-10T00:00:{seq:02d}+00:00",
        kind=kind,
        payload=payload,
    )


_CLEAN_EVENTS = [
    _ev("user_prompt", {"text": "add a foo"}, event_id="e1", seq=1),
    _ev("tool_call", {"tool": "edit", "path": "a.py"}, event_id="e2", seq=2),
    _ev("tool_result", {"ok": True, "output": "done"}, event_id="e3", seq=3),
]


def test_clean_run_yields_clean_report():
    report = Digester().digest(_SESSION, _CLEAN_EVENTS)
    assert report.proposed_class == "clean"
    assert report.failed_steps == []
    assert report.suspected_causes == []
    assert "3 trace event(s)" in report.summary


def test_empty_events_yields_clean_report():
    report = Digester().digest(_SESSION, [])
    assert report.proposed_class == "clean"
    assert report.failed_steps == []
    assert "0 trace event(s)" in report.summary


def test_tool_error_kind_classified_as_tool_error():
    events = [
        *_CLEAN_EVENTS,
        _ev(
            "tool_error",
            {"error": "command failed with exit code 1"},
            event_id="e-fail",
            seq=4,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == "tool-error"
    assert len(report.failed_steps) == 1
    step = report.failed_steps[0]
    assert step["event_id"] == "e-fail"
    assert step["kind"] == "tool_error"
    assert step["index"] == 3
    assert step["signal"] == "kind:tool_error"
    assert report.suspected_causes  # non-empty
    assert report.summary.startswith("tool-error failure")


def test_traceback_payload_classified_as_tool_error():
    events = [
        *_CLEAN_EVENTS,
        _ev(
            "tool_result",
            {"traceback": "Traceback (most recent call last):\nValueError: bad"},
            event_id="e-tb",
            seq=4,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == "tool-error"
    step = report.failed_steps[0]
    # Benign kind, but the payload key tripped the detector.
    assert step["kind"] == "tool_result"
    assert step["signal"] == "payload_key:traceback"
    assert any("payload key present: traceback" in c for c in report.suspected_causes)


@pytest.mark.parametrize(
    ("payload", "kind", "expected"),
    [
        ({"error": "no such tool: frobnicate"}, "tool_error", "wrong-tool"),
        ({"error": "prompt is ambiguous: missing context"}, "task_failed", "bad-prompt"),
        ({"error": "FileNotFoundError: no such file 'foo.txt'"}, "tool_error", "state-leak"),
    ],
    ids=["wrong-tool", "bad-prompt", "state-leak"],
)
def test_keyword_classification(payload, kind, expected):
    events = [
        *_CLEAN_EVENTS,
        _ev(kind, payload, event_id="e-x", seq=4),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == expected


def test_first_failed_step_identified_among_multiple_failures():
    events = [
        *_CLEAN_EVENTS,
        _ev("tool_error", {"error": "first failure"}, event_id="e-first", seq=4),
        _ev("tool_error", {"error": "second failure"}, event_id="e-second", seq=5),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.failed_steps[0]["event_id"] == "e-first"
    assert "second failure" not in report.summary


def test_digest_is_order_independent_by_timestamp():
    ordered = [
        *_CLEAN_EVENTS,
        _ev("tool_error", {"error": "boom"}, event_id="e-fail", seq=4),
    ]
    shuffled = list(reversed(ordered))
    r1 = Digester().digest(_SESSION, ordered)
    r2 = Digester().digest(_SESSION, shuffled)
    assert r1.proposed_class == r2.proposed_class
    assert r1.failed_steps[0]["event_id"] == r2.failed_steps[0]["event_id"] == "e-fail"
    assert r1.failed_steps[0]["index"] == 3  # 3 clean events precede it


def test_failure_report_round_trips_through_pydantic():
    events = [
        *_CLEAN_EVENTS,
        _ev("tool_error", {"error": "exit code 2"}, event_id="e-fail", seq=4),
    ]
    report = Digester().digest(_SESSION, events)
    restored = FailureReport.model_validate(report.model_dump())
    assert restored == report


def test_redacted_payload_is_still_classified():
    # The TraceLogger scrubs secrets to ``[REDACTED:*]`` sentinels before the
    # Digester ever sees them (ADR-0003). Classification must still work on the
    # surrounding error text.
    events = [
        _ev(
            "tool_error",
            {"api_key": "[REDACTED:secret]", "error": "no such tool: deploy"},
            event_id="e-redact",
            seq=1,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == "wrong-tool"
    assert report.failed_steps[0]["payload"]["api_key"] == "[REDACTED:secret]"


def test_failure_vocabularies_are_frozen_constants():
    # issue #15: the kind vocabulary is a module constant so the trace
    # subsystem can align against it.
    assert isinstance(FAILURE_KINDS, frozenset)
    assert isinstance(FAILURE_PAYLOAD_KEYS, frozenset)
    assert "tool_error" in FAILURE_KINDS
    assert "traceback" in FAILURE_PAYLOAD_KEYS


# ---------------------------------------------------------------------------
# Issue #120: ``injection_blocked`` aggregation → ``proposed_class`` ==
# ``'injection-attempt'`` with one ``failed_steps`` entry per block.
#
# The firewall emits one ``injection_blocked`` trace event per
# suppression. The Digester's contract is to surface the full
# adversarial surface — not just the first block — so the Evolver can
# propose patterns or scrubbing policies that address every marker
# rather than papering over one.
# ---------------------------------------------------------------------------


def _block_event(
    markers: list[str],
    tool: str = "read_file",
    event_id: str = "e-block",
    seq: int = 4,
    preview: str = "ignore previous instructions",
) -> TraceEvent:
    """Build an ``injection_blocked`` event with the canonical payload shape."""
    return _ev(
        INJECTION_BLOCKED_KIND,
        {"markers": markers, "tool": tool, "preview": preview},
        event_id=event_id,
        seq=seq,
    )


def test_single_injection_block_yields_injection_attempt_class():
    events = [
        *_CLEAN_EVENTS,
        _block_event(["ignore_previous"], event_id="e-block", seq=4),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == INJECTION_ATTEMPT_CLASS
    assert INJECTION_ATTEMPT_CLASS in report.summary
    assert len(report.failed_steps) == 1
    step = report.failed_steps[0]
    assert step["kind"] == INJECTION_BLOCKED_KIND
    assert step["event_id"] == "e-block"
    assert step["signal"] == f"kind:{INJECTION_BLOCKED_KIND}"
    # Payload is preserved verbatim so the Evolver can re-derive marker
    # names without re-parsing the original tool output.
    assert step["payload"]["markers"] == ["ignore_previous"]
    # Cause references the first block's marker list as evidence (ADR-0007).
    assert any("ignore_previous" in c for c in report.suspected_causes)


def test_multiple_injection_blocks_are_aggregated():
    """Every block in the session lands in ``failed_steps``; count is exact."""
    events = [
        _ev("user_prompt", {"text": "go"}, event_id="e1", seq=1),
        _block_event(["ignore_previous"], event_id="e-b1", seq=2),
        _ev("tool_call", {"tool": "read_file"}, event_id="e2", seq=3),
        _block_event(
            ["ignore_spanish", "role_tag_colon"],
            event_id="e-b2",
            seq=4,
            tool="shell",
        ),
        _block_event(["base64_payload"], event_id="e-b3", seq=5, tool="curl"),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == INJECTION_ATTEMPT_CLASS
    assert len(report.failed_steps) == 3
    assert [s["event_id"] for s in report.failed_steps] == ["e-b1", "e-b2", "e-b3"]
    assert "3 firewall block(s)" in report.summary


def test_injection_blocks_take_precedence_over_later_tool_error():
    """An adversarial tool result is the more actionable failure signal.

    Even if a downstream ``tool_error`` event arrives after one or more
    ``injection_blocked`` events, the Digester must surface the
    injection-attempt class so the Evolver does not propose a fix for
    the wrong problem.
    """
    events = [
        _ev("user_prompt", {"text": "go"}, event_id="e1", seq=1),
        _block_event(["ignore_previous"], event_id="e-block", seq=2),
        _ev(
            "tool_error",
            {"error": "exit code 1"},
            event_id="e-tool-err",
            seq=3,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == INJECTION_ATTEMPT_CLASS
    # ``tool_error`` is still observed (later in the trace) but the
    # reported failure class is the injection attempt.
    assert report.failed_steps[0]["event_id"] == "e-block"


def test_no_injection_blocks_falls_through_to_existing_classifier():
    """Without injection events, the generic first-failure walk still wins."""
    events = [
        *_CLEAN_EVENTS,
        _ev(
            "tool_error",
            {"error": "no such tool: frobnicate"},
            event_id="e-fail",
            seq=4,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == "wrong-tool"
    assert report.failed_steps[0]["kind"] == "tool_error"


def test_injection_block_without_markers_payload_still_classified():
    """A malformed payload (no ``markers`` key) still produces a useful report."""
    events = [
        _ev(
            INJECTION_BLOCKED_KIND,
            {"tool": "read_file", "preview": "adversarial span"},
            event_id="e-malformed",
            seq=2,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == INJECTION_ATTEMPT_CLASS
    assert len(report.failed_steps) == 1
    # Cause still references something trace-derived so the report stays
    # grounded (ADR-0007).
    assert report.suspected_causes


def test_injection_block_vocabulary_is_exported():
    """The new event kind + class are pinned as module-level constants."""
    assert INJECTION_BLOCKED_KIND == "injection_blocked"
    assert INJECTION_ATTEMPT_CLASS == "injection-attempt"
    # ``injection_blocked`` is intentionally NOT in ``FAILURE_KINDS``: it
    # is handled by a dedicated aggregation pass, not the generic
    # first-failure walk.
    assert INJECTION_BLOCKED_KIND not in FAILURE_KINDS


# --- Coverage added by issue #95 --------------------------------------------
# Per-mode keyword exhaustiveness, specificity/precedence, extended
# first-failed-step scenarios, signal/cause contract, and a lock-in of the
# no-failure behaviour. The taxonomy (``bad-prompt`` / ``wrong-tool`` /
# ``tool-error`` / ``state-leak``) is fixed by issue #15 and ADR-0007; new
# keywords land in an ADR, not a test. Each parametrised case below mirrors a
# tuple entry in ``digester._CLASS_KEYWORDS`` so a keyword accidentally
# removed in a future refactor would surface as a test failure here.


_WRONG_TOOL_KEYWORDS: tuple[str, ...] = (
    "no such tool",
    "tool not found",
    "unknown tool",
    "invalid tool",
    "no tool named",
    "no command named",
    "unknown function",
    "is not a valid tool",
)

_BAD_PROMPT_KEYWORDS: tuple[str, ...] = (
    "missing context",
    "under-specified",
    "underspecified",
    "malformed prompt",
    "cannot parse prompt",
    "contradictory",
    "ambiguous",
    "vague",
    "unparseable",
)

_STATE_LEAK_KEYWORDS: tuple[str, ...] = (
    "no such file",
    "file not found",
    "already exists",
    "unexpected state",
    "race condition",
    "dirty tree",
    "is not empty",
    "leftover",
    "stale",
    "leak",
)

_TOOL_ERROR_KEYWORDS: tuple[str, ...] = (
    "traceback",
    "exception",
    "timed out",
    "timeout",
    "exit code",
    "segmentation fault",
    "segfault",
    "broken pipe",
    "command not found",
    "error",
    "failed",
)


def _one_failing_event(
    keyword: str,
    *,
    kind: str = "tool_error",
    event_id: str = "e-x",
    seq: int = 4,
) -> list[TraceEvent]:
    """A minimal single-failure trace carrying ``keyword`` in the payload.

    The ``error`` key both trips the structural payload-key detector
    (``FAILURE_PAYLOAD_KEYS``) and embeds the keyword in the flattened text
    blob used by ``_classify``. ``kind="tool_error"`` also satisfies the
    structural kind detector, but the keyword rules win over the catch-all.
    """
    return [_ev(kind, {"error": f"synthetic failure: {keyword}"}, event_id=event_id, seq=seq)]


@pytest.mark.parametrize("keyword", _WRONG_TOOL_KEYWORDS)
def test_wrong_tool_keyword_classifies_as_wrong_tool(keyword: str) -> None:
    report = Digester().digest(_SESSION, _one_failing_event(keyword))
    assert report.proposed_class == "wrong-tool"
    assert any(f"matched: {keyword}" in c for c in report.suspected_causes)


@pytest.mark.parametrize("keyword", _BAD_PROMPT_KEYWORDS)
def test_bad_prompt_keyword_classifies_as_bad_prompt(keyword: str) -> None:
    report = Digester().digest(_SESSION, _one_failing_event(keyword))
    assert report.proposed_class == "bad-prompt"
    assert any(f"matched: {keyword}" in c for c in report.suspected_causes)


@pytest.mark.parametrize("keyword", _STATE_LEAK_KEYWORDS)
def test_state_leak_keyword_classifies_as_state_leak(keyword: str) -> None:
    report = Digester().digest(_SESSION, _one_failing_event(keyword))
    assert report.proposed_class == "state-leak"
    assert any(f"matched: {keyword}" in c for c in report.suspected_causes)


@pytest.mark.parametrize("keyword", _TOOL_ERROR_KEYWORDS)
def test_tool_error_keyword_classifies_as_tool_error(keyword: str) -> None:
    report = Digester().digest(_SESSION, _one_failing_event(keyword))
    assert report.proposed_class == "tool-error"
    # The tool-error bucket has multiple keywords sharing one class. The
    # priority order in ``_CLASS_KEYWORDS`` means a payload carrying several
    # tool-error tokens (e.g. the literal word ``error`` from the payload
    # key) may resolve to whichever keyword scans first; we therefore only
    # assert the template fired, not the specific matched keyword.
    assert any("Tool execution raised an error" in c for c in report.suspected_causes)


# --- Specificity / precedence ----------------------------------------------
# digester.py:65-71: each mode has a priority, and the most-specific keyword
# wins over the ``tool-error`` catch-all. If a payload accidentally contains
# a generic ``tool-error`` keyword (``"error"``, ``"failed"``, ``"traceback"``)
# alongside a more specific mode keyword, the specific mode must win.


@pytest.mark.parametrize(
    ("keyword", "expected_class"),
    [
        ("no such tool: frobnicate", "wrong-tool"),
        ("invalid tool: rm_rf", "wrong-tool"),
        ("prompt is ambiguous and missing context", "bad-prompt"),
        ("file not found: missing.py", "state-leak"),
        ("already exists: /tmp/x", "state-leak"),
    ],
    ids=[
        "wrong-tool-beats-tool-error",
        "invalid-tool-beats-tool-error",
        "bad-prompt-beats-tool-error",
        "state-leak-beats-tool-error",
        "state-leak-exists-beats-tool-error",
    ],
)
def test_specific_keyword_beats_tool_error_catchall(keyword: str, expected_class: str) -> None:
    """A specific mode keyword in the same payload must outrank the generic ``tool-error`` keyword (e.g. ``error``)."""
    # Embed both: a specific phrase AND the bare word ``error`` / ``failed``.
    payload = {"error": f"{keyword} (also saw error and failed downstream)"}
    events = [_ev("tool_error", payload, event_id="e-spec", seq=4)]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == expected_class


def test_unknown_function_keyword_classifies_as_wrong_tool() -> None:
    """Regression: ``unknown function`` is in the wrong-tool vocabulary (digester.py:87)."""
    report = Digester().digest(
        _SESSION,
        _one_failing_event("unknown function foo()", event_id="e-ufn", seq=4),
    )
    assert report.proposed_class == "wrong-tool"


# --- Structural-signal classification --------------------------------------
# The detector fires on either a known ``kind`` or a known payload key. The
# catch-all ``tool-error`` keyword (``"error"``) is present in nearly every
# failing payload, so the keyword rule still drives the class — but the
# ``signal`` recorded in the failed step is what tripped the detector.


@pytest.mark.parametrize(
    "kind", ["tool_error", "task_failed", "run_failed", "agent_error", "error"]
)
def test_any_failure_kind_signals_via_kind_field(kind: str) -> None:
    """Every value in ``FAILURE_KINDS`` must trip ``signal='kind:<value>'``."""
    events = [_ev(kind, {"unrelated": "context"}, event_id=f"e-{kind}", seq=4)]
    report = Digester().digest(_SESSION, events)
    assert report.proposed_class == "tool-error"
    assert report.failed_steps[0]["signal"] == f"kind:{kind}"


@pytest.mark.parametrize("key", ["error", "traceback", "exception"])
def test_any_failure_payload_key_signals_via_payload_key_field(key: str) -> None:
    """A benign ``kind`` plus any key in ``FAILURE_PAYLOAD_KEYS`` trips the detector via ``payload_key:<key>``."""
    events = [_ev("tool_result", {key: "boom"}, event_id=f"e-{key}", seq=4)]
    report = Digester().digest(_SESSION, events)
    assert report.failed_steps[0]["signal"] == f"payload_key:{key}"
    # The payload-key signal must be reflected in the causes (evidence trail).
    assert any(f"payload key present: {key}" in c for c in report.suspected_causes)


def test_payload_key_signal_is_added_to_causes_when_present() -> None:
    """``_classify`` always appends the payload-key signal line, regardless of which mode matched."""
    events = [
        _ev(
            "tool_result",
            {"traceback": "Traceback (most recent call last):\nValueError"},
            event_id="e-tb",
            seq=4,
        )
    ]
    report = Digester().digest(_SESSION, events)
    # tool-error matched (the literal token ``traceback`` is in _TOOL_ERROR_KEYWORDS);
    # the payload_key signal is appended for traceability.
    assert report.proposed_class == "tool-error"
    assert any("matched: traceback" in c for c in report.suspected_causes)
    assert any("payload key present: traceback" in c for c in report.suspected_causes)


# --- First-failed-step identification --------------------------------------
# Existing ``test_first_failed_step_identified_among_multiple_failures``
# covers the two-tail case. The matrix below fixes the position of the first
# failure in the trace and asserts the reported ``index`` matches the
# timestamp-sorted position.


@pytest.mark.parametrize(
    ("fail_at", "expected_index"),
    [
        (0, 0),  # failure is the very first event
        (1, 1),  # failure is the second event (after one clean preamble)
        (2, 2),  # failure follows two clean events
        (4, 4),  # failure is the last event of a 5-event trace
    ],
    ids=["first-event", "after-one-clean", "after-two-clean", "last-event"],
)
def test_first_failed_step_index_reflects_position(fail_at: int, expected_index: int) -> None:
    """The reported ``index`` is the failure's position in the timestamp-sorted trace (issue #15: "first failed step index")."""
    total = fail_at + 1
    events = [
        _ev(
            "tool_call" if i < fail_at else "tool_error",
            {"ok": True} if i < fail_at else {"error": "boom"},
            event_id=f"e-{i}",
            seq=i,
        )
        for i in range(total)
    ]
    report = Digester().digest(_SESSION, events)
    assert len(report.failed_steps) == 1
    assert report.failed_steps[0]["index"] == expected_index
    assert report.failed_steps[0]["event_id"] == f"e-{fail_at}"


def test_first_failed_step_index_survives_timestamp_reversal() -> None:
    """Reversing the input order must not change the reported first-failure index (digest sorts by timestamp)."""
    forward = [
        _ev("user_prompt", {"text": "hi"}, event_id="e0", seq=0),
        _ev("tool_call", {"tool": "edit"}, event_id="e1", seq=1),
        _ev("tool_error", {"error": "first"}, event_id="e-fail", seq=2),
        _ev("tool_error", {"error": "second"}, event_id="e-after", seq=3),
    ]
    reversed_ = list(reversed(forward))
    r1 = Digester().digest(_SESSION, forward)
    r2 = Digester().digest(_SESSION, reversed_)
    assert r1.failed_steps[0]["index"] == 2
    assert r2.failed_steps[0]["index"] == 2
    assert r1.failed_steps[0]["event_id"] == r2.failed_steps[0]["event_id"] == "e-fail"


def test_failed_step_carries_full_event_metadata() -> None:
    """``failed_steps[0]`` must round-trip every field the digester extracted (issue #15 + ADR-0006)."""
    events = [
        _ev(
            "tool_error",
            {"error": "exit code 1", "stderr": "boom"},
            event_id="e-meta",
            seq=4,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    step = report.failed_steps[0]
    assert set(step) == {"index", "event_id", "kind", "timestamp", "signal", "payload"}
    assert step["event_id"] == "e-meta"
    assert step["kind"] == "tool_error"
    assert step["timestamp"] == "2026-07-10T00:00:04+00:00"
    assert step["payload"] == {"error": "exit code 1", "stderr": "boom"}


def test_summary_mentions_class_and_detail() -> None:
    """The report's summary must name the proposed class and surface a one-line excerpt of the failure (digester._summarise)."""
    # ``_detail`` takes the first non-empty line of a payload-key value
    # (digester.py:228-236); the traceback key triggers that path.
    events = [
        _ev(
            "tool_result",
            {"traceback": "ValueError: nope\nTraceback continues here"},
            event_id="e-sum",
            seq=4,
        ),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.summary.startswith("tool-error failure")
    assert "ValueError: nope" in report.summary
    # Subsequent lines are not echoed into the summary.
    assert "Traceback continues here" not in report.summary


def test_summary_falls_back_to_flattened_text_when_no_known_payload_key() -> None:
    """A failing ``kind`` with no matching payload key produces a summary from the flattened blob."""
    events = [
        _ev("agent_error", {"context": "the agent looped forever"}, event_id="e-flat", seq=4),
    ]
    report = Digester().digest(_SESSION, events)
    assert report.summary.startswith("tool-error failure")
    assert "agent_error" in report.summary
    # No traceback/error/exception key → no detail line, but the head is preserved.
    assert "kind=agent_error" in report.summary


# --- No-failure behaviour: lock-in for issue #95 criterion #4 --------------
# NOTE: issue #95 acceptance criterion #4 says
#     "a trace with no failure produces proposed_class 'unknown' and empty
#      failed_steps"
# but ``Digester.digest()`` explicitly returns ``proposed_class="clean"``
# (digester.py:289). The model default *is* ``"unknown"`` (digester.py:35),
# reached only when ``FailureReport`` is constructed directly without going
# through ``digest()``. Per the issue itself ("Out of scope: Changing the
# classification taxonomy (needs-adr)") and AGENTS.md ("NEVER widen scope")
# this test locks in the *current* behaviour and surfaces the discrepancy in
# the PR description so the taxonomy question is resolved in a follow-up ADR.


def test_no_failure_proposes_clean_class_with_empty_failed_steps() -> None:
    """No-failure traces produce ``proposed_class='clean'`` and empty ``failed_steps`` (digester.py:284-289)."""
    report = Digester().digest(_SESSION, _CLEAN_EVENTS)
    assert report.proposed_class == "clean"
    assert report.failed_steps == []
    assert report.suspected_causes == []


def test_empty_event_list_proposes_clean_class() -> None:
    """An empty trace produces the same clean report as a trace of all-clean events (digester.py:284-289)."""
    report = Digester().digest(_SESSION, [])
    assert report.proposed_class == "clean"
    assert report.failed_steps == []
    # Summary states the event count for traceability.
    assert "0 trace event(s)" in report.summary


def test_model_default_unknown_is_only_reached_when_constructed_directly() -> None:
    """``FailureReport()`` default is ``'unknown'``; ``Digester.digest()`` overrides with ``'clean'`` for no-failure input."""
    # Direct construction → model default applies (used by the render layer
    # for edge cases that never go through the digester, see
    # ``test_render_empty_fields_uses_defaults`` in test_observability.py).
    direct = FailureReport(session_id=_SESSION, summary="manual report")
    assert direct.proposed_class == "unknown"
    # Via digester → clean
    via_digester = Digester().digest(_SESSION, _CLEAN_EVENTS)
    assert via_digester.proposed_class == "clean"
