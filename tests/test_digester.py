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
