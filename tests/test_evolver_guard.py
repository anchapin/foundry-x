from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from foundry_x.evolution.evolver import (
    PROPOSED_EDIT_KIND,
    Evolver,
    EvolverGuardError,
    ProposedEdit,
)
from foundry_x.trace.logger import TraceLogger


def _make_diff(*lines: str) -> str:
    header = "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n"
    hunk = "@@ -0,0 +1 @@\n"
    return header + hunk + "".join(f"+{line}\n" for line in lines)


def _edit(diff: str) -> ProposedEdit:
    return ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="tighten tool guidance",
        unified_diff=diff,
    )


def test_record_proposal_emits_trace_event(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    edit = _edit(_make_diff("be precise"))
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_proposals(edit=edit)

    events = list(logger.iter_events(session_id, kind=PROPOSED_EDIT_KIND))
    assert len(events) == 1
    assert events[0].payload == edit.model_dump(mode="json")


def test_defaults_match_security_doc():
    e = Evolver()
    assert e.max_proposals_per_hour == 10
    assert e.max_diff_lines == 200


def test_invalid_limits_rejected():
    with pytest.raises(EvolverGuardError, match="max_proposals_per_hour"):
        Evolver(max_proposals_per_hour=0)
    with pytest.raises(EvolverGuardError, match="max_diff_lines"):
        Evolver(max_diff_lines=0)


def test_oversized_diff_rejected():
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    big_diff = _make_diff(*[f"line {i}" for i in range(10)])
    with pytest.raises(EvolverGuardError, match="diff too large"):
        e._validate_edit(_edit(big_diff))


def test_diff_at_exact_cap_passes():
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    e._validate_edit(_edit(_make_diff("l0", "l1")))


def test_rate_limit_triggers_after_cap():
    e = Evolver(max_proposals_per_hour=3, max_diff_lines=200)
    e._record_proposals(3)
    with pytest.raises(EvolverGuardError, match="rate limit exceeded"):
        e._check_rate_limit()


def test_rate_limit_below_cap_passes():
    e = Evolver(max_proposals_per_hour=3, max_diff_lines=200)
    e._record_proposals(2)
    e._check_rate_limit()


def test_old_proposals_purged_after_window():
    e = Evolver(max_proposals_per_hour=2, max_diff_lines=200)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    e._proposal_times.append(stale)
    e._proposal_times.append(stale)
    e._check_rate_limit()


def test_partial_window_keeps_recent_only():
    e = Evolver(max_proposals_per_hour=2, max_diff_lines=200)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    e._proposal_times.append(stale)
    e._record_proposals(2)
    with pytest.raises(EvolverGuardError, match="rate limit exceeded"):
        e._check_rate_limit()


def test_propose_calls_guard_before_body():
    e = Evolver(max_proposals_per_hour=1, max_diff_lines=200)
    e._record_proposals(1)
    with pytest.raises(EvolverGuardError, match="rate limit"):
        e.propose(Path("/nonexistent/harness"), failure=object())


def test_propose_body_still_unimplemented_under_cap():
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    with pytest.raises(NotImplementedError):
        e.propose(Path("/nonexistent/harness"), failure=object())
