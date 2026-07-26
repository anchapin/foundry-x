"""Tests for issue #977: trace the LLM→template fallback in propose/propose_async.

When ``generate_edits`` raises ``EvolverLLMError``, ``propose_async`` and
``propose`` fall back to ``_propose_from_template``. Previously this fallback
was silent — no trace event marked the transition. An operator reading the
trace could not tell whether a session's ``proposed_edit`` events came from
the LLM path or a silent template fallback.

These tests assert that a ``generation_attempt`` event with
``error="llm_fallback"`` is emitted *before* the template proposals are
returned on the fallback path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    GENERATION_ATTEMPT_KIND,
    PROPOSED_EDIT_KIND,
    Evolver,
    EvolverLLMError,
)


def _make_failure() -> FailureReport:
    return FailureReport(
        session_id="sess-977",
        summary="agent used the wrong tool",
        proposed_class="wrong-tool",
    )


def _build_harness(tmp_path: Path) -> Path:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("You are a helpful agent.\n", encoding="utf-8")
    return harness_dir


def _fallback_payloads(trace_logger: MagicMock) -> list[dict]:
    """Extract the ``generation_attempt`` payloads whose error starts with llm_fallback."""
    payloads = []
    for call in trace_logger.record.call_args_list:
        if call.args[1] != GENERATION_ATTEMPT_KIND:
            continue
        payload = call.args[2]
        if str(payload.get("error", "")).startswith("llm_fallback"):
            payloads.append(payload)
    return payloads


class TestProposeAsyncLlmFallbackTrace:
    """propose_async must emit an llm_fallback generation_attempt event."""

    @pytest.mark.asyncio
    async def test_emits_llm_fallback_event_before_template(self, tmp_path: Path) -> None:
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-977",
            model_adapter=MagicMock(),
        )
        # Force the LLM path to fail so the fallback fires.
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("boom"))

        edits = await evolver.propose_async(harness_dir, failure=_make_failure())

        # Template proposal still produced.
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"

        # Exactly one llm_fallback generation_attempt event.
        fallbacks = _fallback_payloads(trace_logger)
        assert len(fallbacks) == 1
        assert fallbacks[0]["error"].startswith("llm_fallback")
        assert "boom" in fallbacks[0]["error"]

    @pytest.mark.asyncio
    async def test_fallback_event_precedes_proposed_edit(self, tmp_path: Path) -> None:
        """The llm_fallback attempt must be recorded before the proposed_edit."""
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-977",
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("boom"))

        await evolver.propose_async(harness_dir, failure=_make_failure())

        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        fallback_idx = next(
            i
            for i, kind in enumerate(kinds)
            if kind == GENERATION_ATTEMPT_KIND
            and str(trace_logger.record.call_args_list[i].args[2].get("error", "")).startswith(
                "llm_fallback"
            )
        )
        edit_idx = kinds.index(PROPOSED_EDIT_KIND)
        assert fallback_idx < edit_idx, "llm_fallback event must precede proposed_edit"

    @pytest.mark.asyncio
    async def test_no_fallback_event_on_llm_success(self, tmp_path: Path) -> None:
        """A successful LLM generation must not emit an llm_fallback event."""
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-977",
            model_adapter=MagicMock(),
        )
        # LLM path succeeds → no fallback.
        from foundry_x.evolution.evolver import ProposedEdit

        good_edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="llm fix",
            unified_diff=(
                "--- a/harness/system_prompt.txt\n"
                "+++ b/harness/system_prompt.txt\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        )
        evolver.generate_edits = AsyncMock(return_value=[good_edit])

        edits = await evolver.propose_async(harness_dir, failure=_make_failure())

        assert edits == [good_edit]
        assert _fallback_payloads(trace_logger) == []

    @pytest.mark.asyncio
    async def test_no_fallback_event_without_model_adapter(self, tmp_path: Path) -> None:
        """Without a model adapter the template path runs directly (no fallback)."""
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-977")

        edits = await evolver.propose_async(harness_dir, failure=_make_failure())

        assert len(edits) == 1
        assert _fallback_payloads(trace_logger) == []


class TestProposeLlmFallbackTrace:
    """The sync propose() must emit the same llm_fallback event."""

    def test_emits_llm_fallback_event(self, tmp_path: Path) -> None:
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-977",
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("sync boom"))

        edits = evolver.propose(harness_dir, failure=_make_failure())

        assert len(edits) == 1
        fallbacks = _fallback_payloads(trace_logger)
        assert len(fallbacks) == 1
        assert "sync boom" in fallbacks[0]["error"]

    def test_fallback_event_precedes_proposed_edit(self, tmp_path: Path) -> None:
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-977",
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("boom"))

        evolver.propose(harness_dir, failure=_make_failure())

        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        fallback_idx = next(
            i
            for i, kind in enumerate(kinds)
            if kind == GENERATION_ATTEMPT_KIND
            and str(trace_logger.record.call_args_list[i].args[2].get("error", "")).startswith(
                "llm_fallback"
            )
        )
        edit_idx = kinds.index(PROPOSED_EDIT_KIND)
        assert fallback_idx < edit_idx


class TestNoTraceLogger:
    """Without a trace_logger the fallback still works (no event emitted)."""

    @pytest.mark.asyncio
    async def test_propose_async_fallback_without_logger(self, tmp_path: Path) -> None:
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(model_adapter=MagicMock())
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("no logger"))

        edits = await evolver.propose_async(harness_dir, failure=_make_failure())

        assert len(edits) == 1
