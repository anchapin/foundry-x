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
    async def test_emits_llm_fallback_event_before_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-977",
            model_adapter=MagicMock(),
        )
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
    async def test_fallback_event_precedes_proposed_edit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The llm_fallback attempt must be recorded before the proposed_edit."""
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
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
    async def test_no_fallback_event_on_llm_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful LLM generation must not emit an llm_fallback event."""
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
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

    def test_emits_llm_fallback_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
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

    def test_fallback_event_precedes_proposed_edit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
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
    async def test_propose_async_fallback_without_logger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(model_adapter=MagicMock())
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("no logger"))

        edits = await evolver.propose_async(harness_dir, failure=_make_failure())

        assert len(edits) == 1


class TestEvolverLlmEnvVar:
    """Tests for issue #1116: FOUNDRY_EVOLVER_LLM_ENABLED env var gates LLM edit generation.

    Without the env var set, the Evolver uses the template path even when a
    ModelAdapter is configured. When the env var is set to "1" or "true",
    LLM-driven generation is activated.
    """

    @pytest.mark.asyncio
    async def test_llm_path_taken_when_env_var_set_to_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FOUNDRY_EVOLVER_LLM_ENABLED=true, LLM path is used with ModelAdapter."""
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-1116",
            model_adapter=MagicMock(),
        )
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
        assert evolver.generate_edits.call_count == 1

    @pytest.mark.asyncio
    async def test_llm_path_taken_when_env_var_set_to_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FOUNDRY_EVOLVER_LLM_ENABLED=1, LLM path is used with ModelAdapter."""
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "1")
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(
            model_adapter=MagicMock(),
        )
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
        assert evolver.generate_edits.call_count == 1

    def test_template_path_used_when_env_var_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FOUNDRY_EVOLVER_LLM_ENABLED is not set, template path is used."""
        monkeypatch.delenv("FOUNDRY_EVOLVER_LLM_ENABLED", raising=False)
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = MagicMock()

        edits = evolver.propose(harness_dir, failure=_make_failure())

        assert len(edits) == 1
        assert evolver.generate_edits.call_count == 0

    def test_template_path_used_when_env_var_set_to_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When FOUNDRY_EVOLVER_LLM_ENABLED=false, template path is used."""
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "false")
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = MagicMock()

        edits = evolver.propose(harness_dir, failure=_make_failure())

        assert len(edits) == 1
        assert evolver.generate_edits.call_count == 0

    def test_template_path_used_when_model_adapter_not_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ModelAdapter is None, template path is used regardless of env var."""
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver()
        evolver.generate_edits = MagicMock()

        edits = evolver.propose(harness_dir, failure=_make_failure())

        assert len(edits) == 1
        assert evolver.generate_edits.call_count == 0

    @pytest.mark.asyncio
    async def test_batch_uses_llm_when_env_var_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """propose_batch uses LLM path when env var is set and ModelAdapter is configured.

        Issue #1259: The LLM is called once per batch via _generate_batch_edits_async,
        not once per failure via generate_edits.
        """
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
        from foundry_x.evolution.digester import BatchFailureReport

        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(
            model_adapter=MagicMock(),
        )
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
        evolver._generate_batch_edits_async = AsyncMock(return_value=[good_edit])

        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[_make_failure()],
            total_failures=1,
        )
        edits = await evolver.propose_batch_async(harness_dir, batch)

        assert len(edits) == 1
        assert evolver._generate_batch_edits_async.call_count == 1

    def test_batch_uses_template_when_env_var_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """propose_batch uses template path when env var is not set."""
        monkeypatch.delenv("FOUNDRY_EVOLVER_LLM_ENABLED", raising=False)
        from foundry_x.evolution.digester import BatchFailureReport

        harness_dir = _build_harness(tmp_path)
        evolver = Evolver(
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = MagicMock()

        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[_make_failure()],
            total_failures=1,
        )
        edits = evolver.propose_batch(harness_dir, batch)

        assert len(edits) == 1
        assert evolver.generate_edits.call_count == 0

    @pytest.mark.asyncio
    async def test_unknown_class_llm_fallback_still_emits_llm_fallback_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown class with no template still emits llm_fallback event when LLM fails.

        This is the acceptance criteria: when proposed_class=unknown produces no template
        entry, if FOUNDRY_EVOLVER_LLM_ENABLED is set and the LLM fails, an
        llm_fallback event should be emitted.
        """
        monkeypatch.setenv("FOUNDRY_EVOLVER_LLM_ENABLED", "true")
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-1116",
            model_adapter=MagicMock(),
        )
        evolver.generate_edits = AsyncMock(side_effect=EvolverLLMError("unknown class"))

        unknown_failure = FailureReport(
            session_id="sess-1116",
            summary="agent did something unexpected",
            proposed_class="unknown-class",
        )
        edits = await evolver.propose_async(harness_dir, failure=unknown_failure)

        assert edits == []
        fallbacks = _fallback_payloads(trace_logger)
        assert len(fallbacks) == 1
        assert fallbacks[0]["error"].startswith("llm_fallback")
