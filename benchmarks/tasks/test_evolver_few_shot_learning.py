"""Benchmark task: Evolver few-shot learning from approved_edit events (issue #957).

Verifies that when `approved_edit` events exist in the trace store for a given
failure_class, the Evolver's LLM prompt includes them as few-shot examples and
the generated edit differs from the template-based fallback.

This is a regression target for the few-shot learning path:
- `_get_past_successful_edits` queries approved edits from the trace store
- `generate_edits` injects them into the LLM prompt via `_build_llm_prompt`
- The LLM response differs from the zero-shot template fallback

A regression that breaks the few-shot path (e.g., by removing the
`_get_past_successful_edits` call from `generate_edits`, or by bypassing
`_build_llm_prompt` in favour of a prompt builder that ignores few-shot edits)
surfaces here as a failing benchmark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    Evolver,
    ProposedEdit,
)
from foundry_x.trace.logger import TraceLogger


TASK = BenchmarkTask(
    name="evolver_few_shot_learning",
    description=(
        "Evolver.propose() with model_adapter configured and prior approved_edit "
        "events for the same failure_class: verify the LLM prompt includes "
        "few-shot examples and the result differs from the zero-shot fallback."
    ),
    prompt=(
        "Inspect src/foundry_x/evolution/evolver.py: confirm generate_edits "
        "calls _get_past_successful_edits and passes the result to "
        "_build_llm_prompt as few_shot_edits; confirm _build_llm_prompt "
        "includes PREVIOUSLY SUCCESSFUL EDITS section when few_shot_edits is "
        "non-empty; run benchmarks/tasks/test_evolver_few_shot_learning.py "
        "and assert all tests pass."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "generate_edits produces edits that incorporate past successful patterns; "
        "the few-shot path is exercised when approved_edit events exist for the "
        "failure_class; _get_past_successful_edits returns at most 5 edits, "
        "sorted by target_file, deduplicated by target_file."
    ),
    tags=["evolution", "few-shot", "llm"],
)


def _make_diff(target_file: str, *lines: str) -> str:
    return f"--- a/{target_file}\n+++ b/{target_file}\n@@ -0,0 +1 @@\n" + "".join(
        f"+{line}\n" for line in lines
    )


def _edit(
    target_file: str = "harness/system_prompt.txt",
    diff: str | None = None,
    rationale: str = "tighten tool guidance",
) -> ProposedEdit:
    if diff is None:
        diff = _make_diff(target_file, "be more precise")
    return ProposedEdit(
        target_file=target_file,
        rationale=rationale,
        unified_diff=diff,
    )


@pytest.fixture
def evolver_with_few_shot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Evolver:
    """Evolver pre-seeded with two approved_edit events for 'wrong-tool'."""
    logger = TraceLogger(tmp_path / "trace.db")
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("original system prompt\n", encoding="utf-8")

    with logger.session("harness-v1") as session_id:
        evolver = Evolver(
            trace_logger=logger,
            session_id=session_id,
            max_proposals_per_hour=10,
            max_diff_lines=200,
        )
        edit1 = _edit(
            "harness/system_prompt.txt",
            _make_diff("harness/system_prompt.txt", "check tool list first"),
            "added tool list guidance",
        )
        edit2 = _edit(
            "harness/hooks/check_tool.py",
            _make_diff("harness/hooks/check_tool.py", "def validate(): pass"),
            "added validation hook",
        )
        evolver._record_approved_edit(edit1, failure_class="wrong-tool")
        evolver._record_approved_edit(edit2, failure_class="wrong-tool")

    return evolver


class _SpyModelAdapter:
    """Spy adapter that captures the prompt and returns a deterministic edit."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    async def complete(self, messages: list[Any], **kwargs: Any) -> Any:
        self.calls.append(messages)
        mock_response = type(
            "R",
            (),
            {
                "message": type(
                    "M",
                    (),
                    {
                        "content": (
                            '[{"target_file": "harness/system_prompt.txt", '
                            '"rationale": "llm suggested edit", '
                            '"unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\noriginal\\nupdated\\n"}]'
                        )
                    },
                )()
            },
        )()
        return mock_response


@pytest.mark.benchmark
def test_few_shot_prompt_includes_approved_edit_examples(
    evolver_with_few_shot: Evolver,
    tmp_path: Path,
) -> None:
    """LLM prompt contains PREVIOUSLY SUCCESSFUL EDITS section when few-shot events exist."""
    harness_dir = tmp_path / "harness"
    (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

    adapter = _SpyModelAdapter()
    failure = FailureReport(
        session_id="s",
        summary="tool used incorrectly",
        proposed_class="wrong-tool",
        failed_steps=[{"step": "invoke", "tool": "bash"}],
    )

    import asyncio

    result = asyncio.run(evolver_with_few_shot.generate_edits(adapter, harness_dir, failure))

    assert len(result) == 1
    assert adapter.calls  # adapter was called
    user_prompt = next(
        m.content for call in adapter.calls for m in call if hasattr(m, "role") and m.role == "user"
    )
    assert "PREVIOUSLY SUCCESSFUL EDITS" in user_prompt
    assert "check tool list first" in user_prompt
    assert "added tool list guidance" in user_prompt


@pytest.mark.benchmark
def test_few_shot_propose_diff_differs_from_template_fallback(
    evolver_with_few_shot: Evolver,
    tmp_path: Path,
) -> None:
    """Propose result differs from zero-shot template fallback when few-shot examples exist."""
    harness_dir = tmp_path / "harness"
    (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

    adapter = _SpyModelAdapter()
    failure = FailureReport(
        session_id="s",
        summary="tool used incorrectly",
        proposed_class="wrong-tool",
        failed_steps=[{"step": "invoke", "tool": "bash"}],
    )

    import asyncio

    result = asyncio.run(evolver_with_few_shot.generate_edits(adapter, harness_dir, failure))

    assert len(result) == 1
    fallback = evolver_with_few_shot._propose_from_template(harness_dir, failure)
    assert len(fallback) >= 1
    assert result[0].unified_diff != fallback[0].unified_diff


@pytest.mark.benchmark
def test_get_past_successful_edits_returns_at_most_five(
    evolver_with_few_shot: Evolver,
) -> None:
    """_get_past_successful_edits returns no more than 5 edits."""
    past = evolver_with_few_shot._get_past_successful_edits("wrong-tool")
    assert len(past) <= 5


@pytest.mark.benchmark
def test_get_past_successful_edits_deduplicated_by_target_file(
    evolver_with_few_shot: Evolver,
) -> None:
    """_get_past_successful_edits returns only one edit per target_file."""
    past = evolver_with_few_shot._get_past_successful_edits("wrong-tool")
    target_files = [e.target_file for e in past]
    assert len(target_files) == len(set(target_files)), "duplicate target_file entries found"


@pytest.mark.benchmark
def test_get_past_successful_edits_sorted_by_target_file(
    evolver_with_few_shot: Evolver,
) -> None:
    """_get_past_successful_edits returns edits sorted alphabetically by target_file."""
    past = evolver_with_few_shot._get_past_successful_edits("wrong-tool")
    target_files = [e.target_file for e in past]
    assert target_files == sorted(target_files), "edits not sorted by target_file"
