"""Unit tests for :mod:`foundry_x.evaluation.humaneval_plus` (issue #900).

These tests pin the JSONL loader's contract and the canonical/candidate
solution runners' edge-case behaviour. They write fixtures into pytest's
``tmp_path`` so no on-disk state escapes the test session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from foundry_x.evaluation.humaneval_plus import (
    HumanEvalExecutionError,
    HumanEvalTask,
    load_humaneval_slice,
    run_canonical_solution,
    run_candidate_solution,
    slice_pass_rates,
)


def _make_task(
    *,
    task_id: str = "HumanEval/test",
    entry_point: str = "f",
    prompt: str | None = None,
    canonical_solution: str | None = None,
    test: str | None = None,
) -> HumanEvalTask:
    if prompt is None:
        prompt = '\n\ndef f(x: int) -> int:\n    """ Return x + 1. """\n'
    if canonical_solution is None:
        canonical_solution = "    return x + 1\n"
    if test is None:
        test = (
            "def check(candidate):\n    assert candidate(0) == 1\n    assert candidate(41) == 42\n"
        )
    return HumanEvalTask(
        task_id=task_id,
        prompt=prompt,
        canonical_solution=canonical_solution,
        test=test,
        entry_point=entry_point,
    )


def _write_slice(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "slice.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_human_eval_task_rejects_empty_task_id() -> None:
    with pytest.raises(ValidationError):
        _make_task(task_id="")


def test_human_eval_task_rejects_empty_entry_point() -> None:
    with pytest.raises(ValidationError):
        _make_task(entry_point="   ")


def test_run_canonical_solution_passes() -> None:
    assert run_canonical_solution(_make_task()) is True


def test_run_canonical_solution_returns_false_on_assertion_failure() -> None:
    task = _make_task(canonical_solution="    return x\n")  # returns x, not x+1
    assert run_canonical_solution(task) is False


def test_run_canonical_solution_raises_on_undefined_entry_point() -> None:
    task = _make_task(canonical_solution="    return 0\n", entry_point="missing")
    with pytest.raises(HumanEvalExecutionError, match="entry_point"):
        run_canonical_solution(task)


def test_run_canonical_solution_raises_on_test_without_check() -> None:
    task = _make_task(test="def not_check(candidate):\n    pass\n")
    with pytest.raises(HumanEvalExecutionError, match="callable check"):
        run_canonical_solution(task)


def test_run_candidate_solution_wrong_returns_false() -> None:
    assert run_candidate_solution(_make_task(), "    return x\n") is False


def test_run_candidate_solution_correct_returns_true() -> None:
    assert run_candidate_solution(_make_task(), "    return x + 1\n") is True


def test_run_candidate_solution_syntax_error_raises() -> None:
    with pytest.raises(HumanEvalExecutionError, match="failed to import"):
        run_candidate_solution(_make_task(), "    return this is broken!!!\n")


def test_load_humaneval_slice_round_trip(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": '\n\ndef f(x: int) -> int:\n    """ identity """\n',
            "canonical_solution": "    return x\n",
            "test": "def check(candidate):\n    assert candidate(7) == 7\n",
            "entry_point": "f",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": '\n\ndef g(x: int) -> int:\n    """ double """\n',
            "canonical_solution": "    return x * 2\n",
            "test": "def check(candidate):\n    assert candidate(3) == 6\n",
            "entry_point": "g",
        },
    ]
    p = _write_slice(tmp_path, rows)
    tasks = load_humaneval_slice(p)
    assert [t.task_id for t in tasks] == ["HumanEval/0", "HumanEval/1"]
    passed, total = slice_pass_rates(tasks)
    assert (passed, total) == (2, 2)


def test_load_humaneval_slice_skips_blank_lines(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "\n\ndef f():\n    pass\n",
            "canonical_solution": "    pass\n",
            "test": "def check(candidate):\n    candidate()\n",
            "entry_point": "f",
        }
    ]
    p = tmp_path / "slice.jsonl"
    p.write_text("\n\n" + json.dumps(rows[0]) + "\n\n\n", encoding="utf-8")
    tasks = load_humaneval_slice(p)
    assert len(tasks) == 1


def test_load_humaneval_slice_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_humaneval_slice(tmp_path / "nope.jsonl")


def test_load_humaneval_slice_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="slice is empty"):
        load_humaneval_slice(p)


def test_load_humaneval_slice_malformed_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON"):
        load_humaneval_slice(p)


def test_load_humaneval_slice_invalid_schema_raises(tmp_path: Path) -> None:
    p = _write_slice(tmp_path, [{"task_id": "", "entry_point": "f"}])
    with pytest.raises(ValueError, match="invalid HumanEvalTask"):
        load_humaneval_slice(p)
