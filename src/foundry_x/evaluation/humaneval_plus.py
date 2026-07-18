"""HumanEval+ JSONL loader and canonical-solution runner (issue #900, ADR-0023).

The external half of the validation study is a slice of the EvalPlus
`HumanEval+` dataset. Each task in the slice carries:

- ``task_id``           -- EvalPlus-style id (e.g. ``"HumanEval/0"``).
- ``prompt``            -- the function signature + docstring handed
                           to the agent.
- ``canonical_solution``-- the dataset's reference solution, used to
                           validate the plumbing offline.
- ``test``              -- the body of a ``check(candidate)`` function
                           (EvalPlus format): the body must reference a
                           ``candidate`` parameter and may assert
                           directly.
- ``entry_point``       -- the function name the agent must implement.
- ``context``           -- optional free-form context (unused by the
                           runner; included for traceability).

This module is *machinery*, not an agent. It reads the slice from disk,
materializes each task's ``check`` function deterministically, and
reports whether the canonical solution passes -- proving the test
harness works without invoking a live model.

The real-model invocation (Runner drives the agent against the slice)
lives in ``infra/scripts/run_external_eval.sh``; this module's
:func:`run_canonical_solution` is what that script calls to verify the
slice still parses before it spends model tokens.

Security
--------
:func:`run_canonical_solution` and :func:`run_candidate_solution` exec
Python source. The source is **not** the agent's output: it is the
dataset's reference solution or a candidate supplied by the orchestrator
(not by an untrusted tool call). Exec happens inside a fresh
``dict``-scoped ``globals`` with ``__name__`` set to ``"__not_main__"``
so a solution's ``if __name__ == "__main__"`` block does not fire during
the check. The orchestrator script is responsible for any further
sandboxing (Docker, etc.) when it invokes the agent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HumanEvalTask(BaseModel):
    """A single HumanEval+ task.

    The schema mirrors the EvalPlus JSONL shape exactly so an operator
    can swap our 20-task slice for a full 164-task file (or a SWE-bench
    export adapted to the same shape) without touching the loader.
    """

    task_id: str = Field(
        ...,
        description="EvalPlus-style task id (e.g. 'HumanEval/0').",
    )
    prompt: str = Field(
        ...,
        description=(
            "Function signature + docstring handed to the agent. The "
            "agent's job is to complete the body."
        ),
    )
    canonical_solution: str = Field(
        ...,
        description="The dataset's reference solution (function body only).",
    )
    test: str = Field(
        ...,
        description=(
            "Source of a check(candidate) function in EvalPlus format. Must "
            "include the `def check(candidate):` line; the loader execs the "
            "string verbatim (no wrapping) so a real EvalPlus JSONL row can "
            "be dropped in unchanged."
        ),
    )
    entry_point: str = Field(
        ...,
        description="Function name the agent must implement.",
    )
    context: str = Field(
        default="",
        description="Optional free-form context (unused by the runner).",
    )

    @field_validator("task_id", "entry_point")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value


def load_humaneval_slice(path: str | Path) -> list[HumanEvalTask]:
    """Load a HumanEval+ JSONL slice into typed :class:`HumanEvalTask` rows.

    Args:
        path: Path to a ``.jsonl`` file with one JSON object per line.

    Returns:
        Tasks in file order (EvalPlus convention preserves file order
        so task ids are addressable by index for downstream tooling).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty, contains a malformed line,
            or any row fails :class:`HumanEvalTask` validation. We
            surface rather than swallow (AGENTS.md S2): a corrupt slice
            must not silently degrade the study.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"HumaEval+ slice not found: {p}")

    tasks: list[HumanEvalTask] = []
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{line_no}: malformed JSON: {exc}") from exc
        try:
            tasks.append(HumanEvalTask.model_validate(obj))
        except Exception as exc:  # noqa: BLE001 -- re-raised with location
            raise ValueError(f"{p}:{line_no}: invalid HumanEvalTask: {exc}") from exc

    if not tasks:
        raise ValueError(f"{p}: slice is empty (no non-blank lines)")
    return tasks


def _build_check_namespace(task: HumanEvalTask) -> dict[str, Any]:
    """Build the namespace the EvalPlus check function runs in.

    The namespace contains the candidate entry point plus any helpers
    the test body references via the standard library; everything else
    must be imported inside the test body itself (EvalPlus convention).
    """
    candidate_src = task.prompt + task.canonical_solution
    namespace: dict[str, Any] = {"__name__": "__not_main__"}
    try:
        exec(candidate_src, namespace)  # noqa: S102 -- trusted source, see module doc
    except Exception as exc:  # noqa: BLE001 -- surfaced with task id
        raise HumanEvalExecutionError(
            f"{task.task_id}: canonical solution failed to import: {exc!r}"
        ) from exc
    if task.entry_point not in namespace:
        raise HumanEvalExecutionError(
            f"{task.task_id}: entry_point '{task.entry_point}' not defined "
            "by prompt + canonical_solution"
        )
    namespace["candidate"] = namespace[task.entry_point]
    return namespace


def _compile_check(task: HumanEvalTask, namespace: dict[str, Any]) -> None:
    """Exec ``task.test`` in ``namespace`` and validate it defines ``check``.

    The EvalPlus convention is that ``task.test`` is the source of a
    function named ``check`` whose parameter is named ``candidate``;
    the loader execs the string verbatim (no wrapping) so a real
    EvalPlus JSONL row is interchangeable with our slice.

    Raises :class:`HumanEvalExecutionError` if compilation fails or the
    test source does not define a callable ``check``. We surface rather
    than swallow (AGENTS.md S2): a malformed test row would otherwise
    silently score every candidate as ``False`` and inflate the
    external pass rate the wrong way.
    """
    try:
        exec(task.test, namespace)  # noqa: S102 -- trusted source, see module doc
    except Exception as exc:  # noqa: BLE001 -- re-raised with task id
        raise HumanEvalExecutionError(
            f"{task.task_id}: test source failed to compile: {exc!r}"
        ) from exc
    check_fn = namespace.get("check")
    if not callable(check_fn):
        raise HumanEvalExecutionError(
            f"{task.task_id}: test source did not define a callable check()"
        )


def run_canonical_solution(task: HumanEvalTask) -> bool:
    """Run the dataset's reference solution against the task's check function.

    Returns ``True`` if the canonical solution passes every assertion in
    ``task.test``; ``False`` if any assertion fails. Any other exception
    (``SyntaxError`` in the test source, ``NameError`` from an undefined
    helper, etc.) is wrapped in :class:`HumanEvalExecutionError` and
    re-raised so the orchestrator can distinguish "the test harness
    itself is broken" from "the canonical solution is wrong" -- the
    latter is a dataset bug, the former is a machinery bug.

    Used by the plumbing-validation benchmark task under
    ``benchmarks/tasks/test_external_eval_correlation.py`` and by the
    real-model orchestrator under ``infra/scripts/run_external_eval.sh``
    as a pre-flight check.
    """
    namespace = _build_check_namespace(task)
    _compile_check(task, namespace)
    check_fn = namespace["check"]
    try:
        check_fn(namespace["candidate"])
    except AssertionError as exc:
        _log_plumbing_failure(task, exc)
        return False
    return True


def run_candidate_solution(task: HumanEvalTask, candidate_body: str) -> bool:
    """Run an arbitrary candidate function body against the task's check.

    Used by ``infra/scripts/run_external_eval.sh`` to score the agent's
    output for each task in the slice. ``candidate_body`` is the body
    the agent produced (without the signature); it is concatenated onto
    ``task.prompt`` so the entry point resolves identically to the
    canonical path.

    Returns ``True`` if the candidate passes, ``False`` on assertion
    failure. Any other exception is wrapped in
    :class:`HumanEvalExecutionError` -- a model that emits a syntax
    error should fail the task loudly so the orchestrator records it as
    a non-pass rather than silently inflating the pass rate.
    """
    namespace: dict[str, Any] = {"__name__": "__not_main__"}
    candidate_src = task.prompt + candidate_body
    try:
        exec(candidate_src, namespace)  # noqa: S102 -- see module doc on trust boundary
    except Exception as exc:  # noqa: BLE001 -- surfaced with task id
        raise HumanEvalExecutionError(
            f"{task.task_id}: candidate solution failed to import: {exc!r}"
        ) from exc
    if task.entry_point not in namespace:
        raise HumanEvalExecutionError(
            f"{task.task_id}: candidate did not define entry_point '{task.entry_point}'"
        )
    namespace["candidate"] = namespace[task.entry_point]
    _compile_check(task, namespace)
    check_fn = namespace["check"]
    try:
        check_fn(namespace["candidate"])
    except AssertionError as exc:
        _log_plumbing_failure(task, exc)
        return False
    return True


def _log_plumbing_failure(task: HumanEvalTask, exc: BaseException) -> None:
    """Record a plumbing failure to the project's TraceLogger when one is bound.

    The plumbing-validation benchmark task does not have a TraceLogger in
    scope, and importing the TraceLogger here would create a layering
    violation (machinery should not depend on the trace store). We emit
    to stderr via the standard library only; the orchestrator script
    wires the real TraceLogger for the live run.
    """
    import sys

    print(
        f"[humaneval_plus] {task.task_id}: assertion failed: {exc!r}",
        file=sys.stderr,
    )


class HumanEvalExecutionError(RuntimeError):
    """Raised when the test harness itself is broken (not when a solution fails).

    Distinguishes "the slice is malformed" / "the test body has a
    syntax error" from "the candidate produced wrong output". The
    latter is a normal ``False`` return; the former must surface.
    """


def slice_pass_rates(tasks: Sequence[HumanEvalTask]) -> tuple[int, int]:
    """Return ``(passed, total)`` for the canonical solutions across ``tasks``.

    Convenience wrapper for plumbing-validation callers that just want
    the aggregate count. Each task's canonical solution is run via
    :func:`run_canonical_solution`; an unexpected
    :class:`HumanEvalExecutionError` is allowed to propagate so a
    broken slice cannot masquerade as a 0/N pass rate.
    """
    passed = 0
    for task in tasks:
        if run_canonical_solution(task):
            passed += 1
    return passed, len(tasks)


__all__ = [
    "HumanEvalExecutionError",
    "HumanEvalTask",
    "load_humaneval_slice",
    "run_canonical_solution",
    "run_candidate_solution",
    "slice_pass_rates",
]
