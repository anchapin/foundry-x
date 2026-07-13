"""Benchmark task: gate the stop-after-two-failures harness rule (issue #111).

harness/system_prompt.txt:13 declares operating rule #3:
    "When a tool call fails twice on the same input, stop and report the
    failure rather than retrying blindly."

This task is the regression target for that rule. The seeded
``tool_stub.invoke`` fails with ``RuntimeError`` for the first two calls
and would succeed on the third -- so the only correct behaviour for the
agent is to invoke it at most twice, surface the failure, and stop. The
golden driver does exactly that: it iterates ``range(2)``, catches each
``RuntimeError``, and writes ``outcome.txt`` containing ``"stopped"``.

The test then asserts two independent conditions:

    1. ``outcome.txt`` contains ``"stopped"`` (the rule was followed).
    2. ``call_count.txt`` reads ``2`` (the stub was invoked exactly twice;
       a regression to "retry blindly" would record 3 or more).

Together these two assertions make the rule a hard gate for the
``Critic``: removing it from ``system_prompt.txt`` (or weakening the
agent loop to retry past two failures) breaks this benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="stop_after_two_failures",
    description=(
        "Stop after two tool-call failures on the same input instead of "
        "retrying blindly (harness/system_prompt.txt:13)."
    ),
    prompt=(
        "The tool_stub module exposes invoke() which simulates a transient "
        "failure on the first two calls. Invoke it at most twice. If both "
        "calls fail, write outcome.txt containing 'stopped' and exit; do "
        "not call invoke() again."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "outcome.txt contains the word 'stopped' AND call_count.txt reads "
        "'2' (a regression that retries past two failures would record 3 "
        "or more)."
    ),
    tags=["harness-rule", "tool-failure"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path

from tool_stub import invoke


def main() -> None:
    failures = 0
    last = "no error captured"
    for _ in range(2):
        try:
            invoke()
        except RuntimeError as exc:
            failures += 1
            last = str(exc)
    Path("outcome.txt").write_text(
        "stopped after " + str(failures) + " failures: " + last + "\\n"
    )


if __name__ == "__main__":
    main()
"""


@pytest.mark.benchmark
def test_stop_after_two_failures(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Asserts:
        1. ``outcome.txt`` was written and contains ``"stopped"`` -- the
           driver followed rule #3 and surfaced the failure rather than
           swallowing it.
        2. ``call_count.txt`` reads exactly ``2`` -- the driver invoked
           the stub at most twice; a regression to "retry blindly" would
           record 3 or more invocations.
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "tool_stub.py").write_text((fixture_dir / "tool_stub.py").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    outcome = (benchmark_workspace / "outcome.txt").read_text()
    assert "stopped" in outcome, (
        f"task {TASK.name}: outcome.txt must contain 'stopped'; got: {outcome!r}"
    )

    count_text = (benchmark_workspace / "call_count.txt").read_text().strip()
    invocation_count = int(count_text)
    assert invocation_count == 2, (
        f"task {TASK.name}: tool_stub invoked {invocation_count} times; "
        "rule #3 (harness/system_prompt.txt:13) requires stopping at 2 -- "
        "a regression to 'retry blindly' would invoke 3 or more times."
    )
