"""Benchmark task: verify bash skill executes real subprocess (issue #416).

This task is a smoke-tier benchmark that proves the ``bash`` skill
executor runs an actual subprocess and returns genuine stdout, stderr,
and exit code -- not the fake ``{"status": "ok"}`` stub that issue #416
identified as silently passing benchmark runs.

The task is intentionally minimal: it drives ``Runner.run_task`` with a
stub ``ModelAdapter`` that emits one ``bash`` tool call (``echo hello``),
then a final assistant message. The test asserts the captured ``tool_result``
event carries real subprocess output (``stdout == "hello\\n"``, ``exit_code == 0``)
and that no ``error`` field is present.

Because the task exercises the full asyncio loop and skill dispatcher
(but not a real LLM), it lives under ``benchmarks/tasks/`` alongside other
``easy``-tier integration benchmarks. It requires the ``bash`` skill
to be implemented and registered, so it is added to ``requires_skills``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger

TASK = BenchmarkTask(
    name="bash_real_exec",
    description=(
        "Drive Runner.run_task with a stub adapter emitting one bash tool_call "
        "(echo hello); assert the tool_result carries real subprocess output "
        "(stdout, exit_code, no error field) proving the skill executor is "
        "not the fake stub from issue #416."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "tool_result payload for the bash call carries stdout='hello\\n', "
        "exit_code=0, and no error field."
    ),
    tags=["bash", "skill-executor", "integration"],
)


class _BashAdapter:
    """Stub adapter that emits one bash tool_call then a final answer.

    Verifies the bash skill executor runs a real subprocess and returns
    genuine output (not the fake stub).
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            tool_call = ModelToolCall(
                id="call_bash_real",
                type="function",
                function=ToolCallFunction(
                    name="bash",
                    arguments=json.dumps({"command": "echo hello"}),
                ),
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            )
        if self.calls == 2:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )
        raise RuntimeError(
            f"_BashAdapter exhausted after 2 scripted responses; loop called "
            f"complete() {self.calls} times (possible runaway loop)"
        )

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        response = await self.complete(messages, tools, **kwargs)
        if response.message.content:
            yield ModelResponseChunk(content=response.message.content)
        for i, tc in enumerate(response.tool_calls or []):
            yield ModelResponseChunk(
                tool_calls=[
                    ModelToolCallChunk(
                        index=i,
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunctionChunk(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                ]
            )


def _minimal_harness(harness_dir: Path) -> Path:
    """Build a minimal harness with the bash skill registered."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for bash_real_exec\n")
    skills_dir = harness_dir / "skills"
    skills_dir.mkdir(exist_ok=True)
    (skills_dir / "__init__.py").write_text("")
    return harness_dir


@pytest.mark.benchmark
def test_bash_real_exec(benchmark_workspace: Path) -> None:
    """Verify bash skill executor returns real subprocess output (issue #416).

    Drives ``Runner.run_task`` with a stub adapter that issues one
    ``bash`` tool call (``echo hello``). The test then loads the trace
    and asserts the captured ``tool_result`` for that call carries:

    - ``output["stdout"] == "hello\\n"``  (real subprocess output)
    - ``output["exit_code"] == 0``         (real exit code)
    - ``"error" not in output or output["error"] is None``  (not a stub)

    This is the minimal proof that the bash skill executor is not the
    fake stub that was silently returning ``{"status": "ok"}``.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = _minimal_harness(benchmark_workspace / "harness")

    async def _drive() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "bash-real-exec",
                harness_dir,
                logger,
                session_id,
                model_adapter=_BashAdapter(),
            )

    asyncio.run(_drive())

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)

    tool_results = [e for e in events if e.kind == "tool_result"]
    assert (
        tool_results
    ), f"expected at least one tool_result event; got kinds={[e.kind for e in events]}"

    bash_result = None
    for event in tool_results:
        if event.payload.get("name") == "bash":
            bash_result = event.payload
            break

    assert (
        bash_result is not None
    ), f"expected a tool_result for 'bash'; got ={[e.payload.get('name') for e in tool_results]}"

    output = bash_result.get("output")
    assert output is not None, f"tool_result output must not be None; got {bash_result!r}"

    assert output.get("stdout") == "hello\n", (
        f"expected stdout='hello\\n'; got {output.get('stdout')!r}. "
        f"The bash skill may still be returning the fake stub."
    )
    assert output.get("exit_code") == 0, f"expected exit_code=0; got {output.get('exit_code')!r}"
    error = output.get("error")
    assert error is None, f"expected no error field; got {error!r}"

    assert (
        bash_result.get("error") is None
    ), f"expected tool_result.error=None; got {bash_result.get('error')!r}"
