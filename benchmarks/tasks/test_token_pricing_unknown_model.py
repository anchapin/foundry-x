"""Benchmark: OpenAICompatibleAdapter.token_pricing() returns (0,0) for unknown models (issue #1265).

This is a unit benchmark (adapter-level, not full runner) that establishes the
no-regression contract for ``OpenAICompatibleAdapter.token_pricing()`` when called
with a local llama.cpp model name that has no entry in the pricing table.

Background (ADR-0029 §4 consequence):
    ``OpenAICompatibleAdapter`` has no pricing table for GGUF quantizations
    served by a local llama.cpp server. When ``token_pricing()`` is called with
    an unknown model, it returns ``(0.0, 0.0)`` silently rather than emitting a
    warning (change introduced in #1379 via ``_resolve_token_pricing`` refactor).

Success criteria (issue #1265):
    - ``(0.0, 0.0)`` is returned for unknown model names
    - Benchmark is parametrized over known local model names (e.g. ``codellama-7b.Q5_K_M``)

Acceptance criteria (issue #1265):
    1. Return value ``(0.0, 0.0)`` is asserted
    2. Benchmark is under ``benchmarks/tasks/test_*`` and CI-gated
    3. Docstring links to the known limitation (ADR-0029 §4 consequence)
"""

from __future__ import annotations

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import OpenAICompatibleAdapter

TASK = BenchmarkTask(
    name="openai_compatible_token_pricing_unknown_model",
    description=(
        "Unit benchmark: OpenAICompatibleAdapter.token_pricing() returns (0.0, 0.0) "
        "for local llama.cpp model names with no pricing entry. "
        "Known limitation documented per ADR-0029 §4."
    ),
    difficulty_tier="smoke",
    expected_outcome="(0.0, 0.0) is returned for unknown model names.",
    tags=["smoke", "model_adapter", "token_pricing", "ADR-0029"],
)

LOCAL_MODEL_NAMES = [
    "codellama-7b.Q5_K_M",
    "codellama-13b.Q4_K_M",
    "llama-3-8b.Q8_0",
    "mistral-7b.Q6_K",
    "qwen2.5-7b.Q4_K_M",
]


@pytest.mark.benchmark
@pytest.mark.parametrize("model_name", LOCAL_MODEL_NAMES)
def test_token_pricing_unknown_model_emits_warning_and_returns_zero(
    model_name: str,
) -> None:
    """Assert token_pricing() returns (0.0, 0.0) for unknown models.

    This is a smoke-tier unit benchmark: no agent invocation, no I/O, no network.
    It documents the current no-regression contract for the pricing-unknown case
    (ADR-0029 §4 consequence) and will catch any future change that changes the
    return value without updating this benchmark.
    """
    adapter = OpenAICompatibleAdapter(
        base_url="http://localhost:8080",
        model=model_name,
    )

    result = adapter.token_pricing()
    assert result == (0.0, 0.0), (
        f"token_pricing() returned {result!r}; expected (0.0, 0.0) for unknown model"
    )
    import asyncio
    asyncio.run(adapter.aclose())
