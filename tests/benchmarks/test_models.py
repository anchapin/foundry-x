"""Unit tests for the ``BenchmarkTask`` pydantic schema (ADR-0006, issue #28).

These pin the schema contract shared across the benchmark suite, the Runner,
and the Critic: construction, defaulting, validation, and JSON round-trip
serialization. They must stay green for the regression gate (ADR-0004) to be
non-vacuous.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmarks.models import BenchmarkTask


def test_minimal_construction_backwards_compatible() -> None:
    """A task built with only name + description (issue #30 shape) still works."""
    task = BenchmarkTask(name="sort_a_list", description="Sort integers ascending.")
    assert task.name == "sort_a_list"
    assert task.description == "Sort integers ascending."
    # New fields all take documented defaults (issue #417: defaults prevent open-ended runs).
    assert task.prompt == ""
    assert task.setup_commands == []
    assert task.expected_outcome == ""
    assert task.difficulty_tier == "easy"
    assert task.timeout_seconds == 300
    assert task.token_budget == 50000
    assert task.tags == []


def test_full_construction_populates_every_field() -> None:
    """All issue-#28 fields can be set and round-trip through the model."""
    task = BenchmarkTask(
        name="deploy_service",
        description="Stand up the service and verify it answers.",
        prompt="Deploy the service and curl its health endpoint.",
        setup_commands=["make build", "cp -r fixtures/deploy_service/. ."],
        expected_outcome="GET /healthz returns 200 within 5s.",
        difficulty_tier="medium",
        timeout_seconds=120,
        tags=["deploy", "networking"],
    )
    assert task.setup_commands == ["make build", "cp -r fixtures/deploy_service/. ."]
    assert task.expected_outcome == "GET /healthz returns 200 within 5s."
    assert task.difficulty_tier == "medium"
    assert task.timeout_seconds == 120
    assert task.tags == ["deploy", "networking"]


@pytest.mark.parametrize("tier", ["smoke", "easy", "medium"])
def test_difficulty_tier_accepts_documented_values(tier: str) -> None:
    task = BenchmarkTask(name="t", description="d", difficulty_tier=tier)
    assert task.difficulty_tier == tier


def test_difficulty_tier_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError) as exc_info:
        BenchmarkTask(name="t", description="d", difficulty_tier="hard")
    # Pydantic surfaces the literal constraint in the error.
    assert "difficulty_tier" in str(exc_info.value)


def test_name_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="name must be a non-empty string"):
        BenchmarkTask(name="", description="d")
    with pytest.raises(ValidationError, match="name must be a non-empty string"):
        BenchmarkTask(name="   ", description="d")


@pytest.mark.parametrize("bad_timeout", [0, -1, -60])
def test_timeout_seconds_must_be_positive(bad_timeout: int) -> None:
    with pytest.raises(ValidationError, match="timeout_seconds must be a positive integer"):
        BenchmarkTask(name="t", description="d", timeout_seconds=bad_timeout)


@pytest.mark.parametrize("bad_budget", [0, -1, -5000])
def test_token_budget_must_be_positive(bad_budget: int) -> None:
    with pytest.raises(ValidationError, match="token_budget must be a positive integer"):
        BenchmarkTask(name="t", description="d", token_budget=bad_budget)


def test_round_trip_serialization() -> None:
    """model_dump_json -> model_validate_json yields an equal task (ADR-0006)."""
    original = BenchmarkTask(
        name="reverse_string",
        description="Reverse the characters of the input string.",
        prompt="Read a line from input.txt, reverse it, write to output.txt.",
        setup_commands=["printf 'hello' > input.txt"],
        expected_outcome="output.txt contains 'olleh'.",
        difficulty_tier="easy",
        timeout_seconds=30,
        tags=["strings"],
    )
    restored = BenchmarkTask.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.model_dump() == original.model_dump()


def test_round_trip_preserves_defaults_for_minimal_task() -> None:
    """Defaults survive a JSON round-trip so absent fields rehydrate correctly."""
    original = BenchmarkTask(name="t", description="d")
    restored = BenchmarkTask.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.setup_commands == []
    assert restored.timeout_seconds == 300
    assert restored.token_budget == 50000
