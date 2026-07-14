"""Pydantic schemas for benchmark task definitions (ADR-0006).

``BenchmarkTask`` is the structured payload that every gatekeeping task
carries. It is the data contract shared across the benchmark suite, the
Runner (which executes the prompt), and the Critic (which evaluates the
outcome) -- see ADR-0004 / ADR-0005.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


DifficultyTier = Literal["smoke", "easy", "medium"]


class ModelRequirements(BaseModel):
    """Model identity fields for model-swapping milestone (issue #363, ADR-0014)."""

    model_id: str | None = Field(default=None)
    quantization: str | None = Field(default=None)
    path_or_endpoint: Path | str | None = Field(default=None)


class BenchmarkTask(BaseModel):
    """A single deterministic gatekeeping benchmark task."""

    name: str = Field(..., description="Stable, machine-readable task id (snake_case).")
    description: str = Field(..., description="One-line summary of what the agent must do.")
    prompt: str = Field(
        default="", description="Natural-language task handed to the agent under test."
    )
    setup_commands: list[str] = Field(default_factory=list)
    expected_outcome: str = Field(
        default="", description="Human-readable description of the pass/fail criteria."
    )
    difficulty_tier: DifficultyTier = Field(default="easy")
    timeout_seconds: int | None = Field(default=None)
    requires_skills: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    model_requirements: ModelRequirements | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be a non-empty string")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        return value
