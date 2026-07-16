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
        default="",
        description="Human-readable description of the pass/fail criteria.",
    )
    difficulty_tier: DifficultyTier = Field(
        default="easy",
        description="Tier used to weight the task in the improvement-rate KPI (PRD S5).",
    )
    timeout_seconds: int | None = Field(
        default=None,
        description=(
            "Optional wall-clock cap (seconds) for the Runner. ``None`` means no limit is enforced."
        ),
    )
    token_budget: int | None = Field(
        default=None,
        description=(
            "Optional total-token cap for the Runner. When set, the Runner "
            "aborts the task with ``task_aborted(reason='token_budget')`` if "
            "the agent exceeds this many tokens in its total completion + "
            "prompt tokens. ``None`` means no token limit is enforced. "
            "Enforced via the ``FOUNDRY_TOKEN_BUDGET`` environment variable "
            "wired through the Critic gate (issue #548)."
        ),
    )
    requires_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Names of harness skills (``harness/skills/<name>.json``) the agent "
            "path must have available to attempt this task. The Critic uses this "
            "list to flag a benchmark as 'not yet evaluable' when a required "
            "skill is absent, instead of recording a spurious fail. Empty list "
            "means the task does not require any named skill (e.g. tasks that "
            "are satisfied by ``read_file``/``write_file`` alone). First non-empty "
            "entry as of issue #104 is ``bash``."
        ),
    )

    # --- Grouping ---------------------------------------------------------
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form grouping labels for selection / reporting.",
    )

    # --- Model contract ---------------------------------------------------
    model_requirements: ModelRequirements | None = Field(
        default=None,
        description=(
            "Model identity fields for this task. When set, the Runner uses "
            "these values instead of the environment-derived defaults for this "
            "task only. Added under issue #363 / ADR-0014 for the model-swapping "
            "milestone."
        ),
    )

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

    @field_validator("token_budget")
    @classmethod
    def _token_budget_positive(cls, value: int | None) -> int | None:
        """A non-positive token cap is nonsensical; surface it at validation time."""
        if value is not None and value <= 0:
            raise ValueError("token_budget must be a positive integer")
        return value
