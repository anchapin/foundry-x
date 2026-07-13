"""Pydantic schemas for benchmark task definitions (ADR-0006).

``BenchmarkTask`` is the structured payload that every gatekeeping task
carries. It is the data contract shared across the benchmark suite, the
Runner (which executes the prompt), and the Critic (which evaluates the
outcome) -- see ADR-0004 / ADR-0005.

Per ADR-0006 this is a pydantic v2 model because the task definition
crosses the boundary between the benchmark suite and the machinery that
serializes / persists it.

Field groups:

- **Identity:** ``name``, ``description`` -- uniquely identify the task.
- **Agent contract:** ``prompt``, ``setup_commands`` -- what the agent
  receives and how its workspace is seeded.
- **Evaluation contract:** ``expected_outcome``, ``difficulty_tier``,
  ``timeout_seconds`` -- how the Critic weights and bounds the run.
- **Harness contract:** ``requires_skills`` -- the harness skills (by name)
  the agent path must have available to attempt this task. Lets the Critic
  flag a benchmark whose required skill is missing from the harness as
  "not yet evaluable" instead of a spurious fail (issue #104, ADR-0004).
- **Grouping:** ``tags`` -- free-form labels for selection / reporting.
- **Model contract:** ``model_requirements`` -- model identity fields that
  let the Runner override its environment-derived defaults for a specific
  task (issue #363, ADR-0014).

All fields beyond ``name`` and ``description`` are optional with sane
defaults, so the minimal task shape authored under issue #30 keeps working
(backwards compatible). New fields were added under issue #28 to give the
Runner and Critic the structured contract ADR-0006 calls for. The
``requires_skills`` field was added under issue #104 alongside the
seeding of the ``bash`` skill.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

#: Difficulty tiers, ordered low -> high. ``smoke`` is a cheap sanity check,
#: ``easy`` is the default gatekeeping weight, ``medium`` exercises a
#: multi-step capability. Adding a higher tier (e.g. ``hard``) requires an
#: ADR because it changes the improvement-rate KPI weighting.
DifficultyTier = Literal["smoke", "easy", "medium"]


class ModelRequirements(BaseModel):
    """Model identity fields for model-swapping milestone (issue #363, ADR-0014).

    Three orthogonal fields identify every model variant currently supported
    (local llama.cpp GGUF, OpenAI-compatible remote) without breaking the
    schema for future variants (custom GPTQ, vision models).
    """

    model_id: str | None = Field(
        default=None,
        description="Stable machine-readable identifier sent in the API request body.",
    )
    quantization: str | None = Field(
        default=None,
        description="Quantization label from the filename, e.g. Q5_K_M.",
    )
    path_or_endpoint: Path | str | None = Field(
        default=None,
        description="Local GGUF path or remote URL.",
    )


class BenchmarkTask(BaseModel):
    """A single deterministic gatekeeping benchmark task."""

    # --- Identity ---------------------------------------------------------
    name: str = Field(
        ...,
        description="Stable, machine-readable task id (snake_case).",
    )
    description: str = Field(
        ...,
        description="One-line summary of what the agent must do.",
    )

    # --- Agent contract ---------------------------------------------------
    prompt: str = Field(
        default="",
        description="Natural-language task handed to the agent under test.",
    )
    setup_commands: list[str] = Field(
        default_factory=list,
        description=(
            "Shell commands that seed the workspace before the agent runs. "
            "Executed by the Runner in order; must be deterministic and network-free."
        ),
    )

    # --- Evaluation contract ----------------------------------------------
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
        """Reject blank ids -- a task without a name cannot be referenced."""
        if not value or not value.strip():
            raise ValueError("name must be a non-empty string")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _timeout_positive(cls, value: int | None) -> int | None:
        """A non-positive cap is nonsensical; surface it at validation time."""
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        return value
