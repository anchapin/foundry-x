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

DifficultyTier = Literal["smoke", "easy", "medium", "hard"]


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


class ContextPruningSweepResult(BaseModel):
    """Structured result of a single context-pruning sweep run (issue #956).

    Produced by ``test_context_pruning_sweep.py`` for KPI-layer consumption.
    The sweep runs the benchmark suite at three ``FOUNDRY_CONTEXT_TOKENS``
    thresholds (4096, 8192, 16384) and records pass/fail per threshold
    plus the ``context_pruned`` event count per session, enabling the
    ``context_efficiency`` KPI computation defined in ADR-0021 §6.

    Attributes
    ----------
    threshold:
        The ``FOUNDRY_CONTEXT_TOKENS`` value for this run.
    passed:
        Whether the benchmark task completed successfully at this threshold.
    context_pruned_count:
        Number of ``context_pruned`` events emitted during the session.
        Used as the numerator in ``context_efficiency = 1 - (dropped /
        total_events_in_session)`` (ADR-0021 §6).
    token_budget_hit:
        Whether the session hit the ``FOUNDRY_TOKEN_BUDGET`` abort threshold.
    dropped_total:
        Sum of the ``dropped`` field across all ``context_pruned`` events
        in the session.
    total_events:
        Total number of events recorded in the session.
    """

    threshold: int = Field(..., description="FOUNDRY_CONTEXT_TOKENS value for this run.")
    passed: bool = Field(..., description="Whether the session outcome.status was 'success'.")
    context_pruned_count: int = Field(
        default=0,
        description="Number of context_pruned events emitted during the session.",
    )
    token_budget_hit: bool = Field(
        default=False,
        description="Whether the session was aborted due to token_budget.",
    )
    dropped_total: int = Field(
        default=0,
        description="Sum of 'dropped' across all context_pruned events.",
    )
    total_events: int = Field(
        default=0,
        description="Total events recorded in the session.",
    )


class V4QuantizationSweepResult(BaseModel):
    """Structured result of a GGUF v4 quantization sweep (issue #1078).

    Produced by ``test_quantization_v4_sweep.py`` for KPI-layer consumption.
    The sweep runs the benchmark suite across v4 IQ quantizations (IQ4_XS,
    IQ3_S, IQ3_XXS, IQ2_XXS) plus Q2_K and the v3 K-quant baselines
    (Q4_K_M, Q5_K_M, Q8_0) to validate whether any v4 IQ quantization
    offers a better quality/VRAM tradeoff than Q5_K_M (the current
    recommended floor at ~5.5 GB).

    The key question is whether IQ4_XS (~4.1 GB) matches Q5_K_M quality
    within 2 pp, which would allow 8 GB card operators to run a
    higher-quality model per VRAM dollar.

    Attributes
    ----------
    quantization:
        The GGUF quantization label (e.g. ``"IQ4_XS"``, ``"Q5_K_M"``).
    pass_rate:
        Fraction of benchmark tasks that passed (0.0 to 1.0).
    vrams_gb:
        Estimated VRAM usage in GB for a 7B model at this quantization.
    regression_vs_baseline:
        Pass rate delta vs. Q8_0 baseline in percentage points.
        Negative means lower quality than baseline.
    within_2pp_of_q5km:
        Whether this quantization's pass rate is within 2 pp of Q5_K_M.
        This is the acceptance criterion for v4 IQ quants to be considered
        competitive with the current recommended floor.
    recommended:
        Whether this quantization is recommended as the new floor if
        regression_vs_baseline is within threshold and VRAM is lower.
    notes:
        Free-form annotation for any special circumstances.
    """

    quantization: str = Field(..., description="GGUF quantization label (e.g. IQ4_XS).")
    pass_rate: float = Field(
        default=0.0,
        description="Benchmark pass rate (0.0 to 1.0).",
    )
    vrams_gb: float = Field(
        default=0.0,
        description="Estimated VRAM usage in GB for a 7B model.",
    )
    regression_vs_baseline: float = Field(
        default=0.0,
        description="Pass rate delta vs. Q8_0 baseline in percentage points.",
    )
    within_2pp_of_q5km: bool = Field(
        default=False,
        description="Whether pass rate is within 2 pp of Q5_K_M.",
    )
    recommended: bool = Field(
        default=False,
        description="Whether this quantization is recommended as the new floor.",
    )
    notes: str = Field(
        default="",
        description="Free-form annotation.",
    )
