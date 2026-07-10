"""Pydantic schemas for benchmark task definitions (ADR-0006).

``BenchmarkTask`` is the structured payload that every gatekeeping task
carries. It is deliberately minimal for the initial suite (issue #30);
later work (issue #28) will extend it with scoring options, model
constraints, and serialization helpers as the evolution loop needs them.

Per ADR-0006 this is a pydantic v2 model because the task definition
crosses the boundary between the benchmark suite and the machinery that
serializes / persists it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    """A single deterministic gatekeeping benchmark task."""

    name: str = Field(..., description="Stable, machine-readable task id.")
    description: str = Field(..., description="One-line summary of what the agent must do.")
    prompt: str = Field(default="", description="Natural-language task handed to the agent.")
    tags: list[str] = Field(default_factory=list, description="Free-form grouping labels.")
