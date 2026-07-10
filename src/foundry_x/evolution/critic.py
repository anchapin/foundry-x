from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class CriticVerdict(BaseModel):
    """Result of a Critic gate run against a proposed harness edit (ADR-0006)."""

    approved: bool
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""


class Critic:
    def __init__(
        self,
        harness_dir: Path,
        benchmark_path: Path | None = None,
        pytest_args: list[str] | None = None,
    ) -> None:
        self.harness_dir = harness_dir
        self.benchmark_path = benchmark_path
        self.pytest_args = pytest_args or ["-q", "tests/test_smoke.py"]

    def evaluate(self, proposed_diff: str) -> CriticVerdict:
        raise NotImplementedError(
            "Phase 2: apply the proposed_diff against a sandbox copy of the "
            "harness, run pytest, run a benchmark subset, then return a "
            "CriticVerdict. Never gatekeep against the live harness tree."
        )
