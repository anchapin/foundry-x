from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProposedEdit(BaseModel):
    """A single targeted harness edit proposed by the Evolver (ADR-0006).

    The three string fields are required and non-blank so a malformed edit
    surfaces a ``ValidationError`` at the boundary instead of reaching the
    Critic and wasting a gate run.
    """

    target_file: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    unified_diff: str = Field(min_length=1)


class Evolver:
    def propose(
        self,
        harness_dir: Path,
        failure,
        current_diff: str | None = None,
    ) -> list[ProposedEdit]:
        raise NotImplementedError(
            "Phase 2: meta-agent takes a FailureReport plus the harness "
            "tree, returns one or more ProposedEdit objects describing "
            "targeted edits to system_prompt.txt / hooks / skills."
        )
