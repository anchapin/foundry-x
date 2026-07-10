from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProposedEdit:
    target_file: str
    rationale: str
    unified_diff: str


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
