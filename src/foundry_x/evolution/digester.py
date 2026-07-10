from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FailureReport:
    session_id: str
    summary: str
    failed_steps: list[dict] = field(default_factory=list)
    suspected_causes: list[str] = field(default_factory=list)
    proposed_class: str = "unknown"


class Digester:
    def digest(self, session_id: str, events) -> FailureReport:
        raise NotImplementedError(
            "Phase 2: walk trace events, identify the first failed step, "
            "and classify the failure mode (bad-prompt / wrong-tool / "
            "tool-error / state-leak). Return a FailureReport."
        )
