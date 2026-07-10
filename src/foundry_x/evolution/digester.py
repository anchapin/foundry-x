from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FailureReport(BaseModel):
    """Structured failure payload emitted by the Digester (ADR-0006).

    ``failed_steps`` carries loosely-typed per-step dicts whose shape varies
    by failure mode; ``dict[str, Any]`` is intentional and noted per ADR-0006.
    """

    session_id: str
    summary: str
    failed_steps: list[dict[str, Any]] = Field(default_factory=list)
    suspected_causes: list[str] = Field(default_factory=list)
    proposed_class: str = "unknown"


class Digester:
    def digest(self, session_id: str, events) -> FailureReport:
        raise NotImplementedError(
            "Phase 2: walk trace events, identify the first failed step, "
            "and classify the failure mode (bad-prompt / wrong-tool / "
            "tool-error / state-leak). Return a FailureReport."
        )
