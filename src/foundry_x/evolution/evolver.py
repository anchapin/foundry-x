from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import BaseModel, Field

# Sliding window for the rate limiter. One hour matches the SECURITY.md
# "max N proposals per hour" guardrail.
_RATE_WINDOW = timedelta(hours=1)


class ProposedEdit(BaseModel):
    """A single targeted harness edit proposed by the Evolver (ADR-0006).

    The three string fields are required and non-blank so a malformed edit
    surfaces a ``ValidationError`` at the boundary instead of reaching the
    Critic and wasting a gate run.
    """

    target_file: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    unified_diff: str = Field(min_length=1)


class EvolverGuardError(ValueError):
    """Raised when an edit or proposal cadence violates a guardrail.

    Implements the SECURITY.md "Rate limits" guardrail: max N proposals per
    hour, max M lines of harness diff per proposal. Violations are raised,
    never swallowed, per AGENTS.md §4 ("no swallowed exceptions"). Surfacing
    the failure is the "surface it or re-raise" option in that rule; a
    TraceLogger event will be attached once the propose() body lands and a
    session context is available.
    """


class Evolver:
    """Proposes harness edits, bounded by the SECURITY.md rate/diff guardrails.

    The guardrails are enforced *before* any proposal work happens so a
    runaway loop cannot emit unbounded edits before the Critic catches them
    (PHILOSOPHY.md §2 — reversibility by default). Defaults (10 proposals /
    hour, 200 diff lines) mirror the SECURITY.md prose and are configurable
    via the constructor.
    """

    def __init__(
        self,
        max_proposals_per_hour: int = 10,
        max_diff_lines: int = 200,
    ) -> None:
        if max_proposals_per_hour < 1:
            raise EvolverGuardError("max_proposals_per_hour must be >= 1")
        if max_diff_lines < 1:
            raise EvolverGuardError("max_diff_lines must be >= 1")
        self.max_proposals_per_hour = max_proposals_per_hour
        self.max_diff_lines = max_diff_lines
        self._proposal_times: deque[datetime] = deque()

    def _purge_old(self, now: datetime | None = None) -> None:
        """Drop proposal timestamps that have fallen outside the rate window."""
        cutoff = (now or datetime.now(timezone.utc)) - _RATE_WINDOW
        while self._proposal_times and self._proposal_times[0] < cutoff:
            self._proposal_times.popleft()

    def _check_rate_limit(self) -> None:
        """Raise if the number of recent proposals is already at the cap."""
        self._purge_old()
        if len(self._proposal_times) >= self.max_proposals_per_hour:
            raise EvolverGuardError(
                f"rate limit exceeded: {len(self._proposal_times)} proposals in "
                f"the last hour (cap={self.max_proposals_per_hour})"
            )

    def _record_proposals(self, count: int = 1) -> None:
        """Stamp ``count`` proposals at the current time into the window."""
        now = datetime.now(timezone.utc)
        for _ in range(count):
            self._proposal_times.append(now)

    def _validate_edit(self, edit: ProposedEdit) -> None:
        """Reject an edit whose unified diff exceeds the line cap."""
        line_count = len(edit.unified_diff.splitlines())
        if line_count > self.max_diff_lines:
            raise EvolverGuardError(
                f"diff too large: {line_count} lines for {edit.target_file} "
                f"(cap={self.max_diff_lines})"
            )

    def propose(
        self,
        harness_dir: Path,
        failure,
        current_diff: str | None = None,
    ) -> list[ProposedEdit]:
        # Guardrails first: a runaway caller must hit the cap before any
        # proposal work begins (SECURITY.md "Rate limits").
        self._check_rate_limit()
        raise NotImplementedError(
            "Phase 2: meta-agent takes a FailureReport plus the harness "
            "tree, returns one or more ProposedEdit objects describing "
            "targeted edits to system_prompt.txt / hooks / skills."
        )
