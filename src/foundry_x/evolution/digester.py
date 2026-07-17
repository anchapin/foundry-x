"""Deterministic trace-walking and failure classification (issue #15, Phase 2).

The ``Digester`` is the first link of the evolution loop
(``Runner`` -> trace -> ``Digester`` -> ``Evolver`` -> ``Critic``). It turns
an ordered list of :class:`~foundry_x.trace.logger.TraceEvent` objects into a
:class:`FailureReport` without ever calling an LLM: the report is derived
purely from trace content, satisfying ADR-0007 ("trace-driven development is
the default" — failure reports come from observed traces, not speculation).

Classification is keyword-driven by design (issue #15). The vocabulary of
failure-signalling event ``kind`` values is exposed as module constants so the
trace subsystem can align its emitted kinds against them in a later phase.

Issue #120 adds a structured ``injection_blocked`` event kind emitted by the
``InjectionFirewallHook`` whenever it suppresses a tool result. The Digester
recognizes that kind with a dedicated aggregation pass (one entry per block
in ``failed_steps``) before the generic first-failure walk runs, so an
adversarial tool result is surfaced as ``proposed_class='injection-attempt'``
even when a downstream ``tool_error`` event also occurs.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field

from foundry_x.trace.logger import TraceEvent


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


# --- Failure-signalling vocabulary -----------------------------------------
# Module-level constants (issue #15): the set of ``kind`` values that
# unambiguously mark a trace event as a failure, plus the payload keys that
# signal a failure even when the ``kind`` is benign (e.g. a ``tool_result``
# carrying a traceback). The trace subsystem (``src/foundry_x/trace/``) can
# align its emitted kinds against ``FAILURE_KINDS`` in a later phase.
#
# ``injection_blocked`` (issue #120) is *not* in ``FAILURE_KINDS``: it is
# handled by a dedicated aggregation pass in :meth:`Digester.digest` that
# collects *every* block in the session (not just the first), so the
# generic first-failure walk would under-report. The constant is exported
# separately so tests can pin the contract.
INJECTION_BLOCKED_KIND: str = "injection_blocked"
INJECTION_ATTEMPT_CLASS: str = "injection-attempt"
CONTEXT_OVERFLOW_CLASS: str = "context-overflow"

FAILURE_KINDS: frozenset[str] = frozenset(
    {
        "tool_error",
        "task_failed",
        "run_failed",
        "agent_error",
        "error",
    }
)

FAILURE_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "error",
        "traceback",
        "exception",
    }
)

# Keyword rules driving deterministic classification. ``proposed_class`` is
# always one of the four named classes (issue #15). The classes are checked
# in this fixed order and the first match wins, so:
#   * ``wrong-tool`` (specific multi-word tool-selection phrases) beats the
#     generic ``tool-error`` catch-all.
#   * ``bad-prompt`` and ``state-leak`` (specific phrases) likewise beat the
#     catch-all.
#   * ``tool-error`` is intentionally last: it mops up any structural
#     failure signal (``kind`` / payload key) that no specific keyword
#     matched, e.g. an opaque ``{"error": "exit code 1"}``.
#
# Keyword tuples are ordered most-specific-first so the recorded
# ``suspected_causes`` evidence string is deterministic (not hash-ordered).
# Keyword rules are imperfect by design (issue #15 scope); a traceback that
# happens to contain the word "ambiguous" would mis-classify as ``bad-prompt``.
_CLASS_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "wrong-tool",
        (
            "no such tool",
            "tool not found",
            "unknown tool",
            "invalid tool",
            "no tool named",
            "no command named",
            "unknown function",
            "is not a valid tool",
        ),
    ),
    (
        "bad-prompt",
        (
            "missing context",
            "under-specified",
            "underspecified",
            "malformed prompt",
            "cannot parse prompt",
            "contradictory",
            "ambiguous",
            "vague",
            "unparseable",
        ),
    ),
    (
        "state-leak",
        (
            "no such file",
            "file not found",
            "already exists",
            "unexpected state",
            "race condition",
            "dirty tree",
            "is not empty",
            "leftover",
            "stale",
            "leak",
        ),
    ),
    (
        "tool-error",
        (
            "traceback",
            "exception",
            "timed out",
            "timeout",
            "exit code",
            "segmentation fault",
            "segfault",
            "broken pipe",
            "command not found",
            "error",
            "failed",
        ),
    ),
)

# Cause templates per class. The matched keyword interpolates as evidence
# (ADR-0007): every suspected cause points back at trace content.
_CLASS_CAUSE_TEMPLATES: dict[str, str] = {
    "wrong-tool": (
        "Agent invoked a tool/command that is not registered (matched: "
        "{match}). Review the available-tool list in the prompt."
    ),
    "bad-prompt": (
        "Task prompt appears ambiguous or under-specified (matched: {match}). "
        "Tighten the prompt with concrete acceptance criteria."
    ),
    "state-leak": (
        "Execution hit stale or leaked state (matched: {match}). Check sandbox "
        "reset/cleanup between steps."
    ),
    "tool-error": (
        "Tool execution raised an error (matched: {match}). Inspect the failing call's traceback."
    ),
    # Issue #120: structured ``injection_blocked`` events aggregate into a
    # single ``injection-attempt`` report. The template references the first
    # block's marker list as evidence; later blocks are listed in
    # ``failed_steps`` for full traceability.
    "injection-attempt": (
        "InjectionFirewallHook suppressed {count} tool result(s) for prompt-"
        "injection markers (first block matched: {match}). Treat the agent "
        "as compromised for this session; consider tightening the firewall "
        "patterns or the upstream tool-result scrubbing policy."
    ),
    # Issue #805: context-overflow triggered when the runner agent loop terminates
    # via outcome.status=truncated / outcome.reason=max_steps. The steps value
    # is extracted from the payload so the Evolver can see how many steps ran.
    "context-overflow": (
        "Agent loop reached max_steps (steps={steps}) before producing a final "
        "answer; the context budget was exhausted. Review the pruning hook and "
        "the model's tendency to repeat tool calls."
    ),
}


def _is_failure(event: TraceEvent) -> tuple[bool, str]:
    """Return ``(is_failure, signal)`` for one event.

    ``signal`` describes *what* tripped the detector so the report can carry
    evidence (ADR-0007); it is empty when no failure is present.
    """
    if event.kind in FAILURE_KINDS:
        return True, f"kind:{event.kind}"
    for key in FAILURE_PAYLOAD_KEYS:
        if key in event.payload:
            return True, f"payload_key:{key}"
    return False, ""


def _walk_strings(value: Any) -> Iterable[str]:
    """Yield every string within a nested payload structure.

    Payload keys are yielded too (a key named ``timeout`` is a signal). This
    is the single place the free-form ``Any`` payload is consumed; it is the
    serialization-boundary carve-out described in ADR-0006.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, sub in value.items():
            if isinstance(key, str):
                yield key
            yield from _walk_strings(sub)
    elif isinstance(value, (list, tuple, set)):
        for sub in value:
            yield from _walk_strings(sub)


def _flatten_text(event: TraceEvent) -> str:
    """Lowercase blob of the kind plus every string in the payload."""
    return " ".join((event.kind, *_walk_strings(event.payload))).lower()


def _classify(event: TraceEvent, signal: str) -> tuple[str, list[str]]:
    """Map a failing event to ``(proposed_class, suspected_causes)``.

    ``proposed_class`` is always one of the four named classes. The
    most-specific keyword match wins (see ``_CLASS_KEYWORDS`` precedence);
    when only a structural signal (``kind`` / payload key) is present with no
    recognisable keyword, the class defaults to ``tool-error``.
    """
    text = _flatten_text(event)
    matched_class = ""
    matched_keyword = ""
    for cls, keywords in _CLASS_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                matched_class = cls
                matched_keyword = keyword
                break
        if matched_class:
            break

    if not matched_class:
        matched_class = "tool-error"
        matched_keyword = signal

    causes = [_CLASS_CAUSE_TEMPLATES[matched_class].format(match=matched_keyword)]
    if signal.startswith("payload_key:"):
        causes.append(f"Failing payload key present: {signal.split(':', 1)[1]}.")
    return matched_class, causes


def _detail(event: TraceEvent) -> str:
    """Best-effort one-line excerpt of the failing event's payload."""
    for key in FAILURE_PAYLOAD_KEYS:
        value = event.payload.get(key)
        if isinstance(value, str) and value.strip():
            line = value.strip().splitlines()[0].strip()
            if line:
                return line[:200]
    return _flatten_text(event)[:200]


def _summarise(event: TraceEvent, signal: str, proposed_class: str) -> str:
    detail = _detail(event)
    head = f"{proposed_class} failure (kind={event.kind}, signal={signal})"
    if detail:
        head += f": {detail}"
    return head


def _aggregate_injection_blocks(
    session_id: str,
    ordered: Sequence[TraceEvent],
) -> FailureReport | None:
    """Aggregate every ``injection_blocked`` event in ``ordered`` (issue #120).

    Returns a fully-formed :class:`FailureReport` with
    ``proposed_class == 'injection-attempt'`` and one entry per block in
    ``failed_steps`` so the Evolver can see the full extent of the
    adversarial surface. The summary mentions the block count (per the
    issue's "block count in ``failed_steps``" acceptance criterion) and
    the first block's marker list is used as the cause's ``{match}``
    placeholder so the report stays grounded in trace content (ADR-0007).

    Returns ``None`` when no ``injection_blocked`` event is present, so
    the caller can fall through to the generic first-failure walk.
    """
    blocks = [(i, e) for i, e in enumerate(ordered) if e.kind == INJECTION_BLOCKED_KIND]
    if not blocks:
        return None

    first_event = blocks[0][1]
    # Prefer the structured ``markers`` list from the firewall payload; fall
    # back to the full flattened text so a malformed payload (missing the
    # ``markers`` key, e.g. from a hand-crafted trace) still produces a
    # useful ``{match}`` snippet.
    first_markers = first_event.payload.get("markers")
    if isinstance(first_markers, list) and first_markers:
        match_repr = ",".join(str(m) for m in first_markers)
    else:
        match_repr = _flatten_text(first_event)[:80]

    failed_steps: list[dict[str, Any]] = [
        {
            "index": index,
            "event_id": event.event_id,
            "kind": event.kind,
            "timestamp": event.timestamp,
            "signal": f"kind:{INJECTION_BLOCKED_KIND}",
            "payload": event.payload,
        }
        for index, event in blocks
    ]
    causes = [
        _CLASS_CAUSE_TEMPLATES[INJECTION_ATTEMPT_CLASS].format(
            match=match_repr,
            count=len(blocks),
        )
    ]
    return FailureReport(
        session_id=session_id,
        summary=(
            f"{INJECTION_ATTEMPT_CLASS} failure: {len(blocks)} firewall "
            f"block(s) across {len(ordered)} trace event(s)"
        ),
        failed_steps=failed_steps,
        suspected_causes=causes,
        proposed_class=INJECTION_ATTEMPT_CLASS,
    )


def _aggregate_context_overflow(
    session_id: str,
    ordered: Sequence[TraceEvent],
) -> FailureReport | None:
    """Aggregate a context-overflow failure (issue #805).

    Triggered when the runner agent loop terminates via
    ``outcome.status='truncated'`` / ``outcome.reason='max_steps'``
    (ADR-0010 §Termination semantics). This is a terminal condition: the
    session ended because the context budget was exhausted before the agent
    produced a final answer. The Evolver should propose a pruning-hook
    adjustment or prompt the model to avoid repetitive tool-call loops.

    Returns ``None`` when no outcome event with the trigger payload is
    present, so the caller can fall through to subsequent checks.
    """
    for i, event in enumerate(ordered):
        if event.kind == "outcome":
            status = event.payload.get("status")
            reason = event.payload.get("reason")
            if status == "truncated" and reason == "max_steps":
                steps = event.payload.get("steps", "?")
                failed_steps: list[dict[str, Any]] = [
                    {
                        "index": i,
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "timestamp": event.timestamp,
                        "signal": "outcome:truncated/max_steps",
                        "payload": event.payload,
                    }
                ]
                causes = [_CLASS_CAUSE_TEMPLATES[CONTEXT_OVERFLOW_CLASS].format(steps=steps)]
                return FailureReport(
                    session_id=session_id,
                    summary=(
                        f"{CONTEXT_OVERFLOW_CLASS} failure: agent loop reached "
                        f"max_steps ({steps}) before producing a final answer"
                    ),
                    failed_steps=failed_steps,
                    suspected_causes=causes,
                    proposed_class=CONTEXT_OVERFLOW_CLASS,
                )
    return None


class Digester:
    def digest(
        self,
        session_id: str,
        events: Sequence[TraceEvent],
    ) -> FailureReport:
        """Walk trace events and classify the first failure (issue #15).

        Pure and deterministic: no LLM call, no I/O. Events are stably sorted
        by ``timestamp`` so the "first" failed step is well-defined regardless
        of how the caller assembled the sequence (mirrors the ordering contract
        of ``TraceLogger.load_session``). When no failure is found the report
        is returned with ``proposed_class == "clean"`` and empty
        ``failed_steps``.

        Issue #805 short-circuits when an ``outcome`` event signals
        ``status='truncated'`` / ``reason='max_steps'``: this terminal
        context-overflow takes precedence over all other aggregation passes.

        Issue #120 short-circuits the generic walk when one or more
        ``injection_blocked`` events are present: every block is aggregated
        into a single ``injection-attempt`` report so the Evolver sees the
        full adversarial surface rather than just the first block.
        """
        ordered = sorted(events, key=lambda e: e.timestamp)

        overflow_report = _aggregate_context_overflow(session_id, ordered)
        if overflow_report is not None:
            return overflow_report

        injection_report = _aggregate_injection_blocks(session_id, ordered)
        if injection_report is not None:
            return injection_report

        for index, event in enumerate(ordered):
            is_failure, signal = _is_failure(event)
            if not is_failure:
                continue
            proposed_class, causes = _classify(event, signal)
            failed_step: dict[str, Any] = {
                "index": index,
                "event_id": event.event_id,
                "kind": event.kind,
                "timestamp": event.timestamp,
                "signal": signal,
                "payload": event.payload,
            }
            return FailureReport(
                session_id=session_id,
                summary=_summarise(event, signal, proposed_class),
                failed_steps=[failed_step],
                suspected_causes=causes,
                proposed_class=proposed_class,
            )

        return FailureReport(
            session_id=session_id,
            summary=(f"No failures detected across {len(ordered)} trace event(s)."),
            failed_steps=[],
            suspected_causes=[],
            proposed_class="clean",
        )
