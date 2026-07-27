"""Per-config aggregator for the external-eval study (issue #1035, ADR-0023).

This module extracts the aggregation logic that was previously embedded in
``infra/scripts/run_external_eval.sh`` (the inline ``record_config`` Python
snippets).  It queries the trace store for ``critic_verdict`` events grouped
by session metadata (model × quantization × harness variant) and computes
per-configuration internal pass rates, returning paired observations ready
for :func:`~foundry_x.evaluation.correlation.pearson_binary`.

The module is deliberately kept free of I/O side-effects (no file writes,
no network calls) so the unit tests under ``tests/`` can exercise every
code path with a synthetic in-memory trace store.

Design rationale
----------------
ADR-0023 settles on a *per-configuration* methodology: for each agent
configuration *k* (model × quantization × harness variant) we compute
``internal_pass_rate_k = passed / (passed + failed)`` from the
``critic_verdict`` events emitted by the Critic gate.  The external leg
(human or external-eval pass rate) is supplied by the caller — this
module handles only the internal aggregation side.

Session matching
----------------
A session is attributed to a configuration when its ``TraceSession``
metadata contains a ``study_run_type`` key whose value is
``"internal_suite"``.  The configuration label is derived from the
session's ``quantization`` and ``harness_version`` fields.  When both
fields are ``None`` the session is skipped (we cannot attribute it).

Issue #1035 acceptance criterion
---------------------------------
- ``aggregate_per_config`` returns two equal-length float lists.
- Configs with no matching session are skipped (not zero-filled).
- ``aggregate_per_config`` raises ``ValueError`` on empty results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from foundry_x.trace.logger import TraceLogger, TraceSession


@dataclass(frozen=True)
class ConfigObservation:
    """A single per-configuration observation pair.

    Attributes:
        label: Human-readable configuration label
            (e.g. ``"Q5_K_M/v0.9.0"``).
        internal_rate: Fraction of internal tasks passed
            in ``[0.0, 1.0]``.
        external_rate: Fraction of external tasks passed
            in ``[0.0, 1.0]``.  Supplied by the caller.
        passed: Number of internal tasks passed.
        failed: Number of internal tasks failed.
        session_id: The session ID this observation was derived from.
    """

    label: str
    internal_rate: float
    external_rate: float
    passed: int
    failed: int
    session_id: str = ""


@dataclass
class AggregationResult:
    """Result of :func:`aggregate_per_config`.

    Attributes:
        observations: Per-configuration observation pairs.
        labels: Configuration labels in the same order as the rate lists.
        internal_rates: Per-configuration internal pass rates.
        external_rates: Per-configuration external pass rates (caller-supplied).
    """

    observations: list[ConfigObservation] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    internal_rates: list[float] = field(default_factory=list)
    external_rates: list[float] = field(default_factory=list)


def _config_label(session: TraceSession) -> str | None:
    """Derive a configuration label from a TraceSession's metadata.

    Returns ``None`` when the session lacks the required fields to be
    attributed to a configuration.
    """
    quantization = session.quantization
    harness_version = session.harness_version
    if not quantization and not harness_version:
        return None
    parts: list[str] = []
    if quantization:
        parts.append(quantization)
    if harness_version:
        parts.append(harness_version)
    return "/".join(parts)


def _is_internal_suite(session: TraceSession) -> bool:
    """Check if a session is part of the internal evaluation suite."""
    if session.metadata is None:
        return False
    return session.metadata.get("study_run_type") == "internal_suite"


def aggregate_per_config(
    logger: TraceLogger,
    *,
    harness_version: str | None = None,
) -> AggregationResult:
    """Aggregate internal pass rates per configuration from the trace store.

    Queries every session matching ``harness_version`` (when provided),
    filters to those tagged as ``study_run_type == "internal_suite"`` in
    their metadata, and computes the internal pass rate from
    ``critic_verdict`` events.

    Args:
        logger: TraceLogger instance backed by a traces.db or JSONL store.
        harness_version: Optional harness version filter.  When provided,
            only sessions with a matching ``harness_version`` are included.

    Returns:
        :class:`AggregationResult` with paired per-config observations.
        The ``external_rates`` are all ``0.0`` — the caller must replace
        them with externally-scored values before passing to
        :func:`~foundry_x.evaluation.correlation.pearson_binary`.

    Raises:
        ValueError: When no configurations have matching sessions (empty result).
    """
    # Step 1: collect all sessions, keyed by config label.
    sessions = logger.list_sessions(harness_version=harness_version)

    # Group by config label, keeping the most recent internal_suite session
    # per configuration.
    config_sessions: dict[str, TraceSession] = {}
    for session in sessions:
        if not _is_internal_suite(session):
            continue
        label = _config_label(session)
        if label is None:
            continue
        # Keep the most recent session per config.
        if label not in config_sessions or session.started_at > config_sessions[label].started_at:
            config_sessions[label] = session

    if not config_sessions:
        raise ValueError(
            "no configurations with study_run_type='internal_suite' found "
            "in the trace store; ensure at least one internal suite run "
            "has been completed and its session metadata tagged correctly."
        )

    # Step 2: for each config session, query critic_verdict events and
    # compute internal pass rate.
    observations: list[ConfigObservation] = []
    for label, session in sorted(config_sessions.items()):
        passed = failed = 0
        for event in logger.iter_events(session.session_id, kind="critic_verdict"):
            payload = event.payload
            passed += len(payload.get("passed_checks", []))
            failed += len(payload.get("failed_checks", []))

        total = passed + failed
        internal_rate = passed / total if total > 0 else 0.0

        observations.append(
            ConfigObservation(
                label=label,
                internal_rate=internal_rate,
                external_rate=0.0,  # Caller supplies external rates.
                passed=passed,
                failed=failed,
                session_id=session.session_id,
            )
        )

    # Build the return value.
    labels = [obs.label for obs in observations]
    internal_rates = [obs.internal_rate for obs in observations]
    external_rates = [obs.external_rate for obs in observations]

    return AggregationResult(
        observations=observations,
        labels=labels,
        internal_rates=internal_rates,
        external_rates=external_rates,
    )


def apply_external_rates(
    result: AggregationResult,
    external_rates: dict[str, float],
) -> AggregationResult:
    """Replace placeholder external rates with externally-scored values.

    Args:
        result: The aggregation result from :func:`aggregate_per_config`.
        external_rates: Mapping from config label to external pass rate.

    Returns:
        A new :class:`AggregationResult` with ``external_rates`` updated.

    Raises:
        KeyError: When ``external_rates`` is missing a label present in
            ``result.labels``.
    """
    new_external = []
    for label in result.labels:
        if label not in external_rates:
            raise KeyError(
                f"external_rates is missing label '{label}'; "
                f"available labels: {sorted(external_rates)}"
            )
        new_external.append(external_rates[label])

    new_observations = []
    for obs, ext_rate in zip(result.observations, new_external):
        new_observations.append(
            ConfigObservation(
                label=obs.label,
                internal_rate=obs.internal_rate,
                external_rate=ext_rate,
                passed=obs.passed,
                failed=obs.failed,
                session_id=obs.session_id,
            )
        )

    return AggregationResult(
        observations=new_observations,
        labels=list(result.labels),
        internal_rates=list(result.internal_rates),
        external_rates=new_external,
    )


__all__ = [
    "AggregationResult",
    "ConfigObservation",
    "aggregate_per_config",
    "apply_external_rates",
]
