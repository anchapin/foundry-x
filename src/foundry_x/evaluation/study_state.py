"""Incremental batched-execution state for the external-eval study (issue #1040).

The real-model external-eval study (ADR-0023, issue #900) requires
running >= 30 agent configurations, each costing a live ``fx-runner``
invocation pair against a llama.cpp endpoint. Running all of them in a
single ``infra/scripts/run_external_eval.sh`` invocation is fragile: a
GPU crash or operator interrupt at configuration 15 of 30 wipes all
progress. Issue #1040 asks the orchestrator to checkpoint completed
configurations to a state file so an interrupted run can resume without
re-running finished configs.

Why a pure-Python module
------------------------
``run_external_eval.sh`` is bash that already delegates its heavy logic
to inline Python (slice validation, traces.db aggregation, Pearson
math — see ADR-0023). Keeping the state-management logic in this module
rather than in shell arithmetic follows that same split and — critically
— makes it unit-testable without a live model endpoint, a llama-server,
or even a real ``logs/traces.db``. The bash script loads, mutates, and
persists an :class:`ExternalEvalState` by shelling out to small
``python -c`` snippets backed by the functions below.

State file contract
-------------------
The state file (default ``logs/.run_external_eval_state.json``, overridable
via the script's ``--state-file``) is human-readable JSON so an operator
can audit progress without running the script:

.. code-block:: json

    {
      "study_id": "external_eval_20260727T120000Z",
      "model": "/srv/models/qwen.Q5_K_M.gguf",
      "slice": "/repo/benchmarks/external/humaneval_plus_sample.jsonl",
      "min_pairs": 30,
      "created_at": "2026-07-27T12:00:00Z",
      "updated_at": "2026-07-27T12:42:13Z",
      "configs_planned": ["q4km", "q5km", "..."],
      "configs_completed": ["q4km"],
      "completed_at": {"q4km": "2026-07-27T12:05:11Z"},
      "internal_rates": [0.75],
      "external_rates": [0.5]
    }

``configs_completed`` and the two rate arrays are kept the same length
and in the same order, enforced by the pydantic model's validators
(ADR-0006). A torn write is detected by catching ``ValidationError`` on
load and refusing to resume a corrupt checkpoint rather than silently
re-running configs or dropping data.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from foundry_x.evaluation.correlation import MIN_PAIRED_OBSERVATIONS


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 ``Z``-suffixed string.

    Spelled out so the bash caller, tests, and the model share one
    timestamp format; the ``Z`` suffix keeps the file greppable and
    timezone-unambiguous.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ExternalEvalState(BaseModel):
    """Checkpoint model for an incremental external-eval study run.

    Attributes:
        study_id: Identifier minted by the orchestrator on the first
            invocation (``external_eval_<utc-timestamp>``). Preserved
            across resume invocations so the final report can cite a
            single study id even when the run spanned several batches.
        model: Absolute path to the GGUF model under study. Recorded so
            a resumed run can detect a model swap (the operator changed
            ``--model`` between batches) and refuse rather than mix
            configurations from two different models into one Pearson
            computation.
        slice: Absolute path to the HumanEval+ slice.
        min_pairs: Minimum paired observations required for the study to
            be reportable (issue #900 criterion 2, default 30).
        created_at: ISO-8601 timestamp of the first invocation.
        updated_at: ISO-8601 timestamp of the most recent state write.
        configs_planned: Ordered list of every configuration label read
            from the ``--configs`` file on the first invocation.
        configs_completed: Ordered list of labels whose internal +
            external runs both finished. Parallel to the two rate
            arrays.
        completed_at: Per-label ISO-8601 completion timestamp, keyed by
            label. Lets an operator audit when each config finished.
        internal_rates: Per-configuration internal pass rate in
            ``[0.0, 1.0]``, one entry per completed config.
        external_rates: Per-configuration external pass rate in
            ``[0.0, 1.0]``, one entry per completed config.
    """

    study_id: str = Field(..., description="Orchestrator-minted study identifier.")
    model: str = Field(..., description="Absolute path to the GGUF model under study.")
    slice: str = Field(..., description="Absolute path to the HumanEval+ slice.")
    min_pairs: int = Field(
        default=MIN_PAIRED_OBSERVATIONS,
        ge=1,
        description="Minimum paired observations for a reportable study.",
    )
    created_at: str = Field(default_factory=_utc_now_iso, description="First-invocation timestamp.")
    updated_at: str = Field(
        default_factory=_utc_now_iso, description="Most recent write timestamp."
    )
    configs_planned: list[str] = Field(
        default_factory=list,
        description="Ordered labels read from the --configs file.",
    )
    configs_completed: list[str] = Field(
        default_factory=list,
        description="Ordered labels whose runs finished; parallel to the rate arrays.",
    )
    completed_at: dict[str, str] = Field(
        default_factory=dict,
        description="Per-label ISO-8601 completion timestamp.",
    )
    internal_rates: list[float] = Field(
        default_factory=list,
        description="Per-config internal pass rate in [0.0, 1.0].",
    )
    external_rates: list[float] = Field(
        default_factory=list,
        description="Per-config external pass rate in [0.0, 1.0].",
    )

    @model_validator(mode="after")
    def _check_parallel_lengths(self) -> ExternalEvalState:
        """Ensure the completed-label list and both rate arrays agree.

        A torn write (e.g. the process was killed between appending a
        label and appending its rates) would otherwise leave the
        checkpoint internally inconsistent; resuming it would silently
        misalign labels and rates. This validator turns that into a
        concrete ``ValidationError`` on load.
        """
        n = len(self.configs_completed)
        if len(self.internal_rates) != n or len(self.external_rates) != n:
            raise ValueError(
                f"configs_completed ({n}), internal_rates "
                f"({len(self.internal_rates)}), and external_rates "
                f"({len(self.external_rates)}) must have equal length."
            )
        return self

    @field_validator("internal_rates", "external_rates")
    @classmethod
    def _rates_in_unit_interval(cls, v: list[float]) -> list[float]:
        """Reject pass rates outside ``[0.0, 1.0]``.

        A rate outside the unit interval is a plumbing bug (e.g. the
        aggregator divided by the wrong total); persisting it would
        poison the Pearson computation on resume.
        """
        for r in v:
            if not (0.0 <= r <= 1.0):
                raise ValueError(f"pass rate {r} outside [0.0, 1.0]")
        return v


def load_state(path: str | Path) -> ExternalEvalState | None:
    """Load a checkpoint from ``path``.

    Returns ``None`` when the file does not exist (the common
    first-invocation case). Any other read or validation error is
    raised to the caller so the orchestrator can surface it rather than
    silently starting over — a corrupt checkpoint must not be masked,
    per AGENTS.md S2 (never silently swallow an exception).
    """
    p = Path(path)
    if not p.is_file():
        return None
    return ExternalEvalState.model_validate_json(p.read_text())


def save_state(path: str | Path, state: ExternalEvalState) -> None:
    """Persist ``state`` to ``path`` as human-readable, indented JSON.

    The write is atomic at the POSIX ``rename`` level: we serialise to a
    sibling temporary file in the same directory and ``os.replace`` it
    onto the target, so an interrupt mid-write leaves the previous
    checkpoint intact rather than a truncated JSON document. The state
    file is dot-prefixed by default because it is machine-managed
    checkpoint state, not a report an operator would open directly.
    """
    import os
    import tempfile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = _utc_now_iso()
    payload = state.model_dump_json(indent=2) + "\n"

    # Same-filesystem temp file so os.replace is atomic.
    fd, tmp_name = tempfile.mkstemp(prefix=".run_external_eval_state.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_name, p)
    except BaseException:
        # Clean up the temp file on any failure; never leave partial state.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def record_observation(
    state: ExternalEvalState,
    label: str,
    internal_rate: float,
    external_rate: float,
) -> ExternalEvalState:
    """Append a completed configuration's rates to ``state``.

    Re-recording an already-completed label is rejected: the orchestrator
    skips completed configs before calling this, but guarding here stops
    a double-count bug from silently inflating the Pearson input length
    past ``len(configs_planned)``.

    Returns the mutated state (also mutated in place) so the bash caller
    can chain ``save_state(path, record_observation(...))``.
    """
    if label in state.configs_completed:
        raise ValueError(
            f"configuration '{label}' is already recorded; "
            "the orchestrator should have skipped it (resume logic bug)."
        )
    if not (0.0 <= internal_rate <= 1.0):
        raise ValueError(f"internal_rate {internal_rate} outside [0.0, 1.0]")
    if not (0.0 <= external_rate <= 1.0):
        raise ValueError(f"external_rate {external_rate} outside [0.0, 1.0]")

    state.configs_completed.append(label)
    state.internal_rates.append(internal_rate)
    state.external_rates.append(external_rate)
    state.completed_at[label] = _utc_now_iso()
    return state


def remaining_configs(state: ExternalEvalState, all_labels: Sequence[str]) -> list[str]:
    """Return the planned labels that have not yet completed, in plan order.

    ``all_labels`` is the full ordered list parsed from ``--configs``;
    labels already present in ``configs_completed`` are dropped. Unknown
    completed labels (a resumed run whose ``--configs`` file shrank) are
    surfaced to the caller via the orchestrator rather than silently
    ignored here, but this function stays pure: it simply filters.
    """
    done = set(state.configs_completed)
    return [label for label in all_labels if label not in done]


def is_reportable(state: ExternalEvalState, min_pairs: int | None = None) -> bool:
    """True when enough paired observations exist to compute Pearson r.

    Defaults to the study's own ``min_pairs``; passing an explicit value
    overrides it (used by tests and by the orchestrator's
    ``FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS`` env var).
    """
    threshold = state.min_pairs if min_pairs is None else min_pairs
    return len(state.internal_rates) >= threshold and len(state.external_rates) >= threshold


def new_state(
    study_id: str,
    model: str,
    slice_path: str,
    configs_planned: Sequence[str],
    min_pairs: int = MIN_PAIRED_OBSERVATIONS,
) -> ExternalEvalState:
    """Construct a fresh checkpoint for the first invocation of a study.

    Centralising construction here means the bash orchestrator's
    "create state" path and the tests share one factory, so the field
    defaults (timestamps, empty arrays) cannot drift.
    """
    return ExternalEvalState(
        study_id=study_id,
        model=model,
        slice=slice_path,
        min_pairs=min_pairs,
        configs_planned=list(configs_planned),
    )


__all__ = [
    "ExternalEvalState",
    "is_reportable",
    "load_state",
    "new_state",
    "record_observation",
    "remaining_configs",
    "save_state",
]
