"""Unit tests for :mod:`foundry_x.evaluation.study_state` (issue #1040).

These pin the checkpoint model's invariants, the atomic save/load
round-trip, the resume filter, and the reportability threshold. They are
pure (no model endpoint, no ``logs/traces.db``) so they run in the
default offline pytest suite alongside ``tests/test_correlation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from foundry_x.evaluation.correlation import MIN_PAIRED_OBSERVATIONS
from foundry_x.evaluation.study_state import (
    ExternalEvalState,
    is_reportable,
    load_state,
    new_state,
    record_observation,
    remaining_configs,
    save_state,
)

PLANNED = [f"cfg{i}" for i in range(MIN_PAIRED_OBSERVATIONS)]


def _fresh_state() -> ExternalEvalState:
    return new_state(
        study_id="external_eval_test",
        model="/srv/models/foo.gguf",
        slice_path="/repo/benchmarks/external/humaneval_plus_sample.jsonl",
        configs_planned=PLANNED,
    )


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


def test_new_state_has_empty_completion(tmp_path: Path) -> None:
    state = _fresh_state()
    assert state.configs_planned == PLANNED
    assert state.configs_completed == []
    assert state.internal_rates == []
    assert state.external_rates == []
    assert state.completed_at == {}
    assert state.min_pairs == MIN_PAIRED_OBSERVATIONS
    # Timestamps are populated and ISO-8601 shaped.
    assert state.created_at.endswith("Z")
    assert state.updated_at.endswith("Z")


def test_state_defaults_min_pairs_to_canonical_threshold() -> None:
    state = ExternalEvalState(study_id="x", model="m", slice="s")
    assert state.min_pairs == MIN_PAIRED_OBSERVATIONS == 30


# ---------------------------------------------------------------------------
# record_observation
# ---------------------------------------------------------------------------


def test_record_observation_appends_in_parallel() -> None:
    state = _fresh_state()
    record_observation(state, "cfg0", 0.75, 0.5)
    assert state.configs_completed == ["cfg0"]
    assert state.internal_rates == [0.75]
    assert state.external_rates == [0.5]
    assert "cfg0" in state.completed_at


def test_record_observation_rejects_duplicate_label() -> None:
    state = _fresh_state()
    record_observation(state, "cfg0", 0.75, 0.5)
    with pytest.raises(ValueError, match="already recorded"):
        record_observation(state, "cfg0", 0.8, 0.6)


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, -1.0, 2.0])
def test_record_observation_rejects_out_of_range_rates(bad_rate: float) -> None:
    state = _fresh_state()
    with pytest.raises(ValueError, match="outside"):
        record_observation(state, "cfg0", bad_rate, 0.5)
    with pytest.raises(ValueError, match="outside"):
        record_observation(state, "cfg0", 0.5, bad_rate)


# ---------------------------------------------------------------------------
# remaining_configs
# ---------------------------------------------------------------------------


def test_remaining_configs_skips_completed() -> None:
    state = _fresh_state()
    record_observation(state, "cfg0", 0.7, 0.4)
    record_observation(state, "cfg1", 0.8, 0.6)
    remaining = remaining_configs(state, PLANNED)
    assert remaining == PLANNED[2:]
    # Order is preserved from the plan, not from completion order.
    assert remaining[0] == "cfg2"


def test_remaining_configs_empty_when_all_done() -> None:
    state = _fresh_state()
    for i, label in enumerate(PLANNED):
        record_observation(state, label, 0.5, 0.5)
    assert remaining_configs(state, PLANNED) == []


def test_remaining_configs_handles_shrunk_plan() -> None:
    """A resumed run whose --configs file shrank must not blow up."""
    state = _fresh_state()
    record_observation(state, "cfg0", 0.5, 0.5)
    # Pass a smaller plan; completed labels outside the new plan are simply
    # absent from the result (the orchestrator warns separately).
    remaining = remaining_configs(state, ["cfg0", "cfg1"])
    assert remaining == ["cfg1"]


# ---------------------------------------------------------------------------
# is_reportable
# ---------------------------------------------------------------------------


def test_is_reportable_false_below_threshold() -> None:
    state = _fresh_state()
    for i in range(MIN_PAIRED_OBSERVATIONS - 1):
        record_observation(state, PLANNED[i], 0.5, 0.5)
    assert not is_reportable(state)


def test_is_reportable_true_at_threshold() -> None:
    state = _fresh_state()
    for i in range(MIN_PAIRED_OBSERVATIONS):
        record_observation(state, PLANNED[i], float(i) / 40, float(i) / 40)
    assert is_reportable(state)


def test_is_reportable_respects_explicit_override() -> None:
    state = _fresh_state()
    record_observation(state, "cfg0", 0.5, 0.5)
    record_observation(state, "cfg1", 0.7, 0.6)
    assert not is_reportable(state)
    assert is_reportable(state, min_pairs=2)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    state = _fresh_state()
    record_observation(state, "cfg0", 0.75, 0.5)
    record_observation(state, "cfg1", 0.8, 0.6)

    path = tmp_path / ".run_external_eval_state.json"
    save_state(path, state)

    loaded = load_state(path)
    assert loaded is not None
    assert loaded.configs_completed == ["cfg0", "cfg1"]
    assert loaded.internal_rates == [0.75, 0.8]
    assert loaded.external_rates == [0.5, 0.6]
    assert loaded.configs_planned == PLANNED
    assert loaded.study_id == "external_eval_test"


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert load_state(tmp_path / "absent.json") is None


def test_state_file_is_human_readable_indented_json(tmp_path: Path) -> None:
    state = _fresh_state()
    record_observation(state, "cfg0", 0.75, 0.5)
    path = tmp_path / ".run_external_eval_state.json"
    save_state(path, state)

    text = path.read_text()
    # Indented (multi-line) and parseable directly by stdlib json.
    assert "\n" in text
    parsed = json.loads(text)
    assert parsed["configs_completed"] == ["cfg0"]
    assert parsed["internal_rates"] == [0.75]
    # Auditability: per-label timestamps are present.
    assert "cfg0" in parsed["completed_at"]


def test_save_is_atomic_on_interrupt(tmp_path: Path) -> None:
    """A second save must not leave a truncated file visible to readers.

    save_state writes to a sibling temp file and os.replace's it onto
    the target, so a reader either sees the previous checkpoint or the
    new one — never a half-written document.
    """
    path = tmp_path / ".run_external_eval_state.json"
    state = _fresh_state()
    record_observation(state, "cfg0", 0.75, 0.5)
    save_state(path, state)
    first_text = path.read_text()

    state2 = _fresh_state()
    record_observation(state2, "cfg0", 0.75, 0.5)
    record_observation(state2, "cfg1", 0.8, 0.6)
    save_state(path, state2)

    # After the second save the file reflects the new state, and at no
    # point was an invalid document observable (we assert the final
    # state is valid JSON and distinct from the first).
    second_text = path.read_text()
    assert second_text != first_text
    json.loads(second_text)  # no raise


def test_load_raises_on_corrupt_checkpoint(tmp_path: Path) -> None:
    """A torn/corrupt checkpoint must surface, not silently restart."""
    path = tmp_path / ".run_external_eval_state.json"
    path.write_text("{ not valid json")
    with pytest.raises(ValidationError):
        load_state(path)


def test_load_raises_on_inconsistent_checkpoint(tmp_path: Path) -> None:
    """Parallel-array length mismatch is rejected by the model validator."""
    path = tmp_path / ".run_external_eval_state.json"
    path.write_text(
        json.dumps(
            {
                "study_id": "x",
                "model": "m",
                "slice": "s",
                "configs_completed": ["cfg0", "cfg1"],
                "internal_rates": [0.5],  # length mismatch
                "external_rates": [0.5, 0.6],
            }
        )
    )
    with pytest.raises(ValidationError):
        load_state(path)


def test_load_raises_on_out_of_range_rate(tmp_path: Path) -> None:
    path = tmp_path / ".run_external_eval_state.json"
    path.write_text(
        json.dumps(
            {
                "study_id": "x",
                "model": "m",
                "slice": "s",
                "configs_completed": ["cfg0"],
                "internal_rates": [1.5],  # outside [0,1]
                "external_rates": [0.5],
            }
        )
    )
    with pytest.raises(ValidationError):
        load_state(path)


# ---------------------------------------------------------------------------
# Resume workflow simulation (matches the issue's batched scenario)
# ---------------------------------------------------------------------------


def test_six_batches_of_five_accumulate_to_reportable(tmp_path: Path) -> None:
    """Issue #1040 acceptance: 6 batches of 5 configs accumulate across
    invocations and become reportable after the 30th config."""
    path = tmp_path / ".run_external_eval_state.json"
    batch_size = 5
    expected_batches = MIN_PAIRED_OBSERVATIONS // batch_size

    for _ in range(expected_batches):
        existing = load_state(path)
        state = _fresh_state() if existing is None else existing
        todo = remaining_configs(state, PLANNED)[:batch_size]
        assert len(todo) == batch_size
        for label in todo:
            record_observation(state, label, 0.5 + 0.01 * PLANNED.index(label), 0.4)
        save_state(path, state)

    final = load_state(path)
    assert final is not None
    assert len(final.configs_completed) == MIN_PAIRED_OBSERVATIONS
    assert is_reportable(final)


def test_resume_skips_already_completed(tmp_path: Path) -> None:
    """Second invocation does not re-record configs the first completed."""
    path = tmp_path / ".run_external_eval_state.json"
    state = _fresh_state()
    for label in PLANNED[:5]:
        record_observation(state, label, 0.5, 0.5)
    save_state(path, state)

    resumed = load_state(path)
    assert resumed is not None
    todo = remaining_configs(resumed, PLANNED)
    assert todo == PLANNED[5:]
    # Recording one of the already-done labels is rejected — guards a
    # resume logic bug from double-counting.
    with pytest.raises(ValueError, match="already recorded"):
        record_observation(resumed, "cfg0", 0.5, 0.5)
