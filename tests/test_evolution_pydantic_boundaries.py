"""Frozen-fixture drift guard for pydantic v2 boundary payloads (issue #96).

ADR-0006 mandates pydantic v2 for every payload crossing a module boundary
(``FailureReport``, ``ProposedEdit``, ``CriticVerdict``). Issue #14 migrated
these off ``@dataclass``, but nothing prevents a future contributor from
reverting the migration or silently renaming a field. This file pins the
boundary payloads so unintended schema drift fails CI.

Acceptance criteria (issue #96):

1. Each boundary model is a ``pydantic.BaseModel`` subclass (not a dataclass).
2. ``model_validate`` rejects a payload whose field types do not match.
3. ``model_dump_json`` followed by ``model_validate_json`` round-trips losslessly.
4. A frozen JSON snapshot under ``tests/fixtures/evolution/`` matches the live
   ``model_dump(mode='json')`` output. Drift is detected deterministically.

Refreshing the fixtures (when a payload change is intentional):

* Run the helper test below with the env var ``FOUNDRY_REFRESH_FIXTURES=1``;
  it rewrites the JSON files in-place.
* Commit the refreshed fixtures in the same PR as the schema change so the
  drift guard and the schema move together.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import ProposedEdit

# --- canonical samples ---------------------------------------------------------
# Deterministic instances whose ``model_dump(mode="json")`` output matches the
# frozen fixtures. Built so every model field is populated (no defaults left
# to vary by environment).
_SAMPLE_FAILURE_REPORT = FailureReport(
    session_id="sess-fixture-001",
    summary=("tool-error failure (kind=tool_error, signal=kind:tool_error): no tool named foo"),
    failed_steps=[
        {
            "index": 0,
            "event_id": "evt-001",
            "kind": "tool_error",
            "signal": "kind:tool_error",
            "payload": {
                "tool": "foo",
                "args": {"path": "/tmp/a"},
                "exit_code": 1,
                "stderr": "no such tool",
            },
        }
    ],
    suspected_causes=[
        "Tool execution raised an error (matched: error). Inspect the failing call's traceback."
    ],
    proposed_class="tool-error",
)

_SAMPLE_PROPOSED_EDIT = ProposedEdit(
    target_file="harness/skills/retrieval/SKILL.md",
    rationale="tighten tool guidance per ADR-0004",
    unified_diff=(
        "--- a/harness/skills/retrieval/SKILL.md\n"
        "+++ b/harness/skills/retrieval/SKILL.md\n"
        "@@ -1,3 +1,3 @@\n"
        "# Retrieval skill\n"
        "-old guidance\n"
        "+new guidance\n"
    ),
)

_SAMPLE_CRITIC_VERDICT = CriticVerdict(
    verdict=True,
    passed_checks=["git apply", "pytest"],
    failed_checks=[],
    notes="all checks passed (truncated tail)",
)

# Class-name -> frozen-fixture filename. ``__name__.lower()`` would yield
# ``failurereport`` / ``proposededit`` / ``criticverdict`` (no underscore), so
# we keep an explicit map rather than re-deriving the name from the class.
_FIXTURE_NAMES: dict[type[BaseModel], str] = {
    FailureReport: "failure_report.json",
    ProposedEdit: "proposed_edit.json",
    CriticVerdict: "critic_verdict.json",
}


def _fixtures_dir() -> Path:
    """Return the directory holding the frozen JSON fixtures."""
    return Path(__file__).parent / "fixtures" / "evolution"


def _dump_indented(model: BaseModel) -> str:
    """Stable JSON serialization: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


# --- (1) BaseModel subclass guard ---------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [FailureReport, ProposedEdit, CriticVerdict],
    ids=["FailureReport", "ProposedEdit", "CriticVerdict"],
)
def test_boundary_model_is_pydantic_basemodel_subclass(model_cls: type[BaseModel]) -> None:
    """Each boundary payload must subclass ``pydantic.BaseModel``.

    A regression to ``@dataclass`` (or any non-pydantic schema layer) would
    silently remove JSON validation; this guard fails the moment someone
    swaps the base class.
    """
    assert issubclass(model_cls, BaseModel)
    # ``model_validate``/``model_dump_json`` are pydantic v2-only APIs. Their
    # presence proves we are on v2 (not v1).
    assert hasattr(model_cls, "model_validate")
    assert hasattr(model_cls, "model_dump_json")
    assert hasattr(model_cls, "model_validate_json")


# --- (2) wrong-type rejection via model_validate ------------------------------


def test_failure_report_rejects_non_string_session_id() -> None:
    with pytest.raises(ValidationError):
        FailureReport.model_validate({"session_id": 123, "summary": "x"})


def test_failure_report_rejects_non_list_failed_steps() -> None:
    with pytest.raises(ValidationError):
        FailureReport.model_validate(
            {"session_id": "s", "summary": "x", "failed_steps": "not-a-list"},
        )


def test_proposed_edit_rejects_blank_target_file() -> None:
    with pytest.raises(ValidationError):
        ProposedEdit.model_validate(
            {"target_file": "", "rationale": "r", "unified_diff": "d"},
        )


def test_proposed_edit_rejects_non_string_target_file() -> None:
    with pytest.raises(ValidationError):
        ProposedEdit.model_validate(
            {"target_file": 7, "rationale": "r", "unified_diff": "d"},
        )


def test_proposed_edit_rejects_target_file_outside_harness_tree() -> None:
    # Confinement to the harness tree (ADR-0004) is enforced at the model
    # boundary (ADR-0006); model_validate must catch a violation.
    with pytest.raises(ValidationError):
        ProposedEdit.model_validate(
            {
                "target_file": "src/foundry_x/evolution/evolver.py",
                "rationale": "r",
                "unified_diff": "d",
            },
        )


def test_critic_verdict_rejects_non_bool_approved() -> None:
    # Pydantic v2 coerces the string ``"yes"`` to ``True``, so we use a
    # deliberately non-coercible value (a list) to assert the strict-type
    # contract that ADR-0006 relies on.
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate({"approved": ["not", "a", "bool"]})


def test_critic_verdict_rejects_non_list_passed_checks() -> None:
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate(
            {"approved": True, "passed_checks": "pytest"},
        )


# --- (3) JSON round-trip stability --------------------------------------------


@pytest.mark.parametrize(
    "model_cls",
    [FailureReport, ProposedEdit, CriticVerdict],
    ids=["FailureReport", "ProposedEdit", "CriticVerdict"],
)
def test_model_dump_json_to_model_validate_json_round_trips(
    model_cls: type[BaseModel],
) -> None:
    """``model_dump_json`` then ``model_validate_json`` recovers the original.

    Boundary payloads must round-trip through JSON unchanged so the trace
    store, the Critic gate, and any cross-process tooling can persist and
    re-hydrate them without loss.
    """
    sample = {
        FailureReport: _SAMPLE_FAILURE_REPORT,
        ProposedEdit: _SAMPLE_PROPOSED_EDIT,
        CriticVerdict: _SAMPLE_CRITIC_VERDICT,
    }[model_cls]
    encoded = sample.model_dump_json()
    rehydrated = model_cls.model_validate_json(encoded)
    assert rehydrated == sample
    # ``model_dump`` round-trip too (covers the python-dict path used by
    # in-process callers that do not go through JSON).
    assert model_cls.model_validate(sample.model_dump()) == sample


# --- (4) frozen-fixture drift guard --------------------------------------------


# Parametrize over (model_cls, fixture_name) so each boundary payload has its
# own self-contained snapshot. A failure pinpoints exactly which model drifted.
@pytest.mark.parametrize(
    ("model_cls", "fixture_name"),
    list(_FIXTURE_NAMES.items()),
)
def test_frozen_fixture_matches_current_model_dump_json(
    model_cls: type[BaseModel],
    fixture_name: str,
) -> None:
    """Frozen snapshot must match the live ``model_dump(mode='json')`` output.

    If a model gains, loses, or renames a field, its JSON shape changes; if
    a field type changes (e.g. ``int`` to ``str``) the serialized form
    shifts. Either change trips this guard until the fixture is refreshed
    alongside the schema change (see module docstring).
    """
    fixture_path = _fixtures_dir() / fixture_name
    assert fixture_path.exists(), f"missing frozen fixture: {fixture_path}"

    sample = {
        FailureReport: _SAMPLE_FAILURE_REPORT,
        ProposedEdit: _SAMPLE_PROPOSED_EDIT,
        CriticVerdict: _SAMPLE_CRITIC_VERDICT,
    }[model_cls]

    current = _dump_indented(sample)
    frozen = fixture_path.read_text()

    if os.environ.get("FOUNDRY_REFRESH_FIXTURES") == "1":
        # Convenience escape hatch: setting this env var rewrites the JSON
        # files in-place so an intentional schema change can be captured in
        # one test run, then committed.
        fixture_path.write_text(current)
        pytest.skip(f"refreshed {fixture_path}")

    assert current == frozen, (
        f"Payload drift in {model_cls.__name__!s}. The frozen fixture "
        f"({fixture_path}) does not match the live model_dump(mode='json') "
        "output. If this drift is intentional, refresh the fixture by "
        "running:\n\n"
        "    FOUNDRY_REFRESH_FIXTURES=1 uv run pytest "
        "tests/test_evolution_pydantic_boundaries.py\n\n"
        "and commit the resulting diff in the same PR as the schema "
        "change. Otherwise, revert the schema change."
    )


# --- (5) refresh helper (gated by env var, never silently writes) -------------


def test_refresh_fixtures_when_env_var_set(capsys: pytest.CaptureFixture[str]) -> None:
    """Explicit opt-in rewrite helper.

    CI never sets ``FOUNDRY_REFRESH_FIXTURES``, so this test is a no-op in
    normal runs. With the env var present it rewrites every fixture so the
    developer can stage intentional schema changes for review.
    """
    if os.environ.get("FOUNDRY_REFRESH_FIXTURES") != "1":
        pytest.skip("FOUNDRY_REFRESH_FIXTURES not set; fixtures are read-only")

    for model_cls in (FailureReport, ProposedEdit, CriticVerdict):
        sample = {
            FailureReport: _SAMPLE_FAILURE_REPORT,
            ProposedEdit: _SAMPLE_PROPOSED_EDIT,
            CriticVerdict: _SAMPLE_CRITIC_VERDICT,
        }[model_cls]
        fixture_name = _FIXTURE_NAMES[model_cls]
        fixture_path = _fixtures_dir() / fixture_name
        fixture_path.write_text(_dump_indented(sample))
        print(f"wrote {fixture_path}")


# --- (6) sanity: the fixtures are well-formed JSON ----------------------------


@pytest.mark.parametrize(
    "fixture_name",
    list(_FIXTURE_NAMES.values()),
)
def test_frozen_fixture_is_valid_json(fixture_name: str) -> None:
    """Sanity guard so a hand-edited fixture is rejected at parse time."""
    payload = json.loads((_fixtures_dir() / fixture_name).read_text())
    assert isinstance(payload, dict)
    # Confirm the top-level keys are strings (pydantic field names) and the
    # JSON is non-empty. This catches the "the file is empty" failure mode
    # long before the round-trip guard fires.
    assert payload, f"{fixture_name} is empty"
    assert all(isinstance(k, str) for k in payload.keys())


# --- (7) public re-exports -----------------------------------------------------


def test_module_exposes_boundary_models() -> None:
    """Imports above must resolve to the same classes the rest of the codebase uses.

    A future refactor that moves these classes into ``foundry_x.evolution.*``
    submodules must keep the import targets exported under the documented
    names, or the Critic and Evolver won't find them either.
    """
    assert isinstance(_SAMPLE_FAILURE_REPORT, FailureReport)
    assert isinstance(_SAMPLE_PROPOSED_EDIT, ProposedEdit)
    assert isinstance(_SAMPLE_CRITIC_VERDICT, CriticVerdict)
    # Bonus: the dict-of-any shape for failed_steps must survive, since the
    # trace subsystem relies on it as a per-step escape hatch (ADR-0006).
    assert isinstance(
        _SAMPLE_FAILURE_REPORT.failed_steps[0]["payload"],
        dict,  # type: ignore[index]
    )
