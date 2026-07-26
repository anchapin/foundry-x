from __future__ import annotations

from foundry_x.execution.runner import (
    main,
    resolve_quantization,
)
from foundry_x.trace.logger import TraceLogger


def _stub_harness(harness_dir) -> None:
    """Build a minimal valid harness layout under ``harness_dir`` (issue #90)."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def test_resolve_quantization_returns_env_value():
    """``FOUNDRY_QUANTIZATION`` is returned verbatim when set."""
    env = {"FOUNDRY_QUANTIZATION": " Q5_K_M "}
    assert resolve_quantization(env) == "Q5_K_M"


def test_resolve_quantization_returns_none_when_unset():
    """When ``FOUNDRY_QUANTIZATION`` is absent or blank, None is returned."""
    assert resolve_quantization({}) is None
    assert resolve_quantization({"FOUNDRY_QUANTIZATION": ""}) is None
    assert resolve_quantization({"FOUNDRY_QUANTIZATION": "   "}) is None


def test_main_stamps_quantization_and_harness_variant_into_session_metadata(tmp_path, monkeypatch):
    """Issue #955: quantization and harness_variant from CLI args are stored
    in the session metadata dict so the external-eval aggregator can group
    critic_verdict events per configuration.

    ``FOUNDRY_MODEL_ID`` is set via env var (not CLI) so that ``model_id``
    is populated for traceability without triggering ``build_model_adapter_with_overrides``
    (which requires a live endpoint)."""
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_MODEL_ID", "codellama-7b")
    _stub_harness(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fx-runner",
            "--task",
            "noop",
            "--trace-path",
            str(db),
            "--harness-dir",
            str(tmp_path),
            "--quantization",
            "Q5_K_M",
            "--harness-variant",
            "q5km",
        ],
    )

    async def noop_run_task(task, harness_dir, log, session_id):
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].model_id == "codellama-7b"
    assert sessions[0].metadata.get("quantization") == "Q5_K_M"
    assert sessions[0].metadata.get("harness_variant") == "q5km"


def test_main_stamps_harness_variant_from_env_into_session_metadata(tmp_path, monkeypatch):
    """Issue #955: when ``FOUNDRY_HARNESS_VARIANT`` is set in the environment,
    it is stored in session metadata even without a CLI override."""
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_HARNESS_VARIANT", "q4km")
    _stub_harness(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fx-runner",
            "--task",
            "noop",
            "--trace-path",
            str(db),
            "--harness-dir",
            str(tmp_path),
        ],
    )

    async def noop_run_task(task, harness_dir, log, session_id):
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].metadata.get("harness_variant") == "q4km"


def test_main_stamps_quantization_from_env_into_session_metadata(tmp_path, monkeypatch):
    """Issue #955: ``FOUNDRY_QUANTIZATION`` is stored in session metadata when
    the CLI arg is not provided."""
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_QUANTIZATION", "Q8_0")
    _stub_harness(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fx-runner",
            "--task",
            "noop",
            "--trace-path",
            str(db),
            "--harness-dir",
            str(tmp_path),
        ],
    )

    async def noop_run_task(task, harness_dir, log, session_id):
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].metadata.get("quantization") == "Q8_0"


def test_main_cli_harness_variant_overrides_env(tmp_path, monkeypatch):
    """Issue #955: ``--harness-variant`` CLI arg takes precedence over
    ``FOUNDRY_HARNESS_VARIANT`` env var."""
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_HARNESS_VARIANT", "env-variant")
    _stub_harness(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fx-runner",
            "--task",
            "noop",
            "--trace-path",
            str(db),
            "--harness-dir",
            str(tmp_path),
            "--harness-variant",
            "cli-variant",
        ],
    )

    async def noop_run_task(task, harness_dir, log, session_id):
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].metadata.get("harness_variant") == "cli-variant"


def test_main_session_metadata_empty_when_no_tags_set(tmp_path, monkeypatch):
    """When neither quantization nor harness_variant is set (CLI or env),
    the session metadata dict is empty."""
    db = tmp_path / "traces.db"
    for key in (
        "FOUNDRY_QUANTIZATION",
        "FOUNDRY_HARNESS_VARIANT",
        "FOUNDRY_MODEL_ID",
        "LLAMACPP_MODEL_PATH",
        "OPENCODE_SERVER_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    _stub_harness(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "fx-runner",
            "--task",
            "noop",
            "--trace-path",
            str(db),
            "--harness-dir",
            str(tmp_path),
        ],
    )

    async def noop_run_task(task, harness_dir, log, session_id):
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].metadata == {}
