from __future__ import annotations

import pytest

from foundry_x.execution.runner import main, resolve_model_id
from foundry_x.trace.logger import TraceLogger


def test_explicit_foundry_model_id_wins():
    """Issue #12 (a): ``FOUNDRY_MODEL_ID`` is the highest-precedence source.
    A non-empty value is returned verbatim (whitespace-trimmed) and the
    lower-precedence candidates are ignored even when set."""
    env = {
        "FOUNDRY_MODEL_ID": "  gpt-4o-2024  ",
        "LLAMACPP_MODEL_PATH": "/srv/models/other.gguf",
        "OPENCODE_SERVER_URL": "http://example.com:4096",
    }

    assert resolve_model_id(env) == "gpt-4o-2024"


def test_derives_from_llamacpp_model_path_basename():
    """Issue #12 (b): with no explicit id, the llama.cpp model file basename
    is a stable, human-readable identity for a local-first run."""
    env = {"LLAMACPP_MODEL_PATH": "/srv/models/codellama-7b.Q5_K_M.gguf"}

    assert resolve_model_id(env) == "codellama-7b.Q5_K_M.gguf"


def test_derives_from_opencode_server_url_host():
    """Issue #12 (c): with neither an explicit id nor a model path, the host
    of the OpenAI-compatible endpoint is used, with scheme/port/path stripped."""
    env = {"OPENCODE_SERVER_URL": "http://127.0.0.1:4096/v1"}

    assert resolve_model_id(env) == "127.0.0.1"


def test_returns_none_when_nothing_set():
    """Issue #12 (d): absent all evidence, provenance is unknown rather than
    fabricated — the column stays NULL rather than guessing."""
    assert resolve_model_id({}) is None


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"FOUNDRY_MODEL_ID": "   "},
        {"FOUNDRY_MODEL_ID": "", "LLAMACPP_MODEL_PATH": "  "},
        {"OPENCODE_SERVER_URL": "not a url"},
    ],
)
def test_blank_or_unusable_values_fall_through_to_none(env):
    """Whitespace-only values (and an unparseable URL) never win; resolution
    continues and ultimately yields ``None`` rather than an empty/garbage id."""
    assert resolve_model_id(env) is None


def test_precedence_explicit_beats_model_path_beats_url():
    """The documented precedence is total: explicit > model path > url."""
    assert (
        resolve_model_id(
            {
                "FOUNDRY_MODEL_ID": "explicit",
                "LLAMACPP_MODEL_PATH": "/m/path.gguf",
                "OPENCODE_SERVER_URL": "http://host:1",
            }
        )
        == "explicit"
    )
    assert (
        resolve_model_id(
            {
                "LLAMACPP_MODEL_PATH": "/m/path.gguf",
                "OPENCODE_SERVER_URL": "http://host:1",
            }
        )
        == "path.gguf"
    )


def _stub_harness(harness_dir) -> None:
    """Build a minimal valid harness layout under ``harness_dir`` (issue #90).

    ``main()`` validates the harness layout before ``sys.path`` injection;
    these stubs satisfy that gate so the model-id unit under test runs.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def test_main_stamps_model_id_into_session_when_env_set(tmp_path, monkeypatch):
    """Acceptance test for issue #12: when ``FOUNDRY_MODEL_ID`` is set, the
    opened session's ``model_id`` column is populated (not NULL)."""
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_MODEL_ID", "qwen2.5-coder-7b")
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

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].model_id == "qwen2.5-coder-7b"


def test_main_leaves_model_id_null_when_unset(tmp_path, monkeypatch):
    """When no model-identity env var is present, ``model_id`` stays NULL
    rather than being populated with a guess."""
    db = tmp_path / "traces.db"
    for key in ("FOUNDRY_MODEL_ID", "LLAMACPP_MODEL_PATH", "OPENCODE_SERVER_URL"):
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

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].model_id is None
