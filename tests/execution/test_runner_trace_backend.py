from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from foundry_x.execution.runner import main, resolve_trace_backend


# --- resolve_trace_backend unit tests ----------------------------------------


def test_defaults_to_sqlite_when_unset():
    """Issue #13: an absent ``FOUNDRY_TRACE_BACKEND`` yields ``sqlite``,
    matching the ``.env.example`` default."""
    assert resolve_trace_backend({}) == "sqlite"


def test_returns_jsonl_when_set():
    """The documented export format (ADR-0003) is selected verbatim."""
    assert resolve_trace_backend({"FOUNDRY_TRACE_BACKEND": "jsonl"}) == "jsonl"


@pytest.mark.parametrize(
    "value",
    ["JSONL", "  Sqlite  ", "jsonL"],
)
def test_is_case_and_whitespace_insensitive(value: str):
    """A hand-edited ``.env`` may carry surrounding whitespace or different
    casing; both are normalized rather than rejected."""
    assert resolve_trace_backend({"FOUNDRY_TRACE_BACKEND": value}) in {"sqlite", "jsonl"}
    assert resolve_trace_backend({"FOUNDRY_TRACE_BACKEND": "  jsonl  "}) == "jsonl"
    assert resolve_trace_backend({"FOUNDRY_TRACE_BACKEND": "SQLITE"}) == "sqlite"


def test_empty_string_falls_back_to_default():
    """An empty value is treated as unset, not as an invalid backend."""
    assert resolve_trace_backend({"FOUNDRY_TRACE_BACKEND": ""}) == "sqlite"


def test_invalid_backend_raises_value_error():
    """Issue #13 proposal: an unknown backend fails fast at startup with a
    message naming the valid options, rather than silently producing no
    trace (AGENTS.md §2 — never silently swallow)."""
    with pytest.raises(ValueError) as exc_info:
        resolve_trace_backend({"FOUNDRY_TRACE_BACKEND": "csv"})

    message = str(exc_info.value)
    assert "csv" in message
    assert "sqlite" in message
    assert "jsonl" in message


# --- end-to-end main() acceptance tests --------------------------------------


def _argv(task: str, trace_path, harness_dir, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fx-runner",
            "--task",
            task,
            "--trace-path",
            str(trace_path),
            "--harness-dir",
            str(harness_dir),
        ],
    )


def _stub_harness(harness_dir: Path) -> None:
    """Build a minimal valid harness layout under ``harness_dir`` (issue #90).

    ``main()`` invokes ``validate_harness_layout`` before touching
    ``sys.path``; without these stubs every ``main()`` call below would
    abort with ``HarnessValidationError`` and the unit under test
    (trace backend selection) would never run.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def test_jsonl_backend_writes_jsonl_file(tmp_path, monkeypatch):
    """Acceptance test for issue #13: with ``FOUNDRY_TRACE_BACKEND=jsonl``,
    ``main`` writes a ``.jsonl`` trace file (no SQLite DB is created) for the
    ``task_received`` event."""
    trace_path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("FOUNDRY_TRACE_BACKEND", "jsonl")
    _stub_harness(tmp_path)
    _argv("noop task", trace_path, tmp_path, monkeypatch)

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    # No SQLite database should be produced at the configured path.
    assert trace_path.exists()
    lines = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    kinds = [record.get("kind") for record in lines]
    assert "task_received" in kinds
    assert "task_completed" in kinds


def test_sqlite_default_creates_sqlite_db(tmp_path, monkeypatch):
    """With the backend unset, the runner produces the legacy SQLite store —
    confirming the fix does not change the default behavior."""
    db = tmp_path / "traces.db"
    for key in ("FOUNDRY_TRACE_BACKEND",):
        monkeypatch.delenv(key, raising=False)
    _stub_harness(tmp_path)
    _argv("noop task", db, tmp_path, monkeypatch)

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    assert db.exists()
    # SQLite files begin with the magic header "SQLite format 3".
    assert db.read_bytes()[:15] == b"SQLite format 3"


def test_invalid_backend_aborts_main(tmp_path, monkeypatch):
    """An invalid backend value aborts the run before any session is opened,
    so no partial trace store is left behind."""
    trace_path = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_TRACE_BACKEND", "xml")
    _stub_harness(tmp_path)
    _argv("noop task", trace_path, tmp_path, monkeypatch)

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    with pytest.raises(ValueError):
        main(run_task_fn=noop_run_task)

    assert not trace_path.exists()
