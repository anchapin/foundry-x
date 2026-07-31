"""Tests for wave-planner.js — issue #1404 regression suite."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

WAVE_PLANNER = Path(__file__).parent.parent.parent / "harness" / "scripts" / "wave-planner.js"


def _run_wave_planner(issues: list[dict]) -> dict:
    """Invoke wave-planner.js with issues JSON and return parsed plan."""
    proc = subprocess.run(
        ["node", str(WAVE_PLANNER)],
        input=json.dumps(issues),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 0, f"wave-planner failed: {proc.stderr}"
    return json.loads(proc.stdout)


class TestLogsExclusion:
    """Ensure logs/traces.db is never treated as a code dependency (issue #1404)."""

    def test_logs_traces_db_not_matched_as_file_ref(self) -> None:
        issues = [
            {
                "number": 1,
                "title": "Fix foo",
                "body": "See `logs/traces.db` for details",
                "labels": [],
            },
        ]
        plan = _run_wave_planner(issues)
        affected = plan["waves"][0]["issues"][0]["affected_files"]
        assert "logs/traces.db" not in affected

    def test_logs_traces_db_with_line_number_not_matched(self) -> None:
        issues = [
            {
                "number": 2,
                "title": "Fix bar",
                "body": "See `logs/traces.db:42` for trace",
                "labels": [],
            },
        ]
        plan = _run_wave_planner(issues)
        affected = plan["waves"][0]["issues"][0]["affected_files"]
        assert "logs/traces.db" not in affected

    def test_real_code_files_still_matched(self) -> None:
        issues = [
            {
                "number": 3,
                "title": "Refactor src/foo.py",
                "body": "Changes in `src/foo.py`",
                "labels": [],
            },
        ]
        plan = _run_wave_planner(issues)
        affected = plan["waves"][0]["issues"][0]["affected_files"]
        assert "src/foo.py" in affected

    def test_logs_referenced_alone_not_matched(self) -> None:
        issues = [
            {
                "number": 4,
                "title": "Write to logs",
                "body": "I was looking at logs/traces.db when I found the bug",
                "labels": [],
            },
        ]
        plan = _run_wave_planner(issues)
        affected = plan["waves"][0]["issues"][0]["affected_files"]
        assert "logs/traces.db" not in affected

    def test_logs_traces_db_does_not_cause_false_conflict(self) -> None:
        """Two issues both mentioning logs/traces.db should NOT be put in the same wave."""
        issues = [
            {
                "number": 5,
                "title": "Fix A",
                "body": "See `logs/traces.db` for details",
                "labels": [],
            },
            {
                "number": 6,
                "title": "Fix B",
                "body": "See `logs/traces.db` too",
                "labels": [],
            },
        ]
        plan = _run_wave_planner(issues)
        wave_1 = plan["waves"][0]["issues"]
        wave_2 = plan["waves"][1]["issues"] if len(plan["waves"]) > 1 else []
        wave_1_nums = {issue["number"] for issue in wave_1}
        wave_2_nums = {issue["number"] for issue in wave_2}
        assert wave_1_nums != wave_2_nums, (
            "Issues only sharing logs/traces.db must not be conflict-grouped"
        )


class TestWavePlanning:
    """Basic wave planning smoke tests."""

    def test_empty_input(self) -> None:
        plan = _run_wave_planner([])
        assert plan["total_issues"] == 0
        assert plan["total_waves"] == 0

    def test_single_issue_one_wave(self) -> None:
        issues = [{"number": 10, "title": "Fix x", "body": "", "labels": []}]
        plan = _run_wave_planner(issues)
        assert plan["total_issues"] == 1
        assert plan["total_waves"] == 1

    def test_conflicting_files_same_wave_limit(self) -> None:
        """Four issues touching the same file must spread across two waves."""
        issues = [
            {"number": i, "title": f"Fix {i}", "body": f"`src/shared.py:{i}`", "labels": []}
            for i in range(1, 5)
        ]
        plan = _run_wave_planner(issues)
        assert plan["total_waves"] == 2
        assert all(len(w["issues"]) <= 3 for w in plan["waves"])
