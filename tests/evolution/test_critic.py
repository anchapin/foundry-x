"""Edge-case unit tests for _compute_token_metrics (issue #1350)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from foundry_x.evolution.critic import Critic
from foundry_x.trace.logger import TraceEvent, TraceLogger


class TestComputeTokenMetricsEdgeCases:
    """Direct unit tests for _compute_token_metrics silent-degradation edge cases."""

    def test_compute_token_metrics_empty_store(self):
        """Empty trace store returns (0, None) — no exception raised."""
        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(
            "Q4_K_S", trace_path="/tmp/nonexistent/traces.db"
        )
        assert total_tokens == 0
        assert avg_cycle_time_s is None

    def test_compute_token_metrics_no_matching_sessions(self, tmp_path):
        """model_id mismatch returns (0, None) — no exception raised."""
        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        with logger.session(harness_version="test", model_id="other-model") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "do work"})

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(
            "Q4_K_S", str(trace_path)
        )
        assert total_tokens == 0
        assert avg_cycle_time_s is None

    def test_compute_token_metrics_missing_task_received(self, tmp_path):
        """Session missing task_received event is skipped — no exception raised."""
        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        model_id = "Q4_K_S"
        with logger.session(harness_version="test", model_id=model_id) as sid:
            logger.record(
                sid,
                kind="model_response",
                payload={
                    "step": 0,
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
            logger.record(
                sid,
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 1},
            )

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(
            model_id, str(trace_path)
        )
        assert total_tokens == 15
        assert avg_cycle_time_s is None

    def test_compute_token_metrics_unparseable_timestamp(self, tmp_path):
        """Corrupt ISO timestamp in task_received or outcome is skipped — no exception."""
        trace_path = tmp_path / "traces.db"

        model_id = "Q4_K_S"
        mock_logger = MagicMock()
        mock_logger.list_sessions.return_value = [
            MagicMock(model_id=model_id, session_id="test-session"),
        ]
        mock_logger.load_session.return_value = [
            TraceEvent(
                event_id="e1",
                session_id="test-session",
                timestamp="not-a-valid-timestamp",
                kind="task_received",
                payload={"prompt": "do work"},
            ),
            TraceEvent(
                event_id="e2",
                session_id="test-session",
                timestamp="2035-13-45T99:99:99",
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 1},
            ),
        ]

        with patch("foundry_x.evolution.critic.TraceLogger", return_value=mock_logger):
            critic = Critic(harness_dir=Path("/tmp/nonexistent"))
            total_tokens, avg_cycle_time_s = critic._compute_token_metrics(
                model_id, str(trace_path)
            )
        assert total_tokens == 0
        assert avg_cycle_time_s is None

    def test_compute_token_metrics_null_token_usage(self, tmp_path):
        """model_response with null/None token_usage is counted as zero — no exception."""
        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        model_id = "Q4_K_S"
        with logger.session(harness_version="test", model_id=model_id) as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "do work"})
            logger.record(
                sid,
                kind="model_response",
                payload={"step": 0, "token_usage": None},
            )
            logger.record(
                sid,
                kind="model_response",
                payload={"step": 1, "token_usage": {"prompt_tokens": 5}},
            )
            logger.record(
                sid,
                kind="model_response",
                payload={"step": 2, "token_usage": "not-a-dict"},
            )
            logger.record(
                sid,
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 3},
            )

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(
            model_id, str(trace_path)
        )
        assert total_tokens == 0
        assert avg_cycle_time_s is not None
        assert avg_cycle_time_s >= 0
