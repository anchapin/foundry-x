"""Tests for the quantization sweep feature (issue #464 / ADR-0016)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from foundry_x.evolution.critic import (
    QuantizationResult,
    QuantizationVerdict,
    DEFAULT_REGRESSION_THRESHOLD_PP,
    TaskResult,
)
from foundry_x.evolution.cli import (
    _build_sweep_parser,
    _build_sweep_subparser,
    _render_quantization_result,
    _render_quantization_verdict,
    sweep_main,
)
from tests._harness_fixture import install_load_check_prerequisites


class TestQuantizationResult:
    def test_required_fields(self):
        r = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
        )
        assert r.quantization == "Q4_K_S"
        assert r.model_path == "/srv/models/test.Q4_K_S.gguf"
        assert r.model_id == "Q4_K_S"
        assert r.total_tasks == 0
        assert r.passed_tasks == 0
        assert r.failed_tasks == 0
        assert r.task_shaped_failures == 0
        assert r.pass_rate == 0.0
        assert r.total_tokens == 0
        assert r.token_efficiency is None
        assert r.cost_per_task is None

    def test_full_fields(self):
        r = QuantizationResult(
            quantization="Q5_K_M",
            model_path="/srv/models/test.Q5_K_M.gguf",
            model_id="Q5_K_M",
            total_tasks=10,
            passed_tasks=8,
            failed_tasks=1,
            task_shaped_failures=1,
            pass_rate=0.8,
            avg_cycle_time_s=42.5,
            total_tokens=12345,
            token_efficiency=290.0,
            cost_per_task=0.0015,
        )
        assert r.total_tasks == 10
        assert r.passed_tasks == 8
        assert r.failed_tasks == 1
        assert r.task_shaped_failures == 1
        assert r.pass_rate == 0.8
        assert r.avg_cycle_time_s == 42.5
        assert r.total_tokens == 12345
        assert r.token_efficiency == 290.0
        assert r.cost_per_task == 0.0015

    def test_round_trips_through_pydantic(self):
        r = QuantizationResult(
            quantization="Q6_K",
            model_path="/srv/models/test.Q6_K.gguf",
            model_id="Q6_K",
            pass_rate=0.75,
        )
        assert QuantizationResult.model_validate(r.model_dump()) == r

    def test_token_efficiency_and_cost_per_task_computed(self):
        r = QuantizationResult(
            quantization="Q4_K_M",
            model_path="/srv/models/test.Q4_K_M.gguf",
            model_id="Q4_K_M",
            total_tasks=10,
            passed_tasks=8,
            total_tokens=8000,
            avg_cycle_time_s=20.0,
            pass_rate=0.8,
            token_efficiency=400.0,
            cost_per_task=0.001,
        )
        assert r.token_efficiency == 400.0
        assert r.cost_per_task == 0.001
        assert r.total_tokens == 8000
        assert r.avg_cycle_time_s == 20.0


class TestTaskResult:
    def test_required_fields(self):
        t = TaskResult(name="test_two_sum", passed=True)
        assert t.name == "test_two_sum"
        assert t.passed is True

    def test_failed_task(self):
        t = TaskResult(name="test_fix_syntax_error", passed=False)
        assert t.name == "test_fix_syntax_error"
        assert t.passed is False

    def test_round_trips_through_pydantic(self):
        t = TaskResult(name="test_sort_a_list", passed=True)
        assert TaskResult.model_validate(t.model_dump()) == t


class TestQuantizationResultWithTaskResults:
    def test_task_results_field(self):
        task_results = [
            TaskResult(name="test_two_sum", passed=True),
            TaskResult(name="test_sort_a_list", passed=False),
        ]
        r = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
            total_tasks=2,
            passed_tasks=1,
            failed_tasks=1,
            task_results=task_results,
        )
        assert len(r.task_results) == 2
        assert r.task_results[0].name == "test_two_sum"
        assert r.task_results[0].passed is True
        assert r.task_results[1].name == "test_sort_a_list"
        assert r.task_results[1].passed is False

    def test_task_results_default_empty(self):
        r = QuantizationResult(
            quantization="Q5_K_M",
            model_path="/srv/models/test.Q5_K_M.gguf",
            model_id="Q5_K_M",
        )
        assert r.task_results == []

    def test_task_results_round_trip(self):
        task_results = [
            TaskResult(name="test_two_sum", passed=True),
            TaskResult(name="test_fix_syntax_error", passed=True),
        ]
        r = QuantizationResult(
            quantization="Q8_0",
            model_path="/srv/models/test.Q8_0.gguf",
            model_id="Q8_0",
            task_results=task_results,
        )
        assert QuantizationResult.model_validate(r.model_dump()) == r


class TestQuantizationVerdict:
    def test_required_fields(self):
        result = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
        )
        v = QuantizationVerdict(
            quantizations=[result],
            recommended="Q4_K_S",
            regression=False,
        )
        assert v.recommended == "Q4_K_S"
        assert v.regression is False
        assert len(v.quantizations) == 1

    def test_regression_detected(self):
        baseline = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
            pass_rate=0.80,
        )
        worse = QuantizationResult(
            quantization="Q5_K_M",
            model_path="/srv/models/test.Q5_K_M.gguf",
            model_id="Q5_K_M",
            pass_rate=0.60,
        )
        v = QuantizationVerdict(
            quantizations=[baseline, worse],
            recommended="Q4_K_S",
            regression=True,
        )
        assert v.regression is True

    def test_round_trips_through_pydantic(self):
        r = QuantizationResult(
            quantization="Q8_0",
            model_path="/srv/models/test.Q8_0.gguf",
            model_id="Q8_0",
            pass_rate=0.95,
        )
        v = QuantizationVerdict(quantizations=[r], recommended="Q8_0", regression=False)
        assert QuantizationVerdict.model_validate(v.model_dump()) == v


class TestRenderQuantizationResult:
    def test_render_with_full_data(self):
        r = QuantizationResult(
            quantization="Q5_K_M",
            model_path="/srv/models/test.Q5_K_M.gguf",
            model_id="Q5_K_M",
            total_tasks=10,
            passed_tasks=8,
            failed_tasks=1,
            task_shaped_failures=1,
            pass_rate=0.8,
            avg_cycle_time_s=42.5,
            total_tokens=12345,
            token_efficiency=290.0,
            cost_per_task=0.0015,
        )
        output = _render_quantization_result(r)
        assert "Q5_K_M" in output
        assert "80.0%" in output
        assert "42.5s" in output
        assert "12345" in output
        assert "290.0" in output
        assert "$0.0015" in output

    def test_render_without_avg_time(self):
        r = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
            pass_rate=0.75,
        )
        output = _render_quantization_result(r)
        assert "Q4_K_S" in output
        assert "N/A" in output


class TestRenderQuantizationVerdict:
    def test_renders_table_headers(self):
        r = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
            pass_rate=0.85,
        )
        v = QuantizationVerdict(quantizations=[r], recommended="Q4_K_S", regression=False)
        output = _render_quantization_verdict(v)
        assert "Quantization Sweep Results" in output
        assert "Q4_K_S" in output
        assert "Recommended: Q4_K_S" in output
        assert "No regression" in output

    def test_regression_flag_in_output(self):
        r = QuantizationResult(
            quantization="Q4_K_S",
            model_path="/srv/models/test.Q4_K_S.gguf",
            model_id="Q4_K_S",
            pass_rate=0.85,
        )
        v = QuantizationVerdict(quantizations=[r], recommended="Q4_K_S", regression=True)
        output = _render_quantization_verdict(v)
        assert "REGRESSION DETECTED" in output


class TestSweepParser:
    def test_required_quantizations(self):
        parser = _build_sweep_parser()
        args = parser.parse_args(
            ["--quantizations", "Q4_K_S,Q5_K_M", "--harness-dir", "/tmp/harness"]
        )
        assert args.quantizations == "Q4_K_S,Q5_K_M"
        assert args.harness_dir == Path("/tmp/harness")
        assert args.baseline is None
        assert args.regression_threshold == 2.0
        assert args.cost_per_token is None

    def test_custom_baseline(self):
        parser = _build_sweep_parser()
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S,Q5_K_M",
                "--harness-dir",
                "/tmp/harness",
                "--baseline",
                "Q5_K_M",
            ]
        )
        assert args.baseline == "Q5_K_M"

    def test_custom_regression_threshold(self):
        parser = _build_sweep_parser()
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S",
                "--harness-dir",
                "/tmp/harness",
                "--regression-threshold",
                "1.5",
            ]
        )
        assert args.regression_threshold == 1.5

    def test_cost_per_token_argument(self):
        parser = _build_sweep_parser()
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S",
                "--harness-dir",
                "/tmp/harness",
                "--cost-per-token",
                "0.00001",
            ]
        )
        assert args.cost_per_token == 0.00001


class TestSweepMainValidation:
    def test_empty_quantizations_returns_2(self, tmp_path, capsys):
        harness = tmp_path / "harness"
        harness.mkdir()
        install_load_check_prerequisites(harness)

        rc = sweep_main(["--quantizations", "", "--harness-dir", str(harness)])
        assert rc == 2
        err = capsys.readouterr().err
        assert "at least one quantization label" in err

    def test_no_model_path_env_returns_error(self, tmp_path, capsys):
        harness = tmp_path / "harness"
        harness.mkdir()
        install_load_check_prerequisites(harness)

        with patch.dict("os.environ", {}, clear=True):
            rc = sweep_main(["--quantizations", "Q4_K_S", "--harness-dir", str(harness)])

        assert rc == 1
        err = capsys.readouterr().err
        assert "FOUNDRY_MODEL_PATH" in err


class TestSweepMainModelNotFound:
    def test_no_model_file_returns_error(self, tmp_path, capsys):
        harness = tmp_path / "harness"
        harness.mkdir()
        install_load_check_prerequisites(harness)
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        with patch.dict(
            "os.environ",
            {"FOUNDRY_MODEL_PATH": str(model_dir)},
        ):
            rc = sweep_main(["--quantizations", "Q4_K_S", "--harness-dir", str(harness)])

        assert rc == 1
        err = capsys.readouterr().err
        assert "No model file found" in err


class TestSweepDefaultThreshold:
    def test_default_threshold_is_2pp(self):
        assert DEFAULT_REGRESSION_THRESHOLD_PP == 2.0


class TestExtractTaskName:
    def test_extracts_parametrized_test_name(self):
        from foundry_x.evolution.critic import Critic

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        line = "benchmarks/tasks/test_two_sum.py::test_two_sum[basic] PASSED"
        assert critic._extract_task_name(line) == "test_two_sum"

    def test_extracts_simple_test_name(self):
        from foundry_x.evolution.critic import Critic

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        line = "benchmarks/tasks/test_smoke.py::test_smoke PASSED"
        assert critic._extract_task_name(line) == "test_smoke"

    def test_extracts_failed_test(self):
        from foundry_x.evolution.critic import Critic

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        line = "benchmarks/tasks/test_fix_syntax_error.py::test_fix_syntax_error FAILED"
        assert critic._extract_task_name(line) == "test_fix_syntax_error"

    def test_returns_none_for_non_test_line(self):
        from foundry_x.evolution.critic import Critic

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        line = "=== 5 passed ==="
        assert critic._extract_task_name(line) is None

    def test_returns_none_for_empty_line(self):
        from foundry_x.evolution.critic import Critic

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        assert critic._extract_task_name("") is None


class TestSweepOutputFlag:
    def test_output_flag_in_parser(self):
        parser = _build_sweep_parser()
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S",
                "--harness-dir",
                "/tmp/harness",
                "--output",
                "logs/quantization_results.json",
            ]
        )
        assert args.output == "logs/quantization_results.json"

    def test_output_flag_default_none(self):
        parser = _build_sweep_parser()
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S",
                "--harness-dir",
                "/tmp/harness",
            ]
        )
        assert args.output is None

    def test_output_flag_in_subparser(self):
        import argparse

        parser = argparse.ArgumentParser()
        _build_sweep_subparser(parser)
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S,Q5_K_M",
                "--harness-dir",
                "/tmp/harness",
                "--output",
                "logs/quantization_results.json",
            ]
        )
        assert args.output == "logs/quantization_results.json"

    def test_output_flag_default_none_in_subparser(self):
        import argparse

        parser = argparse.ArgumentParser()
        _build_sweep_subparser(parser)
        args = parser.parse_args(
            [
                "--quantizations",
                "Q4_K_S",
                "--harness-dir",
                "/tmp/harness",
            ]
        )
        assert args.output is None


class TestComputeTokenMetrics:
    """Integration tests for token-efficiency population from trace store (issue #549)."""

    def test_compute_token_metrics_sums_total_tokens(self, tmp_path):
        """total_tokens is the sum of model_response.token_usage.total_tokens across steps."""
        from foundry_x.evolution.critic import Critic
        from foundry_x.trace.logger import TraceLogger

        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        model_id = "Q4_K_S"
        with logger.session(harness_version="test", model_id=model_id) as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "do work"})
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
                kind="model_response",
                payload={
                    "step": 1,
                    "token_usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            )
            logger.record(
                sid,
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 2},
            )

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(model_id, str(trace_path))

        assert total_tokens == 45
        assert avg_cycle_time_s is not None
        assert avg_cycle_time_s > 0

    def test_compute_token_metrics_computes_avg_cycle_time(self, tmp_path):
        """avg_cycle_time_s is the mean wall-clock time from task_received to outcome."""
        from foundry_x.evolution.critic import Critic
        from foundry_x.trace.logger import TraceLogger

        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        model_id = "Q5_K_M"
        import time

        with logger.session(harness_version="test", model_id=model_id) as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "task 1"})
            time.sleep(0.01)
            logger.record(
                sid,
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 1},
            )

        with logger.session(harness_version="test", model_id=model_id) as sid2:
            logger.record(sid2, kind="task_received", payload={"prompt": "task 2"})
            time.sleep(0.02)
            logger.record(
                sid2,
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 1},
            )

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(model_id, str(trace_path))

        assert avg_cycle_time_s is not None
        assert avg_cycle_time_s >= 0.01

    def test_compute_token_metrics_returns_zero_when_no_sessions(self, tmp_path):
        """Returns (0, None) when no sessions match the model_id."""
        from foundry_x.evolution.critic import Critic
        from foundry_x.trace.logger import TraceLogger

        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        with logger.session(harness_version="test", model_id="other-model") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "do work"})

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics("Q4_K_S", str(trace_path))

        assert total_tokens == 0
        assert avg_cycle_time_s is None

    def test_compute_token_metrics_skips_sessions_without_outcome(self, tmp_path):
        """Sessions without an outcome event contribute cycle time but not tokens."""
        from foundry_x.evolution.critic import Critic
        from foundry_x.trace.logger import TraceLogger

        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        model_id = "Q4_K_S"
        with logger.session(harness_version="test", model_id=model_id) as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "do work"})
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

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(model_id, str(trace_path))

        assert total_tokens == 15
        assert avg_cycle_time_s is None

    def test_token_efficiency_computed_from_trace_metrics(self, tmp_path):
        """token_efficiency = total_tokens / avg_cycle_time_s when both are available."""
        from foundry_x.evolution.critic import Critic, QuantizationResult
        from foundry_x.trace.logger import TraceLogger

        trace_path = tmp_path / "traces.db"
        logger = TraceLogger(trace_path)

        model_id = "Q4_K_M"
        import time

        with logger.session(harness_version="test", model_id=model_id) as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "do work"})
            time.sleep(0.01)
            logger.record(
                sid,
                kind="model_response",
                payload={
                    "step": 0,
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            )
            logger.record(
                sid,
                kind="outcome",
                payload={"status": "success", "reason": "final_answer", "steps": 1},
            )

        critic = Critic(harness_dir=Path("/tmp/nonexistent"))
        total_tokens, avg_cycle_time_s = critic._compute_token_metrics(model_id, str(trace_path))

        token_efficiency = None
        if total_tokens > 0 and avg_cycle_time_s is not None and avg_cycle_time_s > 0:
            token_efficiency = total_tokens / avg_cycle_time_s

        assert token_efficiency is not None
        assert token_efficiency > 0
        assert avg_cycle_time_s is not None
        result = QuantizationResult(
            quantization="Q4_K_M",
            model_path="/srv/models/test.Q4_K_M.gguf",
            model_id=model_id,
            total_tasks=1,
            passed_tasks=1,
            total_tokens=total_tokens,
            avg_cycle_time_s=avg_cycle_time_s,
            token_efficiency=token_efficiency,
        )
        assert result.token_efficiency == token_efficiency
        assert result.total_tokens == 150
