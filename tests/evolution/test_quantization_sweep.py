"""Tests for the quantization sweep feature (issue #464 / ADR-0016)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from foundry_x.evolution.critic import (
    QuantizationResult,
    QuantizationVerdict,
    DEFAULT_REGRESSION_THRESHOLD_PP,
)
from foundry_x.evolution.cli import (
    _build_sweep_parser,
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
        )
        assert r.total_tasks == 10
        assert r.passed_tasks == 8
        assert r.failed_tasks == 1
        assert r.task_shaped_failures == 1
        assert r.pass_rate == 0.8
        assert r.avg_cycle_time_s == 42.5
        assert r.total_tokens == 12345

    def test_round_trips_through_pydantic(self):
        r = QuantizationResult(
            quantization="Q6_K",
            model_path="/srv/models/test.Q6_K.gguf",
            model_id="Q6_K",
            pass_rate=0.75,
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
        )
        output = _render_quantization_result(r)
        assert "Q5_K_M" in output
        assert "80.0%" in output
        assert "42.5s" in output
        assert "12345" in output

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
