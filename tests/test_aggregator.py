"""Tests for the per-config aggregator (issue #1035, ADR-0023).

Covers the three acceptance criteria from the issue:
- Happy path: multiple configs produce paired observations.
- Underpowered: fewer than 30 configs raises ``UnderpoweredStudyError``
  when passed to ``pearson_binary``.
- Zero-variance: all configs producing the same internal rate.
- Config-label mismatch in ``apply_external_rates``.
"""

from __future__ import annotations

import time

import pytest

from foundry_x.evaluation.aggregator import (
    aggregate_per_config,
    apply_external_rates,
)
from foundry_x.evaluation.correlation import (
    UnderpoweredStudyError,
    ZeroVarianceError,
    pearson_binary,
)
from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceLogger


def _seed_config_session(
    logger: TraceLogger,
    quantization: str,
    harness_version: str,
    *,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
) -> str:
    """Create an internal_suite session with a critic_verdict for a config."""
    with logger.session(
        harness_version=harness_version,
        quantization=quantization,
        metadata={"study_run_type": "internal_suite"},
    ) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "test"})
        time.sleep(0.005)
        record_verdict(
            logger,
            sid,
            CriticVerdict(
                verdict=len(passed_checks or []) > len(failed_checks or []),
                passed_checks=passed_checks or [],
                failed_checks=failed_checks or [],
            ),
        )
    return sid


class TestAggregatePerConfig:
    """Tests for aggregate_per_config."""

    def test_happy_path(self, tmp_path):
        """Multiple configs produce paired observations."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a", "b", "c"],
            failed_checks=["d"],
        )
        _seed_config_session(
            logger,
            "Q5_K_M",
            "v1",
            passed_checks=["a", "b"],
            failed_checks=["c", "d"],
        )
        _seed_config_session(
            logger,
            "Q8_0",
            "v2",
            passed_checks=["a"],
            failed_checks=["b", "c", "d"],
        )

        result = aggregate_per_config(logger)

        assert len(result.labels) == 3
        assert len(result.internal_rates) == 3
        assert len(result.external_rates) == 3
        assert len(result.observations) == 3

        # All external rates are 0.0 (placeholder).
        assert all(r == 0.0 for r in result.external_rates)

        # Labels are sorted.
        assert result.labels == sorted(result.labels)

        # Check individual rates.
        rates_by_label = {obs.label: obs.internal_rate for obs in result.observations}
        assert rates_by_label["Q4_K_M/v1"] == pytest.approx(0.75)  # 3/4
        assert rates_by_label["Q5_K_M/v1"] == pytest.approx(0.5)  # 2/4
        assert rates_by_label["Q8_0/v2"] == pytest.approx(0.25)  # 1/4

    def test_skips_non_internal_sessions(self, tmp_path):
        """Sessions without study_run_type=internal_suite are skipped."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        # This session is NOT internal_suite.
        with logger.session(harness_version="v1", quantization="Q4_K_M") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "test"})
            record_verdict(
                logger,
                sid,
                CriticVerdict(verdict=True, passed_checks=["a"]),
            )

        # This one IS.
        _seed_config_session(
            logger,
            "Q5_K_M",
            "v1",
            passed_checks=["a", "b"],
        )

        result = aggregate_per_config(logger)

        assert len(result.labels) == 1
        assert result.labels[0] == "Q5_K_M/v1"

    def test_most_recent_session_per_config(self, tmp_path):
        """When multiple sessions exist for a config, the most recent wins."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        # Older session with different pass rate.
        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a"],
            failed_checks=["b", "c", "d"],
        )
        time.sleep(0.01)
        # Newer session with different pass rate.
        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a", "b", "c"],
            failed_checks=["d"],
        )

        result = aggregate_per_config(logger)

        assert len(result.labels) == 1
        # The newer session should be used (3/4 = 0.75).
        assert result.observations[0].internal_rate == pytest.approx(0.75)

    def test_empty_raises(self, tmp_path):
        """No matching sessions raises ValueError."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        with pytest.raises(ValueError, match="no configurations"):
            aggregate_per_config(logger)

    def test_harness_version_filter(self, tmp_path):
        """Filtering by harness_version works."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a", "b"],
        )
        _seed_config_session(
            logger,
            "Q5_K_M",
            "v2",
            passed_checks=["a", "b", "c"],
        )

        result = aggregate_per_config(logger, harness_version="v1")

        assert len(result.labels) == 1
        assert result.labels[0] == "Q4_K_M/v1"

    def test_sessions_without_quantization_are_skipped(self, tmp_path):
        """Sessions with no quantization are skipped."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        # No quantization set.
        with logger.session(harness_version="v1") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "test"})
            record_verdict(
                logger,
                sid,
                CriticVerdict(verdict=True, passed_checks=["a"]),
            )

        # This one has quantization.
        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a", "b"],
        )

        result = aggregate_per_config(logger)

        assert len(result.labels) == 1
        assert result.labels[0] == "Q4_K_M/v1"

    def test_zero_verdicts_gives_zero_rate(self, tmp_path):
        """A config with no passed or failed checks gives 0.0 rate."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        with logger.session(
            harness_version="v1",
            quantization="Q4_K_M",
            metadata={"study_run_type": "internal_suite"},
        ) as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "test"})

        result = aggregate_per_config(logger)

        assert len(result.labels) == 1
        assert result.observations[0].internal_rate == 0.0


class TestApplyExternalRates:
    """Tests for apply_external_rates."""

    def test_happy_path(self, tmp_path):
        """External rates are correctly applied."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a", "b"],
        )
        _seed_config_session(
            logger,
            "Q5_K_M",
            "v1",
            passed_checks=["a"],
            failed_checks=["b"],
        )

        result = aggregate_per_config(logger)
        updated = apply_external_rates(
            result,
            {"Q4_K_M/v1": 0.8, "Q5_K_M/v1": 0.6},
        )

        assert updated.external_rates == [0.8, 0.6]
        # Internal rates unchanged.
        assert updated.internal_rates == result.internal_rates
        # Observations updated.
        for obs in updated.observations:
            assert obs.external_rate > 0.0

    def test_missing_label_raises(self, tmp_path):
        """Missing label in external_rates raises KeyError."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        _seed_config_session(
            logger,
            "Q4_K_M",
            "v1",
            passed_checks=["a"],
        )

        result = aggregate_per_config(logger)

        with pytest.raises(KeyError, match="missing label"):
            apply_external_rates(result, {"nonexistent/v1": 0.5})


class TestPearsonWithAggregator:
    """Integration: aggregate + pearson_binary on real trace data."""

    def test_small_study_raises_underpowered(self, tmp_path):
        """A study with < 30 configs raises UnderpoweredStudyError."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        for i in range(10):
            _seed_config_session(
                logger,
                f"Q{i}_K_M",
                "v1",
                passed_checks=[f"task_{j}" for j in range(i + 1)],
                failed_checks=[f"task_fail_{j}" for j in range(10 - i)],
            )

        result = aggregate_per_config(logger)
        # Fake external rates with some correlation.
        external = {label: rate * 0.9 for label, rate in zip(result.labels, result.internal_rates)}
        updated = apply_external_rates(result, external)

        with pytest.raises(UnderpoweredStudyError):
            pearson_binary(updated.internal_rates, updated.external_rates)

    def test_zero_variance_raises(self, tmp_path):
        """All configs with identical pass rates raise ZeroVarianceError."""
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)

        for i in range(35):
            _seed_config_session(
                logger,
                f"Q{i}_K_M",
                "v1",
                passed_checks=["a", "b"],
                failed_checks=["c", "d"],
            )

        result = aggregate_per_config(logger)
        # External rates have variance.
        external = {label: (i % 2) * 0.5 for i, label in enumerate(result.labels)}
        updated = apply_external_rates(result, external)

        with pytest.raises(ZeroVarianceError, match="internal_rates"):
            pearson_binary(updated.internal_rates, updated.external_rates)
