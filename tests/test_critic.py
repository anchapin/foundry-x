"""Sandbox diff-apply and pytest-gate tests for the Critic (issue #16).

Acceptance per issue #16 / ADR-0004:

- A clean harness fixture with a no-op diff yields ``approved=True``.
- A diff that breaks a test in the fixture yields ``approved=False`` with the
  failing check (``"pytest"``) named.
- The live ``harness_dir`` is byte-identical before and after ``evaluate``
  (asserted via a directory hash).
- A patch that does not apply cleanly is rejected before pytest runs.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path

import pytest

from foundry_x.evolution.critic import Critic, CriticVerdict

_SANITY_TEST = """\
def test_pass():
    assert True
"""

_SYSTEM_PROMPT = "You are a helpful agent.\n"


def _make_harness(root: Path) -> Path:
    """Create a minimal harness fixture with a single passing test."""
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    (root / "system_prompt.txt").write_text(_SYSTEM_PROMPT)
    (tests_dir / "test_sanity.py").write_text(_SANITY_TEST)
    return root


def _make_diff(path: str, old: str, new: str) -> str:
    """Produce a ``git apply``-compatible unified diff for *path*."""
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _hash_dir(path: Path) -> str:
    """Deterministic SHA-256 over every file's relative path + bytes."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


@pytest.fixture()
def harness_dir(tmp_path: Path) -> Path:
    return _make_harness(tmp_path / "harness")


def test_noop_diff_on_clean_harness_approves(harness_dir: Path) -> None:
    critic = Critic(harness_dir, pytest_args=["-q", "tests/test_sanity.py"])
    verdict = critic.evaluate("")
    assert verdict.approved is True
    assert "pytest" in verdict.passed_checks
    assert verdict.failed_checks == []


def test_diff_that_breaks_test_is_rejected(harness_dir: Path) -> None:
    breaking = _make_diff(
        "tests/test_sanity.py",
        _SANITY_TEST,
        _SANITY_TEST.replace("assert True", "assert False"),
    )
    critic = Critic(harness_dir, pytest_args=["-q", "tests/test_sanity.py"])
    verdict = critic.evaluate(breaking)
    assert verdict.approved is False
    assert "pytest" in verdict.failed_checks
    assert verdict.notes


def test_live_harness_is_byte_identical_after_evaluate(harness_dir: Path) -> None:
    breaking = _make_diff(
        "tests/test_sanity.py",
        _SANITY_TEST,
        _SANITY_TEST.replace("assert True", "assert False"),
    )
    critic = Critic(harness_dir, pytest_args=["-q", "tests/test_sanity.py"])
    before = _hash_dir(harness_dir)
    critic.evaluate(breaking)
    assert _hash_dir(harness_dir) == before


def test_patch_that_does_not_apply_is_rejected(harness_dir: Path) -> None:
    bad_diff = _make_diff(
        "tests/test_sanity.py",
        "this content does not exist\n",
        "replacement\n",
    )
    critic = Critic(harness_dir, pytest_args=["-q", "tests/test_sanity.py"])
    verdict = critic.evaluate(bad_diff)
    assert verdict.approved is False
    assert "git apply" in verdict.failed_checks
    assert "pytest" not in verdict.passed_checks


def test_clean_diff_that_passes_is_approved(harness_dir: Path) -> None:
    clean_diff = _make_diff(
        "system_prompt.txt",
        _SYSTEM_PROMPT,
        "You are an excellent agent.\n",
    )
    critic = Critic(harness_dir, pytest_args=["-q", "tests/test_sanity.py"])
    verdict = critic.evaluate(clean_diff)
    assert verdict.approved is True
    assert "git apply" in verdict.passed_checks
    assert "pytest" in verdict.passed_checks
    assert (harness_dir / "system_prompt.txt").read_text() == _SYSTEM_PROMPT


def test_verdict_round_trips_through_pydantic() -> None:
    v = CriticVerdict(approved=True, passed_checks=["pytest"], notes="ok")
    assert CriticVerdict.model_validate(v.model_dump()) == v


def test_default_pytest_args_target_smoke_suite() -> None:
    """Default pytest selection is ``-q -m benchmark`` (issue #185).

    Updated from the original ``tests/test_smoke.py`` default now that the
    in-process ``BenchmarkTask`` registry (issue #108) and the security-evals
    family (ADR-0009) are wired. The Critic gate runs the full benchmark
    suite by default so a harness edit that breaks any
    ``@pytest.mark.benchmark`` task is caught.
    """
    critic = Critic(Path("/tmp/nonexistent"))
    assert critic.pytest_args == ["-q", "-m", "benchmark"]


def test_critic_exposes_benchmark_tasks_from_registry() -> None:
    """``Critic.benchmark_tasks`` lazy-loads from the in-process registry (issue #108).

    The Critic can now enumerate benchmark tasks without spawning pytest.
    The property is lazy: importing ``foundry_x.evolution.critic`` does
    not eagerly load every task module, and tests that never touch
    ``benchmark_tasks`` (the existing ``test_*`` cases above) pay no cost.
    """
    from benchmarks.models import BenchmarkTask

    critic = Critic(Path("/tmp/nonexistent"))
    tasks = critic.benchmark_tasks
    assert tasks, "registry must return at least one benchmark task"
    assert all(isinstance(t, BenchmarkTask) for t in tasks)
    # Cached: second access returns the same list object.
    assert critic.benchmark_tasks is tasks


def test_critic_accepts_pre_seeded_benchmark_tasks() -> None:
    """``Critic(benchmark_tasks=[...])`` skips the registry lookup.

    Tests that want full control over the task list (or want to avoid
    touching the registry at all) can pre-seed it in the constructor.
    The cached list is returned verbatim on every property access.
    """
    from benchmarks.models import BenchmarkTask

    sentinel = BenchmarkTask(name="sentinel", description="test fixture")
    critic = Critic(Path("/tmp/nonexistent"), benchmark_tasks=[sentinel])
    assert critic.benchmark_tasks == [sentinel]
    assert critic.benchmark_tasks is critic.benchmark_tasks  # cached


def test_default_pytest_args_derived_from_benchmark_tasks() -> None:
    """Default pytest selection is derived from the benchmark registry (issue #185).

    With the in-process ``BenchmarkTask`` registry shipping (issue #108) and
    the security-evals family wired through ``tags=['security']`` (ADR-0009),
    the Critic gate defaults to ``-q -m benchmark`` — running the full
    benchmark suite rather than just ``tests/test_smoke.py``. This means a
    harness edit that breaks any ``@pytest.mark.benchmark`` task is caught at
    the gate, and the verdict records which benchmark tags were covered.
    """
    critic = Critic(Path("/tmp/nonexistent"))
    assert critic.pytest_args == ["-q", "-m", "benchmark"]


def test_passed_checks_list_benchmark_tags(harness_dir: Path) -> None:
    """A successful pytest run lists every benchmark tag in ``passed_checks`` (issue #185).

    Pre-seeded tasks with known tags let us assert the exact tag set the
    verdict reports, without depending on the live registry contents.
    Duplicate tags across tasks are de-duplicated and sorted alphabetically.
    """
    from benchmarks.models import BenchmarkTask

    critic = Critic(
        harness_dir,
        pytest_args=["-q", "tests/test_sanity.py"],
        benchmark_tasks=[
            BenchmarkTask(
                name="t1",
                description="d",
                tags=["security", "smoke"],
            ),
            BenchmarkTask(
                name="t2",
                description="d",
                tags=["security", "math"],
            ),
        ],
    )
    verdict = critic.evaluate("")
    assert verdict.approved is True
    assert "pytest" in verdict.passed_checks
    assert "benchmark:security" in verdict.passed_checks
    assert "benchmark:smoke" in verdict.passed_checks
    assert "benchmark:math" in verdict.passed_checks
