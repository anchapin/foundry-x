"""Sandbox diff-apply and pytest-gate tests for the Critic (issue #16).

Acceptance per issue #16 / ADR-0004:

- A clean harness fixture with a no-op diff yields ``approved=True``.
- A diff that breaks a test in the fixture yields ``approved=False`` with the
  failing check (``"pytest"``) named.
- The live ``harness_dir`` is byte-identical before and after ``evaluate``
  (asserted via a directory hash).
- A patch that does not apply cleanly is rejected before pytest runs.

Issue #186 / ADR-0004 step 3 extends this contract: the Critic rejects
``evaluate()`` results that flip a previously-passing benchmark task to
failing (``failed_checks=['regression:<task_name>']``). The contract is
exercised by ``test_regression_baseline_rejects_flip`` plus the supporting
baseline-persistence tests below.
"""

from __future__ import annotations

import difflib
import hashlib
import subprocess
from pathlib import Path

import pytest

from foundry_x.evolution.critic import Critic, CriticVerdict
from tests._harness_fixture import install_load_check_prerequisites

_SANITY_TEST = """\
def test_pass():
    assert True
"""

_SYSTEM_PROMPT = "You are a helpful agent.\n"


def _make_harness(root: Path) -> Path:
    """Create a minimal harness fixture with a single passing test.

    Includes the load_check prerequisites (issue #187) so the harness is
    load-check-compliant: the Critic now runs ``scripts/load_check.py``
    against the sandbox before pytest.
    """
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    (root / "system_prompt.txt").write_text(_SYSTEM_PROMPT)
    (tests_dir / "test_sanity.py").write_text(_SANITY_TEST)
    install_load_check_prerequisites(root)
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
    # Pre-seed an empty benchmark task list so passed_checks stays
    # deterministic — HEAD (issue #185) appends ``benchmark:<tag>``
    # entries derived from the registry, but this test asserts the
    # load_check precondition ordering only (issue #187).
    critic = Critic(
        harness_dir,
        pytest_args=["-q", "tests/test_sanity.py"],
        benchmark_tasks=[],
    )
    verdict = critic.evaluate(clean_diff)
    assert verdict.approved is True
    # Issue #187: passed_checks order is git apply -> load_check -> pytest.
    assert verdict.passed_checks == ["git apply", "load_check", "pytest"]
    assert (harness_dir / "system_prompt.txt").read_text() == _SYSTEM_PROMPT


def test_load_check_failure_rejected_pre_pytest(harness_dir: Path) -> None:
    """A harness tree load_check rejects (issue #187) must fail the gate
    *before* pytest runs: ``failed_checks == ["load_check"]`` and pytest
    appears in neither passed nor failed checks."""
    # Inject a broken skill so harness/scripts/load_check.py exits 1.
    (harness_dir / "skills" / "broken.json").write_text("{ not valid json")
    critic = Critic(harness_dir, pytest_args=["-q", "tests/test_sanity.py"])
    verdict = critic.evaluate("")
    assert verdict.approved is False
    assert verdict.failed_checks == ["load_check"]
    assert "pytest" not in verdict.passed_checks
    assert "pytest" not in verdict.failed_checks
    assert "broken.json" in verdict.notes


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


def test_default_gate_timeout_matches_issue_spec() -> None:
    """The default ``gate_timeout_s`` matches issue #188's literal (``300``).

    Issue #188 names a wall-clock cap so a hanging benchmark or a proposed
    harness edit that re-introduces an infinite loop in a hook cannot make
    the Critic itself a runaway (docs/SECURITY.md "Runaway detection",
    ADR-0004 §Consequences). The Critic carries its own constant rather
    than mirroring the Runner's ``DEFAULT_TASK_TIMEOUT_S`` so a regression
    in either cap surfaces independently.
    """
    critic = Critic(Path("/tmp/nonexistent"))
    assert critic.gate_timeout_s == DEFAULT_GATE_TIMEOUT_S == 300


def test_custom_gate_timeout_is_preserved() -> None:
    """``gate_timeout_s=...`` overrides the default verbatim."""
    critic = Critic(Path("/tmp/nonexistent"), gate_timeout_s=42)
    assert critic.gate_timeout_s == 42


def test_pytest_exceeds_timeout_rejected(harness_dir: Path) -> None:
    """Critic rejects when pytest exceeds ``gate_timeout_s`` (issue #188).

    The harness is extended with a single test that sleeps longer than
    the cap. ``evaluate`` must abort the run, return
    ``approved=False``, surface ``"pytest:timeout"`` in ``failed_checks``,
    and leave ``notes`` truthy so an engineer skimming the trace can see
    why the gate tripped.
    """
    hanging_test = "import time\n" "\n" "def test_hang():\n" "    time.sleep(10)\n"
    (harness_dir / "tests" / "test_hang.py").write_text(hanging_test)
    critic = Critic(
        harness_dir,
        pytest_args=["-q", "tests/test_hang.py"],
        gate_timeout_s=1,
    )
    verdict = critic.evaluate("")
    assert verdict.approved is False
    assert "pytest:timeout" in verdict.failed_checks
    assert "pytest" not in verdict.passed_checks
    assert verdict.notes, "timeout verdict must carry diagnostic notes"


def test_pytest_within_timeout_still_approved(harness_dir: Path) -> None:
    """A pytest run that finishes before the cap is approved unchanged (issue #188).

    Guards against a regression where the new timeout machinery accidentally
    aborts well-behaved runs (e.g. by misreading the cap as zero, by
    catching the wrong exception, or by corrupting the ``passed_checks`` list).
    """
    critic = Critic(
        harness_dir,
        pytest_args=["-q", "tests/test_sanity.py"],
        gate_timeout_s=30,
    )
    verdict = critic.evaluate("")
    assert verdict.approved is True
    assert "pytest" in verdict.passed_checks
    assert "pytest:timeout" not in verdict.failed_checks


def test_git_apply_exceeds_timeout_rejected(harness_dir: Path) -> None:
    """A ``git apply`` that exceeds the cap is rejected with ``:timeout`` (issue #188).

    ``git apply`` is overwhelmingly fast for any realistic harness diff, so
    we drive the timeout path with a 1-second cap and a no-op diff applied
    through ``subprocess.run`` mock — the live code path we cover is the
    ``except subprocess.TimeoutExpired`` branch in ``evaluate`` and its
    verdict shape (``failed_checks=["git apply:timeout"]``,
    ``approved=False``, non-empty ``notes``).
    """
    from unittest.mock import patch

    critic = Critic(
        harness_dir,
        pytest_args=["-q", "tests/test_sanity.py"],
        gate_timeout_s=1,
    )
    with patch(
        "foundry_x.evolution.critic.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["git", "apply"], timeout=1),
    ):
        verdict = critic.evaluate("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n")
    assert verdict.approved is False
    assert "git apply:timeout" in verdict.failed_checks
    assert verdict.notes


# ---------------------------------------------------------------------------
# Regression baseline (issue #186 / ADR-0004 step 3)
# ---------------------------------------------------------------------------


_BASELINE_PASSSOURCE = (
    "def test_task_alpha():\n"
    "    assert True\n"
    "\n"
    "def test_task_beta():\n"
    "    assert True\n"
)


def _make_baseline_harness(root: Path, source: str = _BASELINE_PASSSOURCE) -> Path:
    """Create a harness fixture with two passing pytest tests for baseline tests."""
    harness_dir = root / "harness"
    tests_dir = harness_dir / "tests"
    tests_dir.mkdir(parents=True)
    (harness_dir / "system_prompt.txt").write_text(_SYSTEM_PROMPT)
    (tests_dir / "test_baseline_regression.py").write_text(source)
    return harness_dir


def test_regression_baseline_rejects_flip(tmp_path: Path) -> None:
    """Issue #186 acceptance: a flip of a previously-passing task rejects the gate.

    Workflow:

    1. ``evaluate()`` is called twice on the same ``Critic``.
    2. The first call writes ``logs/critic_baseline.json`` with both
       benchmark tasks marked passing.
    3. The second call applies a diff that flips ``task_alpha`` to failing.
    4. The verdict is ``approved=False`` and ``failed_checks`` carries
       ``"regression:task_alpha"`` (not ``task_beta``).
    """
    harness_dir = _make_baseline_harness(tmp_path)
    baseline_path = tmp_path / "critic_baseline.json"

    task_alpha = BenchmarkTask(name="task_alpha", description="synthetic baseline task A")
    task_beta = BenchmarkTask(name="task_beta", description="synthetic baseline task B")
    critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        benchmark_tasks=[task_alpha, task_beta],
        baseline_path=baseline_path,
    )

    first_verdict = critic.evaluate("")
    assert first_verdict.approved is True
    assert baseline_path.exists()
    baseline_payload = CriticBaseline.model_validate_json(baseline_path.read_text())
    assert baseline_payload.entries["task_alpha"].passing is True
    assert baseline_payload.entries["task_beta"].passing is True

    # Flip only ``test_task_alpha`` to failing; leave ``test_task_beta`` alone.
    flipped_source = _BASELINE_PASSSOURCE.replace(
        "test_task_alpha():\n    assert True",
        "test_task_alpha():\n    assert False",
        1,
    )
    flip_diff = _make_diff(
        "tests/test_baseline_regression.py",
        _BASELINE_PASSSOURCE,
        flipped_source,
    )

    second_verdict = critic.evaluate(flip_diff)
    assert second_verdict.approved is False
    assert "regression:task_alpha" in second_verdict.failed_checks
    assert "regression:task_beta" not in second_verdict.failed_checks
    assert "pytest" in second_verdict.failed_checks


def test_regression_baseline_first_call_writes_file(tmp_path: Path) -> None:
    """First ``evaluate()`` after the Critic lands writes the baseline JSON (issue #186).

    The Critic pins the convention that an opt-in ``baseline_path`` (or the
    default ``logs/critic_baseline.json`` for project-level invocations)
    persists a baseline on the first observation so the second call has
    something to diff against.
    """
    harness_dir = _make_baseline_harness(tmp_path)
    baseline_path = tmp_path / "critic_baseline.json"
    assert not baseline_path.exists()

    task_alpha = BenchmarkTask(name="task_alpha", description="synthetic")
    critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        benchmark_tasks=[task_alpha],
        baseline_path=baseline_path,
    )

    assert critic.evaluate("").approved is True
    assert baseline_path.exists()

    payload = CriticBaseline.model_validate_json(baseline_path.read_text())
    assert payload.schema_version == 1
    assert payload.entries["task_alpha"].passing is True


def test_regression_baseline_persists_across_critic_instances(tmp_path: Path) -> None:
    """A second ``Critic`` pointing at the same baseline path inherits the prior state.

    Confirms the persistence contract: the baseline file on disk is the
    source of truth, so a fresh ``Critic`` (e.g. across CLI invocations)
    picks up the previously recorded pass/fail snapshot rather than
    silently starting from empty.
    """
    harness_dir = _make_baseline_harness(tmp_path)
    baseline_path = tmp_path / "critic_baseline.json"

    first_task = BenchmarkTask(name="task_alpha", description="first")
    first_critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        benchmark_tasks=[first_task],
        baseline_path=baseline_path,
    )
    assert first_critic.evaluate("").approved is True

    # A second Critic constructed fresh -- same baseline_path, new
    # in-process object -- sees the prior baseline recorded on disk.
    second_critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        benchmark_tasks=[first_task],
        baseline_path=baseline_path,
    )

    flipped_source = _BASELINE_PASSSOURCE.replace(
        "test_task_alpha():\n    assert True",
        "test_task_alpha():\n    assert False",
        1,
    )
    flip_diff = _make_diff(
        "tests/test_baseline_regression.py",
        _BASELINE_PASSSOURCE,
        flipped_source,
    )
    flipped_verdict = second_critic.evaluate(flip_diff)

    assert flipped_verdict.approved is False
    assert "regression:task_alpha" in flipped_verdict.failed_checks


def test_regression_baseline_skips_unknown_tasks(tmp_path: Path) -> None:
    """Tasks whose pytest test was not observed this run are left out of the baseline.

    A benchmark task whose ``test_<name>`` was not collected (e.g. because
    ``pytest_args`` filtered it out) has no observation either way: the
    Critic writes no opinion on it, and a later regression cannot be
    attributed to a task the gate never saw pass. This guards against a
    silent ``False`` in the baseline that would otherwise mis-attribute
    later flips.
    """
    harness_dir = _make_baseline_harness(tmp_path)
    baseline_path = tmp_path / "critic_baseline.json"

    # ``task_gamma`` is registered but no pytest test with that name exists.
    observed_task = BenchmarkTask(name="task_alpha", description="observed")
    unobserved_task = BenchmarkTask(name="task_gamma", description="unobserved")
    critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        benchmark_tasks=[observed_task, unobserved_task],
        baseline_path=baseline_path,
    )

    assert critic.evaluate("").approved is True
    payload = CriticBaseline.model_validate_json(baseline_path.read_text())
    # task_alpha is recorded; task_gamma is omitted.
    assert "task_alpha" in payload.entries
    assert "task_gamma" not in payload.entries


def test_regression_baseline_off_when_path_is_none(tmp_path: Path) -> None:
    """``baseline_path=None`` disables the regression gate and skips persistence.

    The pre-#186 call shape (``Critic(harness_dir=..., pytest_args=...)``
    with no baseline wiring) keeps behaving exactly as it always did:
    a diff that breaks a test still yields ``approved=False`` with
    ``pytest`` in ``failed_checks``, but no ``regression:*`` entries are
    added because no baseline file is written or read.
    """
    harness_dir = _make_baseline_harness(tmp_path)
    critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        baseline_path=None,
    )

    flipped_source = _BASELINE_PASSSOURCE.replace(
        "test_task_alpha():\n    assert True",
        "test_task_alpha():\n    assert False",
        1,
    )
    flip_diff = _make_diff(
        "tests/test_baseline_regression.py",
        _BASELINE_PASSSOURCE,
        flipped_source,
    )
    verdict = critic.evaluate(flip_diff)

    assert verdict.approved is False
    assert "pytest" in verdict.failed_checks
    assert not any(c.startswith("regression:") for c in verdict.failed_checks)
    # And no baseline JSON was written. The harness fixture file tree is
    # expected under tmp_path; only JSON artifacts of the regression gate
    # must be absent.
    assert not any(p.suffix == ".json" for p in tmp_path.rglob("*"))


def test_critic_default_baseline_path_is_logs_critic_baseline_json() -> None:
    """The constructor default points at ``logs/critic_baseline.json`` (issue #186).

    This is the canonical home for the regression baseline (ADR-0004
    step 3) and is what a project-level Critic invocation lands on when
    no explicit path is passed.
    """
    critic = Critic(Path("/tmp/nonexistent"))
    assert critic.baseline_path == DEFAULT_BASELINE_PATH
    assert critic.baseline_path == Path("logs") / "critic_baseline.json"


def test_regression_baseline_does_not_regress_when_never_passed(tmp_path: Path) -> None:
    """A task that has *never* been recorded as passing cannot regress (issue #186).

    The gate fires only on a previously-passing → now-failing flip. A task
    that fails on first observation (and therefore was never recorded as
    passing) is absent from the prior baseline; its next failure is just
    a baseline write, never a regression.
    """
    harness_dir = _make_baseline_harness(tmp_path)
    baseline_path = tmp_path / "critic_baseline.json"

    failing_source = (
        "def test_task_alpha():\n"
        "    assert False\n"
        "\n"
        "def test_task_beta():\n"
        "    assert True\n"
    )
    (harness_dir / "tests" / "test_baseline_regression.py").write_text(failing_source)

    task_alpha = BenchmarkTask(name="task_alpha", description="always-failing")
    task_beta = BenchmarkTask(name="task_beta", description="passing")
    critic = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_baseline_regression.py"],
        benchmark_tasks=[task_alpha, task_beta],
        baseline_path=baseline_path,
    )

    verdict = critic.evaluate("")
    assert verdict.approved is False
    assert "pytest" in verdict.failed_checks
    assert not any(c.startswith("regression:") for c in verdict.failed_checks)


def test_critic_baseline_round_trips_through_pydantic() -> None:
    """``CriticBaseline`` is a pydantic v2 model (ADR-0006 boundary model)."""
    baseline = CriticBaseline(
        schema_version=1,
        entries={
            "alpha": BaselineEntry(task_name="alpha", passing=True),
            "beta": BaselineEntry(task_name="beta", passing=False),
        },
    )
    assert CriticBaseline.model_validate(baseline.model_dump()) == baseline
    assert CriticBaseline.model_validate_json(baseline.model_dump_json()) == baseline
