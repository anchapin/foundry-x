from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from defusedxml import ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.models import BenchmarkTask
from benchmarks.registry import load_all_tasks

_NOTES_TAIL_CHARS = 4000

# Wall-clock cap (seconds) for the Critic gate. Mirrors the Runner's
# ``RunLimits.task_timeout_s`` pattern (runner.py:DEFAULT_TASK_TIMEOUT_S) so
# a malformed benchmark that hangs in collection, or a proposed harness edit
# that re-introduces an infinite loop in a hook, cannot make the gate
# itself a runaway. See docs/SECURITY.md "Runaway detection" and ADR-0004
# §Consequences ("Evolution runs are slower but bounded").
DEFAULT_GATE_TIMEOUT_S: int = 300

#: Default location of the regression baseline JSON written by the Critic
#: (ADR-0004 step 3, issue #186). Relative to the process working directory
#: so an invocation from the repo root lands at ``logs/critic_baseline.json``.
DEFAULT_BASELINE_PATH: Path = Path("logs") / "critic_baseline.json"


class CriticVerdict(BaseModel):
    """Result of a Critic gate run against a proposed harness edit (ADR-0006)."""

    approved: bool
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""


class BaselineEntry(BaseModel):
    """One task's pass/fail state in the Critic regression baseline (ADR-0006).

    Per ADR-0004 step 3 the Critic's regression gate keeps a record of which
    benchmark tasks were passing on the last observation (issue #186). A
    previously-passing task that flips to failing on a later evaluation
    surfaces as ``regression:<task_name>`` in the verdict's
    ``failed_checks`` and rejects the gate.
    """

    task_name: str
    passing: bool


class CriticBaseline(BaseModel):
    """Critic regression baseline, persisted at ``logs/critic_baseline.json``.

    First ``Critic.evaluate()`` after the Critic lands on a project writes the
    baseline; subsequent calls diff against it. Schema-bump via
    ``schema_version`` when the on-disk shape changes (see ADR-0008 for the
    convention commit discipline applied to this artifact).
    """

    schema_version: int = 1
    entries: dict[str, BaselineEntry] = Field(default_factory=dict)


class Critic:
    """Gatekeeper that evaluates proposed harness edits in a sandbox.

    Per ADR-0004 every harness edit must pass through this gate before it is
    marked active. The gate applies the ``proposed_diff`` against a *copy* of
    the harness inside a temporary directory and runs pytest there — the live
    ``harness_dir`` is never mutated.

    Benchmark-subset selection (ADR-0004 step 2) uses ``-m benchmark`` by
    default so every ``@pytest.mark.benchmark`` task gates the edit
    (ADR-0005, issue #185). The verdict's ``passed_checks`` lists every
    benchmark tag the run covered.

    The Critic also maintains a regression baseline (ADR-0004 step 3,
    issue #186). The first ``evaluate()`` call on a baseline-path that does
    not yet have a ``critic_baseline.json`` writes one; later calls diff the
    current benchmark-task outcomes against the persisted baseline, and any
    task that previously passed but now fails rejects the gate with a
    ``regression:<task_name>`` entry in ``failed_checks``.

    Per-test results are harvested from pytest's JUnit-XML report (written
    to a temp file we control), not by parsing pytest's stdout. This keeps
    the verdict's ``notes`` field identical to the pre-#186 shape — the
    regression gate is invisible to consumers of the verdict text.
    """

    def __init__(
        self,
        harness_dir: Path,
        benchmark_path: Path | None = None,
        pytest_args: list[str] | None = None,
        benchmark_tasks: list[BenchmarkTask] | None = None,
        gate_timeout_s: int = DEFAULT_GATE_TIMEOUT_S,
        baseline_path: Path | None = None,
    ) -> None:
        self.harness_dir = harness_dir
        self.benchmark_path = benchmark_path
        # Next-step wiring target (issue #107): prepend `python
        # harness/scripts/load_check.py --harness-dir <copy>` so a broken
        # `harness/skills/*.json` or an unimportable hook fails the gate
        # *before* pytest runs. Out of scope for #107.
        # Default selection runs the full benchmark suite via ``-m benchmark``
        # (ADR-0005, issue #185) — a harness edit that breaks any
        # ``@pytest.mark.benchmark`` task is caught at the gate.
        self.pytest_args = pytest_args or ["-q", "-m", "benchmark"]
        # Wall-clock cap (issue #188) applied to every subprocess inside
        # ``evaluate`` so a hanging benchmark or a proposed harness edit that
        # re-introduces an infinite loop in a hook cannot make the gate
        # itself a runaway (docs/SECURITY.md "Runaway detection").
        self.gate_timeout_s = gate_timeout_s
        # In-process registry wiring (issue #108): the Critic can now
        # enumerate benchmark tasks without spawning pytest. Stored as
        # ``None`` so the registry is loaded lazily on first access --
        # importing ``foundry_x.evolution.critic`` must not eagerly pull
        # in every task module (and the pytest import chain those tasks
        # transitively trigger).
        self._benchmark_tasks: list[BenchmarkTask] | None = (
            list(benchmark_tasks) if benchmark_tasks is not None else None
        )
        # Regression baseline (ADR-0004 step 3, issue #186). When ``None``
        # the Critic behaves exactly as it did before #186: no baseline
        # file is written and no regression check runs. The default
        # ``DEFAULT_BASELINE_PATH`` (``logs/critic_baseline.json``) opts
        # callers into the regression gate so the project-level Critic
        # invocation ships a baseline on its first run.
        self.baseline_path: Path | None = (
            Path(baseline_path) if baseline_path is not None else DEFAULT_BASELINE_PATH
        )

    @property
    def benchmark_tasks(self) -> list[BenchmarkTask]:
        """The ``BenchmarkTask`` instances this Critic will gate against (issue #108).

        Lazy-loaded from the in-process registry on first access; cached on
        the instance so subsequent accesses are O(1). Tests can pre-seed
        ``benchmark_tasks=...`` in the constructor to avoid touching the
        registry at all (see ``tests/test_critic.py``).
        """
        if self._benchmark_tasks is None:
            self._benchmark_tasks = load_all_tasks()
        return self._benchmark_tasks

    def evaluate(self, proposed_diff: str) -> CriticVerdict:
        """Apply ``proposed_diff`` to a sandbox copy of the harness and run pytest.

        Steps (ADR-0004):

        1. Copy ``harness_dir`` into a fresh ``TemporaryDirectory``.
        2. Apply ``proposed_diff`` via ``git apply``. A patch that does not
           apply cleanly is rejected immediately (``failed_checks=["git apply"]``).
        3. Run pytest with ``self.pytest_args`` in the sandbox, plus a
           ``--junit-xml=<tmpfile>`` flag we own so per-test pass/fail is
           recoverable from structured XML (issue #186: the regression
           gate parses the JUnit XML, not stdout).
        4. Diff the observed per-task results against the persisted
           baseline (``self.baseline_path``). Any benchmark task that
           ``was passing`` and ``now fails`` appends
           ``regression:<task_name>`` to ``failed_checks``.
        5. Write the current pass/fail state as the new baseline.

        Every subprocess inside this method is bounded by
        ``self.gate_timeout_s`` (issue #188). On
        :class:`subprocess.TimeoutExpired` the verdict is
        ``approved=False`` with ``failed_checks`` carrying the offending check
        name suffixed ``":timeout"`` (e.g. ``"pytest:timeout"``), and
        ``notes`` holds the trailing window of any partial output the
        process managed to write before being killed — or a wall-clock-cap
        message when no partial output was captured.

        The verdict's ``approved`` flag is ``True`` only when every check that
        runs succeeds. All filesystem mutations are confined to the temp copy;
        the baseline file is the only durable mutation.
        """
        with tempfile.TemporaryDirectory(prefix="critic-sandbox-") as sandbox:
            sandbox_root = Path(sandbox) / "harness"
            shutil.copytree(self.harness_dir, sandbox_root)

            passed_checks: list[str] = []
            failed_checks: list[str] = []

            # 1. Apply the proposed diff to the sandbox copy only.
            if proposed_diff.strip():
                try:
                    apply_result = subprocess.run(
                        ["git", "apply", "--whitespace=nowarn"],
                        input=proposed_diff,
                        cwd=sandbox_root,
                        capture_output=True,
                        text=True,
                        timeout=self.gate_timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    return CriticVerdict(
                        approved=False,
                        passed_checks=[],
                        failed_checks=["git apply:timeout"],
                        notes=_timeout_notes(exc, self.gate_timeout_s, "git apply"),
                    )
                if apply_result.returncode != 0:
                    return CriticVerdict(
                        approved=False,
                        passed_checks=[],
                        failed_checks=["git apply"],
                        notes=_tail(apply_result.stderr or apply_result.stdout),
                    )
                passed_checks.append("git apply")

            # 2. Run pytest in the sandbox with a JUnit-XML report we own.
            # The XML is the structured channel the regression gate parses
            # (step 4); user-supplied verbosity flags (e.g. ``-q``) are
            # untouched so the verdict's ``notes`` field keeps its
            # pre-#186 shape. The wall-clock cap (issue #188) applies
            # here too so a hanging benchmark cannot make the gate a runaway.
            with tempfile.TemporaryDirectory(prefix="critic-junit-") as junit_dir:
                junit_path = Path(junit_dir) / "results.xml"
                pytest_cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"--junit-xml={junit_path}",
                    "--no-header",
                    *self.pytest_args,
                ]
                try:
                    pytest_result = subprocess.run(
                        pytest_cmd,
                        cwd=sandbox_root,
                        capture_output=True,
                        text=True,
                        timeout=self.gate_timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    return CriticVerdict(
                        approved=False,
                        passed_checks=passed_checks,
                        failed_checks=["pytest:timeout"],
                        notes=_timeout_notes(exc, self.gate_timeout_s, "pytest"),
                    )

                combined = (pytest_result.stdout or "") + (pytest_result.stderr or "")
                per_test_results = _parse_junit_results(junit_path)

            # 3. Map the configured benchmark tasks to their current
            # pass/fail state. Tasks whose pytest test was not collected
            # are omitted from the baseline (they simply were not observed).
            current_entries = _evaluate_benchmark_tasks(self.benchmark_tasks, per_test_results)

            # 4. Regression gate: any task that was passing and now fails
            # surfaces as a failure. This step is a no-op when
            # ``baseline_path`` is ``None`` (no persisted history).
            if self.baseline_path is not None:
                previous_baseline = _load_baseline(self.baseline_path)
                regressions = _detect_regressions(previous_baseline, current_entries)
                for task_name in regressions:
                    failed_checks.append(f"regression:{task_name}")

                # 5. Persist the current pass/fail snapshot as the new
                # baseline so the next ``evaluate()`` can diff against it.
                _write_baseline(
                    self.baseline_path,
                    CriticBaseline(
                        schema_version=1,
                        entries={entry.task_name: entry for entry in current_entries},
                    ),
                )


            if pytest_result.returncode == 0:
                passed_checks.append("pytest")
                # Record every benchmark tag the run covered (issue #185).
                covered_tags = sorted({tag for task in self.benchmark_tasks for tag in task.tags})
                passed_checks.extend(f"benchmark:{tag}" for tag in covered_tags)
            else:
                failed_checks.append("pytest")

            return CriticVerdict(
                approved=not failed_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                notes=_tail(combined),
            )


def _parse_junit_results(junit_path: Path) -> dict[str, bool]:
    """Parse a pytest JUnit-XML report into ``{node_id: passing?}`` (issue #186).

    Pytest writes one ``<testcase classname="..." name="...">`` per collected
    test. A ``<failure>`` or ``<error>`` child marks the test as
    non-passing; absence of those (and the absence of ``<skipped>``) is a
    pass. The node id we surface is ``classname::name`` — the same shape
    pytest uses on the command line (``tests/test_x.py::test_task_alpha``).
    """
    if not junit_path.exists():
        return {}
    try:
        tree = ET.parse(junit_path)
    except ET.ParseError:
        return {}
    root = tree.getroot()
    results: dict[str, bool] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        if not classname or not name:
            continue
        # pytest reports ``classname`` as the file's dotted path
        # (``tests.test_x``) for some versions and the slash path
        # (``tests/test_x.py``) for others. We normalise both to the
        # ``tests/test_x.py`` form so downstream lookups match the
        # convention ``endswith("::test_<task_name>")``.
        normalised_classname = classname.replace(".", "/") + ".py"
        node_id = f"{normalised_classname}::{name}"
        has_failure = any(child.tag in {"failure", "error"} for child in case)
        results[node_id] = not has_failure
    return results


def _evaluate_benchmark_tasks(
    tasks: list[BenchmarkTask],
    per_test_results: dict[str, bool],
) -> list[BaselineEntry]:
    """Map each configured ``BenchmarkTask`` to its current pass/fail state.

    Convention: a benchmark task with ``name="task_alpha"`` corresponds to
    a pytest test function named ``test_task_alpha`` somewhere on the
    test node surface. This is the convention used by every
    ``benchmarks/tasks/test_*.py`` file (e.g.
    ``tests/test_smoke.py::test_smoke_marker_and_fixture_resolve`` ↔
    ``TASK.name == "smoke_marker_and_fixture_resolve"``) and by the
    synthetic fixtures in this test file.

    A task whose ``test_<name>`` did not show up in the parsed output is
    omitted: it was not observed this run and the baseline therefore has
    no opinion on it. The next evaluation will try again.
    """
    entries: list[BaselineEntry] = []
    for task in tasks:
        expected_test_name = f"test_{task.name}"
        for node, passed in per_test_results.items():
            if node.endswith(f"::{expected_test_name}"):
                entries.append(BaselineEntry(task_name=task.name, passing=passed))
                break
    return entries


def _detect_regressions(
    previous: CriticBaseline,
    current_entries: list[BaselineEntry],
) -> list[str]:
    """Return task names that previously passed and now fail.

    The set is sorted for deterministic verdict ordering — stable order
    simplifies debugging and keeps verdicts round-trippable through
    ``CriticVerdict``'s ``failed_checks`` list.
    """
    prev_passing = {entry.task_name for entry in previous.entries.values() if entry.passing}
    current_failing = {entry.task_name for entry in current_entries if not entry.passing}
    return sorted(prev_passing & current_failing)


def _load_baseline(path: Path) -> CriticBaseline:
    """Load the persisted baseline, or return an empty one if missing/corrupt.

    A missing file is the first-run case (the baseline will be written
    after this evaluation). A corrupt file is treated as empty so a stray
    ``git checkout`` of an old shape does not break the gate; the next
    ``_write_baseline`` will overwrite the artifact with a valid v1.
    """
    if not path.exists():
        return CriticBaseline()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return CriticBaseline()
    if not text.strip():
        return CriticBaseline()
    try:
        loaded = CriticBaseline.model_validate_json(text)
    except (ValueError, json.JSONDecodeError):
        return CriticBaseline()
    if loaded.schema_version != CriticBaseline.model_fields["schema_version"].default:
        return CriticBaseline()
    return loaded


def _write_baseline(path: Path, baseline: CriticBaseline) -> None:
    """Persist *baseline* at *path* via a temp-file + atomic rename.

    Atomic rename keeps an interrupted ``evaluate()`` from leaving a
    half-written JSON that the next call would mistake for a corrupt
    baseline (handled defensively in :func:`_load_baseline` but worth
    avoiding anyway).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _tail(text: str) -> str:
    """Return the trailing window of *text* for inclusion in verdict notes."""
    return text.strip()[-_NOTES_TAIL_CHARS:]


def _timeout_notes(exc: subprocess.TimeoutExpired, gate_timeout_s: int, check: str) -> str:
    """Build verdict ``notes`` for a ``subprocess.TimeoutExpired``.

    Prefers the trailing window of any partial stdout/stderr the subprocess
    managed to write before being killed (the same shape callers expect for
    non-timeout failures). Falls back to a wall-clock-cap message naming the
    offending check so the verdict is never empty: the
    ``test_pytest_exceeds_timeout_rejected`` acceptance test (issue #188)
    asserts ``verdict.notes`` is truthy.
    """
    partial = (exc.stdout or b"") + (exc.stderr or b"")
    if isinstance(partial, bytes):
        partial = partial.decode("utf-8", errors="replace")
    if partial.strip():
        return _tail(partial)
    return _tail(f"{check} exceeded {gate_timeout_s}s wall-clock cap")
