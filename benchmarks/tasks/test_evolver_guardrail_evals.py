"""Benchmark task: Evolver guardrails are active (SECURITY.md §"Rate limits").

Regression target for ``src/foundry_x/evolution/evolver.py`` (ADR-0004).
Two guards must hold simultaneously:

1. ``ProposedEdit(target_file=...)`` confines edits to the harness tree:
   only ``harness/system_prompt.txt``, files beneath
   ``harness/hooks/``, and files beneath ``harness/skills/`` are
   acceptable. Absolute paths, ``..`` traversal escapes, sibling
   files (``harness/README.md``), and the directory entries themselves
   are rejected at the pydantic boundary (ADR-0006).
2. ``Evolver._check_rate_limit`` raises ``EvolverGuardError`` once the
   rolling one-hour proposal window is full. Defaults (10 proposals /
   hour, 200 lines per diff) mirror SECURITY.md prose and any softening
   of those limits via the constructor must still observe the contract.

A regression that widens ``_HARNESS_SUBDIRS``, drops a path-traversal
check, or weakens the rate limiter surfaces here as a failing
benchmark and blocks the harness edit at PR review (ADR-0004).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from benchmarks.models import BenchmarkTask
from foundry_x.evolution.evolver import (
    Evolver,
    EvolverGuardError,
    ProposedEdit,
)


TASK = BenchmarkTask(
    name="evolver_guardrail",
    description=(
        "ProposedEdit confines target_file to harness/{system_prompt.txt,"
        "hooks/*,skills/*}; Evolver caps proposals at max_proposals_per_hour"
        " in any rolling hour; oversized diffs raise EvolverGuardError."
    ),
    prompt=(
        "Inspect src/foundry_x/evolution/evolver.py: confirm "
        "_confine_to_harness_tree rejects paths outside the harness tree "
        "and ProposedEdit enforces the confine at the pydantic boundary; "
        "confirm Evolver._check_rate_limit still trips at the cap and "
        "_validate_edit still rejects oversized unified diffs."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "ProposedEdit raises ValidationError for paths outside "
        "harness/system_prompt.txt / hooks/ / skills/ ; canonicalises "
        "every accepted path to POSIX; Evolver(max_proposals_per_hour=N) "
        "raises EvolverGuardError after N proposals in the rolling hour; "
        "Evolver(max_diff_lines=M) raises EvolverGuardError for diffs "
        "whose line count exceeds M."
    ),
    tags=["security"],
)


# --- Path rejection (10 cases) ---------------------------------------------
# Each case is a deliberate fixture in its own test function so the
# hygiene marker-coverage check (tests/benchmarks/test_hygiene.py) finds
# the @pytest.mark.benchmark decorator on every source-level definition.


@pytest.mark.benchmark
def test_proposed_edit_rejects_etc_passwd() -> None:
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="etc/passwd",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_traversal_escape() -> None:
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="../etc/passwd",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_absolute_path() -> None:
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="/absolute/path/harness/hooks/a.py",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_harness_readme() -> None:
    """A sibling file at the harness root is not in the allowed set."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="harness/README.md",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_harness_version() -> None:
    """The ``VERSION`` file is inside ``harness/`` but is not editable."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="harness/VERSION",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_prompt_as_directory() -> None:
    """``system_prompt.txt`` is a leaf file, not a directory prefix."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="harness/system_prompt.txt/hooks/inner.py",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_hooks_directory_itself() -> None:
    """``hooks`` is a directory; an edit must target a file inside it."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="harness/hooks",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_skills_directory_itself() -> None:
    """``skills`` is a directory; an edit must target a file inside it."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="harness/skills",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_src_path() -> None:
    """An edit to ``src/foundry_x/...`` is not under ``harness/``."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="src/foundry_x/execution/runner.py",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


@pytest.mark.benchmark
def test_proposed_edit_rejects_backslash_component() -> None:
    """Backslash is an illegal component (Windows-style escapes)."""
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file="harness\\hooks\\a.py",
            rationale="tighten tool guidance",
            unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+a\n",
        )


# --- Path acceptance (5 cases) ---------------------------------------------


@pytest.mark.benchmark
def test_proposed_edit_accepts_system_prompt_txt() -> None:
    edit = ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="x",
        unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+one\n",
    )
    assert edit.target_file == "harness/system_prompt.txt"


@pytest.mark.benchmark
def test_proposed_edit_accepts_hooks_file() -> None:
    edit = ProposedEdit(
        target_file="harness/hooks/a.py",
        rationale="x",
        unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+one\n",
    )
    assert edit.target_file == "harness/hooks/a.py"


@pytest.mark.benchmark
def test_proposed_edit_accepts_hooks_nested_file() -> None:
    edit = ProposedEdit(
        target_file="harness/hooks/sub/b.py",
        rationale="x",
        unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+one\n",
    )
    assert edit.target_file == "harness/hooks/sub/b.py"


@pytest.mark.benchmark
def test_proposed_edit_accepts_skills_file() -> None:
    edit = ProposedEdit(
        target_file="harness/skills/x.md",
        rationale="x",
        unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+one\n",
    )
    assert edit.target_file == "harness/skills/x.md"


@pytest.mark.benchmark
def test_proposed_edit_accepts_skills_nested_file() -> None:
    edit = ProposedEdit(
        target_file="harness/skills/nested/y.md",
        rationale="x",
        unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+one\n",
    )
    assert edit.target_file == "harness/skills/nested/y.md"


# --- Normalisation + rate / diff guards + defaults -------------------------


@pytest.mark.benchmark
def test_proposed_edit_canonicalises_dotdot_dot_paths() -> None:
    """Redundant ``.`` and non-escaping ``..`` segments collapse to canonical POSIX.

    The canonicalise step is the contract that lets the Critic compare
    diff targets to the allow-list without playing normalisation games.
    """
    edit = ProposedEdit(
        target_file="harness/./hooks/../hooks/a.py",
        rationale="x",
        unified_diff="--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+one\n",
    )
    assert edit.target_file == "harness/hooks/a.py"


@pytest.mark.benchmark
def test_evolver_rate_limit_raises_after_cap() -> None:
    """``_check_rate_limit`` raises once the rolling hour is at the cap.

    The bound is exposed as ``max_proposals_per_hour`` (default 10) and
    any loosening of that constant — or any regression that forgets to
    purge stale timestamps — breaks the SECURITY.md "max N proposals
    per hour" guarantee.
    """
    evolver = Evolver(max_proposals_per_hour=3, max_diff_lines=200)
    evolver._record_proposals(3)
    with pytest.raises(EvolverGuardError, match="rate limit exceeded"):
        evolver._check_rate_limit()


@pytest.mark.benchmark
def test_evolver_old_proposals_purge_outside_window() -> None:
    """Proposals older than one hour fall out of the rolling window.

    The ``deque`` is purged against a cutoff of ``now - 1h``. A
    regression that stops purging (or that uses a tighter window)
    surfaces as a stale timestamp counting toward today's cap.
    """
    evolver = Evolver(max_proposals_per_hour=2, max_diff_lines=200)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    evolver._proposal_times.append(stale)
    evolver._proposal_times.append(stale)
    # Both stale entries must be purged before the cap check; the test
    # would otherwise raise.
    evolver._check_rate_limit()


@pytest.mark.benchmark
def test_evolver_rejects_oversized_diff() -> None:
    """A unified diff exceeding ``max_diff_lines`` raises ``EvolverGuardError``."""
    evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    header = "--- a/x\n+++ b/x\n"
    hunk = "@@ -0,0 +1 @@\n"
    big_diff = header + hunk + "".join(f"+line {i}\n" for i in range(10))
    with pytest.raises(EvolverGuardError, match="diff too large"):
        evolver._validate_edit(
            ProposedEdit(
                target_file="harness/hooks/a.py",
                rationale="x",
                unified_diff=big_diff,
            )
        )


@pytest.mark.benchmark
def test_evolver_defaults_match_security_doc() -> None:
    """Defaults mirror SECURITY.md "max N proposals per hour, max M lines of diff".

    SECURITY.md §"Rate limits" calls out two numbers; a regression that
    changes the defaults diverges from the prose and from this benchmark.
    """
    evolver = Evolver()
    assert evolver.max_proposals_per_hour == 10
    assert evolver.max_diff_lines == 200


@pytest.mark.benchmark
def test_evolver_rejects_zero_or_negative_limits() -> None:
    """A non-positive cap is nonsensical and must surface at construction time.

    ``AGENTS.md`` §2 forbids silently swallowing configuration errors,
    so a cap of 0 must raise rather than disable the guard.
    """
    with pytest.raises(EvolverGuardError, match="max_proposals_per_hour"):
        Evolver(max_proposals_per_hour=0)
    with pytest.raises(EvolverGuardError, match="max_diff_lines"):
        Evolver(max_diff_lines=0)
