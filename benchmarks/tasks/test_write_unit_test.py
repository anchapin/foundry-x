"""Benchmark task: write a unit test for a given function.

This benchmark family is parameterized across difficulty tiers (easy, medium, hard),
each generating tests of increasing complexity for different target functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask, DifficultyTier
from benchmarks.support import run_module

# --- Tier definitions -------------------------------------------------------

TIERS: dict[DifficultyTier, dict[str, str]] = {
    "easy": {
        "name": "write_unit_test_easy",
        "description": "Author a passing pytest suite for a simple add function.",
        "prompt": (
            "The file target.py defines add(a, b). Write test_add.py with pytest "
            "cases that exercise add, then leave it in the workspace."
        ),
    },
    "medium": {
        "name": "write_unit_test_medium",
        "description": "Author a passing pytest suite for a string repeat function.",
        "prompt": (
            "The file target.py defines repeat(text, times) that repeats a string "
            "`times` times separated by commas, raising ValueError for negative times. "
            "Write test_repeat.py with pytest cases that exercise repeat, then leave it "
            "in the workspace."
        ),
    },
    "hard": {
        "name": "write_unit_test_hard",
        "description": "Author a passing pytest suite for a calculator function.",
        "prompt": (
            "The file target.py defines calculator(expression) that evaluates simple "
            "arithmetic expressions like '2 + 3' or '10 / 4'. It returns None for "
            "division by zero and raises ValueError for invalid expressions. "
            "Write test_calculator.py with pytest cases that exercise calculator, "
            "then leave it in the workspace."
        ),
    },
}

# --- Golden tests (agent output that must pass) -----------------------------

GOLDEN_TESTS: dict[DifficultyTier, str] = {
    "easy": """\
from target import add


def test_add_positive():
    assert add(2, 3) == 5


def test_add_zero():
    assert add(0, 0) == 0


def test_add_negative():
    assert add(-1, 1) == 0
""",
    "medium": """\
import pytest
from target import repeat


def test_repeat_basic():
    assert repeat("hi", 3) == "hi, hi, hi"


def test_repeat_zero():
    assert repeat("hello", 0) == ""


def test_repeat_negative():
    with pytest.raises(ValueError, match="non-negative"):
        repeat("test", -1)


def test_repeat_one():
    assert repeat("x", 1) == "x"


def test_repeat_longer():
    result = repeat("ab", 4)
    assert result == "ab, ab, ab, ab"
""",
    "hard": """\
import pytest
from target import calculator


def test_calculator_add():
    assert calculator("2 + 3") == 5.0


def test_calculator_subtract():
    assert calculator("10 - 4") == 6.0


def test_calculator_multiply():
    assert calculator("3 * 4") == 12.0


def test_calculator_divide():
    assert calculator("10 / 2") == 5.0


def test_calculator_division_by_zero():
    assert calculator("5 / 0") is None


def test_calculator_invalid_expression():
    with pytest.raises(ValueError, match="Invalid expression"):
        calculator("not an expression")


def test_calculator_wrong_arity():
    with pytest.raises(ValueError, match="Invalid expression"):
        calculator("1 + 2 + 3")


def test_calculator_unknown_operator():
    with pytest.raises(ValueError, match="Invalid expression"):
        calculator("1 ^ 2")


def test_calculator_negative_result():
    assert calculator("3 - 8") == -5.0


def test_calculator_float_result():
    result = calculator("10 / 4")
    assert abs(result - 2.5) < 0.0001
""",
}

# --- BenchmarkTask definitions (validated by Pydantic on access) -----------

_TASKS = {
    tier: BenchmarkTask(
        name=meta["name"],
        description=meta["description"],
        prompt=meta["prompt"],
        difficulty_tier=tier,
        tags=["testing", "test_generation", f"difficulty_tier:{tier}"],
    )
    for tier, meta in TIERS.items()
}

# --- Test cases -------------------------------------------------------------

_TIERS_LIST = list(TIERS.keys())


@pytest.mark.parametrize("tier", _TIERS_LIST)
@pytest.mark.benchmark
def test_write_unit_test(benchmark_workspace: Path, tier: DifficultyTier) -> None:
    """Parameterized pass/fail check across difficulty tiers (easy/medium/hard)."""
    tier_meta = TIERS[tier]
    fixture_dir = Path(__file__).parent.parent / "fixtures" / "write_unit_test" / tier
    golden = GOLDEN_TESTS[tier]

    # Set up target function from fixture
    (benchmark_workspace / "target.py").write_text((fixture_dir / "target.py").read_text())

    # Write the golden test (agent's expected output)
    test_filename = (
        "test_add.py"
        if tier == "easy"
        else ("test_repeat.py" if tier == "medium" else "test_calculator.py")
    )
    (benchmark_workspace / test_filename).write_text(golden)

    result = run_module(benchmark_workspace, "pytest")

    assert result.returncode == 0, (
        f"task {tier_meta['name']}: tests failed\n{result.stdout}{result.stderr}"
    )
    # Verify correct number of tests passed based on tier
    expected_counts = {"easy": "3 passed", "medium": "5 passed", "hard": "10 passed"}
    assert expected_counts[tier] in result.stdout, (
        f"task {tier_meta['name']}: unexpected pytest summary\n{result.stdout}"
    )
