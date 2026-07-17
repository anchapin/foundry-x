"""Benchmark task: edit_config_file precision (issue #812).

Every other edit-shaped benchmark (``surgical_edit``, ``fix_syntax_error``)
validates its golden solution by running the fixed code end-to-end. No task
tests whether an agent can surgically edit a *structured config file* (JSON)
while preserving the rest of the file structure.

This task seeds a ``config.json`` where the ``port`` value under
``server`` is a string ``"8080"`` instead of an integer ``8080``. The
golden solution patches only that one key-value pair, leaving all other
keys byte-identical. It exercises ``edit_file`` precision
(``harness/skills/edit_file.json``) on a non-Python format and gives
the Critic a regression target for "the agent corrupted unrelated config
keys."

The golden fix is modelled as a targeted ``old_string`` -> ``new_string``
replacement -- exactly the ``edit_file`` contract -- NOT a full-file
rewrite. The test then asserts:

    1. **Pre-condition** -- the seeded config causes validation to fail
       (port is a string, not an integer).
    2. **Surgical edit** -- the ``old_string`` matches exactly once
       (the patch is unique and targeted) and the replacement is
       applied in place.
    3. **Post-condition** -- the validation script exits 0 with the
       expected stdout.
    4. **Precision** -- all keys except ``port`` under ``server`` are
       byte-identical (structural diff check), and the ``port`` value
       actually changed (non-vacuous guard).

See also ADR-0005 (pytest as evaluation framework).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="edit_config_file",
    description=(
        "Fix a broken key-value pair in a JSON config file so that "
        "validation passes, leaving all other keys byte-identical "
        "(exercises edit_file precision on structured config)."
    ),
    prompt=(
        "The file config.json contains a broken value for the key "
        '"port" under the "server" section. The port must be an integer '
        '(e.g., 8080), not a string (e.g., "8080"). Fix ONLY the '
        "broken port value; do not rewrite the whole file and do not "
        "modify any other keys. After the fix, validation must pass "
        "(python validate_config.py exits 0)."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After a targeted edit to the port value only, "
        "python validate_config.py exits 0 with the expected stdout, "
        "and all other config keys are byte-identical."
    ),
    timeout_seconds=30,
    requires_skills=["edit_file"],
    tags=["editing", "precision", "config", "ops"],
)

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

CONFIG_FILE = "config.json"
VALIDATE_SCRIPT = "validate_config.py"

BROKEN_KEY_PATH = ("server", "port")
BROKEN_KEY_STR = '"port": "8080"'
FIXED_KEY_STR = '"port": 8080'


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _json_sha256(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _run_validate(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, VALIDATE_SCRIPT],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


@pytest.mark.benchmark
def test_edit_config_file(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Asserts:
        1. The seeded config is genuinely broken (validation fails).
        2. The golden ``old_string`` matches exactly once (surgical/unique).
        3. After the in-place edit, ``python validate_config.py`` exits 0.
        4. All keys except ``server.port`` are byte-identical and the port
           value actually changed (non-vacuous).
    """
    config_path = benchmark_workspace / CONFIG_FILE
    validate_path = benchmark_workspace / VALIDATE_SCRIPT

    broken_config_text = (_FIXTURE_DIR / CONFIG_FILE).read_text()
    validate_script_text = (_FIXTURE_DIR / VALIDATE_SCRIPT).read_text()

    config_path.write_text(broken_config_text)
    validate_path.write_text(validate_script_text)

    broken_data = _load_json(config_path)

    pre_hash = _json_sha256(broken_data)

    bad = _run_validate(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded config must fail validation before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert "port must be an integer" in bad.stderr, (
        f"task {TASK.name}: expected 'port must be an integer' in stderr; got stderr={bad.stderr!r}"
    )

    occurrences = broken_config_text.count(BROKEN_KEY_STR)
    assert occurrences == 1, (
        f"task {TASK.name}: golden old_string must match exactly once "
        f"in the seeded config; found {occurrences}"
    )

    fixed_config_text = broken_config_text.replace(BROKEN_KEY_STR, FIXED_KEY_STR, 1)
    config_path.write_text(fixed_config_text)

    fixed_data = _load_json(config_path)
    post_hash = _json_sha256(fixed_data)

    good = _run_validate(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: patched config must pass validation; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert "Config valid:" in good.stdout, (
        f"task {TASK.name}: expected 'Config valid:' in stdout; got stdout={good.stdout!r}"
    )

    broken_data_cmp = dict(broken_data)
    fixed_data_cmp = dict(fixed_data)

    server_broken = broken_data_cmp.get("server", {})
    server_fixed = fixed_data_cmp.get("server", {})

    port_broken = server_broken.pop("port")
    port_fixed = server_fixed.pop("port")

    assert port_broken == "8080", (
        f"task {TASK.name}: broken port must be '8080' (string); got {port_broken!r}"
    )
    assert port_fixed == 8080, (
        f"task {TASK.name}: fixed port must be 8080 (int); got {port_fixed!r}"
    )

    assert server_broken == server_fixed, (
        f"task {TASK.name}: all keys under 'server' except 'port' must be "
        f"unchanged; got broken={server_broken!r} fixed={server_fixed!r}"
    )

    broken_data_cmp["server"] = server_broken
    fixed_data_cmp["server"] = server_fixed

    for key in list(broken_data_cmp.keys()):
        if key == "server":
            continue
        assert broken_data_cmp[key] == fixed_data_cmp[key], (
            f"task {TASK.name}: top-level key {key!r} changed during edit; "
            f"the fix must modify only the port value"
        )

    assert pre_hash != post_hash, (
        f"task {TASK.name}: config hash unchanged after the golden edit; "
        f"the patch did not alter the config"
    )
