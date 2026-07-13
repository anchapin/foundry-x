from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

from foundry_x.execution.runner import (
    _FALLBACK_HARNESS_VERSION,
    resolve_harness_version,
)


def test_resolve_reads_version_file_and_trims_whitespace(tmp_path: Path):
    """Issue #11 (a): when harness/VERSION exists, its content (whitespace
    trimmed) is returned verbatim — the default path."""
    (tmp_path / "VERSION").write_text("  1.2.3-rc4\n", encoding="utf-8")

    assert resolve_harness_version(tmp_path) == "1.2.3-rc4"


def test_resolve_ignores_blank_version_file_and_falls_back(tmp_path: Path):
    """A VERSION file that is only whitespace must not win; resolution
    continues to the git/source fallback rather than stamping an empty
    string onto the session."""
    (tmp_path / "VERSION").write_text("\n  \n", encoding="utf-8")

    with mock.patch("subprocess.run") as fake_run:
        fake_run.side_effect = OSError("no git")
        # Blank file + no git -> literal fallback.
        assert resolve_harness_version(tmp_path) == _FALLBACK_HARNESS_VERSION


def test_resolve_falls_back_to_git_when_version_file_absent(tmp_path: Path):
    """Issue #11 (b): without a VERSION file, ``git describe`` of the harness
    directory is used so an evolved checkout self-describes."""
    completed = subprocess.CompletedProcess(
        args=["git", "describe", "--tags", "--always"],
        returncode=0,
        stdout="v0.4.2-3-gabc1234\n",
        stderr="",
    )
    with mock.patch("subprocess.run", return_value=completed) as fake_run:
        result = resolve_harness_version(tmp_path)

    assert result == "v0.4.2-3-gabc1234"
    # git must be invoked *inside* the harness directory.
    assert fake_run.call_args.kwargs["cwd"] == str(tmp_path)


def test_resolve_falls_back_to_constant_when_neither_available(tmp_path: Path):
    """Issue #11 (c): no VERSION file and a git failure (e.g. git missing or
    not a repo) yields the literal fallback so the run can still proceed."""
    with mock.patch("subprocess.run", side_effect=OSError("git not installed")):
        assert resolve_harness_version(tmp_path) == _FALLBACK_HARNESS_VERSION


def test_resolve_falls_back_to_constant_on_subprocess_error(tmp_path: Path):
    """A non-zero git exit (e.g. not a git repo, no tags yet) also degrades to
    the literal rather than crashing the runner."""
    err = subprocess.CalledProcessError(128, ["git", "describe"])
    with mock.patch("subprocess.run", side_effect=err):
        assert resolve_harness_version(tmp_path) == _FALLBACK_HARNESS_VERSION
