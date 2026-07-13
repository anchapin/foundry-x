from __future__ import annotations

from pathlib import Path

from foundry_x.execution.runner import resolve_harness_version


def test_resolve_reads_version_file_and_trims_whitespace(tmp_path: Path):
    """When ``_version.txt`` exists, its content (whitespace trimmed) is
    returned verbatim."""
    (tmp_path / "_version.txt").write_text("  1.2.3-rc4\n", encoding="utf-8")

    assert resolve_harness_version(tmp_path) == "1.2.3-rc4"


def test_resolve_ignores_blank_version_file_and_falls_back_to_env(tmp_path: Path, monkeypatch):
    """A ``_version.txt`` that is only whitespace must not win; resolution
    continues to the ``FOUNDRY_HARNESS_VERSION`` env var."""
    (tmp_path / "_version.txt").write_text("\n  \n", encoding="utf-8")
    monkeypatch.setenv("FOUNDRY_HARNESS_VERSION", "2.0.0")
    assert resolve_harness_version(tmp_path) == "2.0.0"


def test_resolve_falls_back_to_env_when_version_file_absent(tmp_path: Path, monkeypatch):
    """Without a ``_version.txt`` file, ``FOUNDRY_HARNESS_VERSION`` is used."""
    monkeypatch.setenv("FOUNDRY_HARNESS_VERSION", "3.1.0")
    assert resolve_harness_version(tmp_path) == "3.1.0"


def test_resolve_env_takes_precedence_over_blank_file(tmp_path: Path, monkeypatch):
    """Even when ``FOUNDRY_HARNESS_VERSION`` is set, a non-blank ``_version.txt``
    wins (file has higher precedence than env)."""
    (tmp_path / "_version.txt").write_text("  1.0.0\n", encoding="utf-8")
    monkeypatch.setenv("FOUNDRY_HARNESS_VERSION", "2.0.0")
    assert resolve_harness_version(tmp_path) == "1.0.0"


def test_resolve_returns_empty_when_neither_source_available(tmp_path: Path):
    """No ``_version.txt`` file and no ``FOUNDRY_HARNESS_VERSION`` env var
    yields an empty string; the caller is responsible for surfacing an error."""
    assert resolve_harness_version(tmp_path) == ""
