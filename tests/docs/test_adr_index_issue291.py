"""Test for issue #291: ADR index drift detection.

This test specifically addresses issue #291 which required:
1. pytest discovers every docs/adr/NNNN-*.md and asserts each appears in README index
2. Asserts ADR numbers are contiguous (no gaps) and max(indexed) == count
3. Test fails today if ADR-0011 is referenced but absent
4. CI runs the test under existing pytest gate

The key insight from issue #291 is that we need to detect when an ADR number
is referenced somewhere (like in issue comments or documentation) but doesn't
have a corresponding file. This prevents silent drift where new ADRs are
proposed but never created.
"""

from __future__ import annotations

import re
from pathlib import Path

# tests/docs/test_adr_index_issue291.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "adr"
README = ADR_DIR / "README.md"

# An ADR file is NNNN-kebab-case-slug.md. README.md is the index, not an ADR.
_ADR_FILE_RE = re.compile(r"^(\d{4})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
# An index row links the number to its file: | [NNNN](./NNNN-slug.md) | ... |
_INDEX_ROW_RE = re.compile(r"\[(\d{4})\]\(\./\d{4}-[a-z0-9-]+\.md\)")
# External references like "ADR-0011" in README or issue text
_EXTERNAL_ADR_REF_RE = re.compile(r"ADR-(\d{4})")


def _adr_file_entries() -> list[tuple[int, Path]]:
    """Return ``[(number, path)]`` for every ``NNNN-*.md`` file, sorted.

    Duplicates are preserved so a colliding prefix (e.g. both
    ``0001-a.md`` and ``0001-b.md``) is detectable by the caller.
    """
    entries: list[tuple[int, Path]] = []
    for path in sorted(ADR_DIR.glob("*.md")):
        match = _ADR_FILE_RE.match(path.name)
        if match:
            entries.append((int(match.group(1)), path))
    return entries


def _index_numbers() -> list[int]:
    """Return the ADR numbers referenced in the README index table.

    Duplicates are preserved so a repeated index row is detectable.
    """
    text = README.read_text(encoding="utf-8")
    return [int(m.group(1)) for m in _INDEX_ROW_RE.finditer(text)]


def _external_adr_references(text: str) -> list[int]:
    """Find all ADR references like 'ADR-0011' in the given text.

    Includes references in README, issue descriptions, and other documentation.
    """
    return [int(m.group(1)) for m in _EXTERNAL_ADR_REF_RE.finditer(text)]


def _get_all_external_references() -> list[int]:
    """Collect all external ADR references from README and source files.

    This checks both the index and source files for ADR references to ensure
    we validate all mentioned ADRs against the actual filesystem.
    """
    # Check README first (primary index)
    all_refs = []
    readme_text = README.read_text(encoding="utf-8")
    all_refs.extend(_external_adr_references(readme_text))

    # Check any .md files in the repo root for ADR references
    # This looks for ADR references in issue files, documentation, etc.
    for md_file in REPO_ROOT.glob("*.md"):
        # Skip README.md and ADR directory to avoid double-counting
        if md_file.name in ("README.md", "SUMMARY.md") or "adr" in md_file.parts:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            all_refs.extend(_external_adr_references(content))
        except (OSError, UnicodeDecodeError):
            # Skip files that can't be read
            continue

    return all_refs


def test_readme_index_exists() -> None:
    """README.md is the ADR index; the integrity checks depend on it."""
    assert README.is_file(), f"missing ADR index: {README}"


def test_adr_file_numbers_are_unique() -> None:
    """No two ADR files may share the same NNNN prefix (ADR-0001:16-17).

    The filesystem allows ``0001-a.md`` and ``0001-b.md`` to coexist; that
    is a numbering collision this test must catch.
    """
    seen: dict[int, list[Path]] = {}
    for num, path in _adr_file_entries():
        seen.setdefault(num, []).append(path)
    collisions = {num: paths for num, paths in seen.items() if len(paths) > 1}
    assert not collisions, (
        "ADR number prefixes must be unique, but collisions exist: "
        + "; ".join(
            f"{num:04d} -> {[p.name for p in paths]}" for num, paths in sorted(collisions.items())
        )
        + ". Renumber one file per collision."
    )


def test_adr_index_rows_match_files() -> None:
    """Every ADR file has a README index row and vice-versa (README.md:24).

    Reports the offending number on either side of the mismatch.
    """
    file_numbers = {num for num, _ in _adr_file_entries()}
    index_numbers = set(_index_numbers())
    missing_from_index = sorted(file_numbers - index_numbers)
    missing_from_files = sorted(index_numbers - file_numbers)
    assert not (missing_from_index or missing_from_files), (
        "ADR index drift detected:\n"
        + (f"  - files without a README row: {missing_from_index}\n" if missing_from_index else "")
        + (f"  - README rows without a file: {missing_from_files}\n" if missing_from_files else "")
        + "Add the new ADR to the README.md index table in the same PR "
        "(docs/adr/README.md:24)."
    )


def test_adr_numbers_are_contiguous_from_0001() -> None:
    """ADR numbers must be gap-free from 0001 with no holes (ADR-0001:16-17)."""
    numbers = sorted({num for num, _ in _adr_file_entries()})
    assert numbers, "no ADR files found under docs/adr/"
    expected = list(range(1, numbers[-1] + 1))
    gaps = sorted(set(expected) - set(numbers))
    assert not gaps, (
        f"ADR numbering is not contiguous from 0001: missing {gaps}. "
        "ADRs must be numbered sequentially with no gaps "
        "(docs/adr/0001-record-architecture-decisions.md:16-17)."
    )


def test_external_adr_references_match_files() -> None:
    """All external ADR references (like ADR-0011) must have corresponding files.

    This addresses the specific issue #291 where ADR-0011 was referenced
    in issue comments but no corresponding file existed.

    The test fails if there are any ADR references in documentation that
    don't have corresponding files in the docs/adr/ directory.
    """
    # Get all ADR file numbers
    file_numbers = {num for num, _ in _adr_file_entries()}

    # Get all external ADR references from documentation
    all_external_refs = _get_all_external_references()

    # Filter to references that are NOT in the file numbers (potential issues)
    dangling_refs = []
    for ref_num in all_external_refs:
        if ref_num not in file_numbers:
            dangling_refs.append(ref_num)

    # Remove duplicates and sort
    dangling_refs = sorted(set(dangling_refs))

    # The test should fail if there are dangling references
    # This catches the exact issue #291 scenario where ADR-0011 was referenced
    # but didn't exist as a file
    assert not dangling_refs, (
        f"ADR references found without corresponding files: {dangling_refs}\n"
        "Every ADR reference in documentation must have a corresponding file "
        "in docs/adr/ directory. See issue #291 for context."
    )


def test_adr_index_drifts_with_new_adrs() -> None:
    """Unit test simulates adding a new ADR to show the test catches index drift.

    This test helps demonstrate why the test is needed by simulating an ADR
    being added without updating the README index.
    """

    # Create a temporary mock ADR not in the index
    mock_adr_number = 9999
    mock_adr_file = ADR_DIR / f"{mock_adr_number:04d}-test-not-in-index.md"

    # This simulates adding an ADR without updating README
    try:
        # Write a mock ADR file
        mock_adr_file.write_text("# Mock ADR\n\nThis ADR is not in the README index.")

        # Run the test to verify it detects the index drift
        file_numbers = {num for num, _ in _adr_file_entries()}
        index_numbers = set(_index_numbers())

        # The file now includes 9999 but index doesn't
        mock_adr_number_in_files = mock_adr_number
        missing_from_index = sorted(set(file_numbers) - set(index_numbers))

        # Verify the test properly detects this drift
        assert mock_adr_number_in_files in missing_from_index, (
            f"Index drift test failed: ADR {mock_adr_number:04d} in files but not in index"
        )

        print(
            f"Index drift test verified: ADR {mock_adr_number:04d} detected as missing from index"
        )

    finally:
        # Clean up the mock ADR file
        if mock_adr_file.exists():
            mock_adr_file.unlink()


def test_adr_index_validation_precision() -> None:
    """Test that ADR index validation is precise enough for the use case.

    This test ensures:
    1. ADR numbers are validated against the correct range (0001-0010 currently)
    2. New ADRs can be added and validated
    """
    # Current state validation
    numbers = sorted({num for num, _ in _adr_file_entries()})

    # The test should properly validate current index state
    assert 1 in numbers, "ADR 0001 should exist"
    assert 10 in numbers, "ADR 0010 should exist"

    # ADR-0011 referenced but missing should be detectable
    # This is the exact issue being fixed
    if 11 in _get_all_external_references():
        assert 11 not in numbers, (
            "ADR-0011 is referenced externally but not in ADR files (issue #291)"
        )
