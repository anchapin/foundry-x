"""Guard test: ADR index integrity and sequential numbering.

ADR-0001 (docs/adr/0001-record-architecture-decisions.md:16-17) requires
that ADRs be "numbered sequentially with a short kebab-case slug", and
docs/adr/README.md:24 adds: "When adding a new ADR, append it to this
table in the same PR." Until now nothing verified either rule. The moment
an agent or human lands ADR-0009 without editing README.md — or reuses a
colliding number across two slug files — the drift is silent and the
``docs/adr/`` directory stops being the "faithful map" ADR-0001:39-40
promises.

This module asserts (a) every ``NNNN-*.md`` file under ``docs/adr/`` has
a row in the ``README.md`` index table and vice-versa, and (b) the set of
numbers is gap-free starting at 0001 with no duplicates. Pure ``pathlib``
+ ``re`` over the existing ``pytest`` toolchain — no new dependency. See
ADR-0005 (pytest as the unified evaluation framework). Tracks issue #33.
"""

from __future__ import annotations

import re
from pathlib import Path

# tests/docs/test_adr_index.py -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "adr"
README = ADR_DIR / "README.md"

# An ADR file is NNNN-kebab-case-slug.md. README.md is the index, not an ADR.
_ADR_FILE_RE = re.compile(r"^(\d{4})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
# An index row links the number to its file: | [NNNN](./NNNN-slug.md) | ... |
_INDEX_ROW_RE = re.compile(r"\[(\d{4})\]\(\./\d{4}-[a-z0-9-]+\.md\)")


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
