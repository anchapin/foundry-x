"""Guard test: project vocabulary must be defined in CONTEXT.md.

``docs/CONTEXT.md`` declares itself the source of truth for project
vocabulary (CONTEXT.md:8-9): "If you introduce a new term, add it here
in the same PR that introduces the concept." That rule was previously
unenforced. This test maintains an allowlist of load-bearing terms and
asserts each one has a ``**Term** —`` glossary heading in CONTEXT.md,
so introducing a new term without defining it is caught at test time.

Pure string matching, no new dependency. See ADR-0005 (pytest as the
unified evaluation framework) and issue #36.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root: tests/docs/test_context_glossary.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_MD = REPO_ROOT / "docs" / "CONTEXT.md"

# Load-bearing project terms that must carry their own glossary heading.
# Add a term here in the same PR that introduces the concept it names;
# the test then fails until a matching ``**<term>** —`` heading exists.
REQUIRED_TERMS: tuple[str, ...] = (
    "meta-agent",
    "failure report",
)

# A glossary heading is a markdown list item whose bolded lead term is
# followed by an em dash, e.g. ``- **FoundryX** — the framework ...``.
_HEADING_RE = re.compile(r"^\s*-\s*\*\*(.+?)\*\*\s*—", re.MULTILINE)


def _glossary_headings(text: str) -> set[str]:
    """Return the set of terms that appear as ``**Term** —`` headings."""
    return {m.group(1) for m in _HEADING_RE.finditer(text)}


def test_context_md_exists():
    """CONTEXT.md must be present so the vocabulary guard can run."""
    assert CONTEXT_MD.is_file(), f"missing glossary: {CONTEXT_MD}"


def test_required_terms_have_glossary_headings():
    """Every allowlisted term must be defined as a glossary heading.

    Fails if an entry is removed from CONTEXT.md or if a term is added
    to ``REQUIRED_TERMS`` without a matching ``**<term>** —`` heading.
    Grounding: docs/CONTEXT.md:8-9, docs/PRD.md:25, docs/ROADMAP.md:15.
    """
    headings = _glossary_headings(CONTEXT_MD.read_text(encoding="utf-8"))
    missing = [term for term in REQUIRED_TERMS if term not in headings]
    assert not missing, (
        "REQUIRED_TERMS without a `**Term** —` glossary heading in "
        f"CONTEXT.md: {missing}. Add each as "
        "`- **<term>** — <one sentence>` under the appropriate section "
        "(docs/CONTEXT.md:8-9)."
    )
