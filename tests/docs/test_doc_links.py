"""Validate internal Markdown cross-references under ``docs/`` resolve.

The docs tree is heavily cross-linked: ``docs/PHILOSOPHY.md`` points at
``../AGENTS.md``, ``./SECURITY.md`` and ``./adr/``; ``docs/adr/README.md``
links every ADR row by relative path; ``docs/CONTEXT.md`` and
``docs/ideas/README.md`` link back into the rest of the tree. Today a scan
finds 0 broken relative links, but nothing enforces that. The pre-commit
hooks (``ruff``, ``gitleaks``, hygiene) and the pytest gate do not look at
Markdown links, so the first agent-authored PR that renames or moves an ADR
severs the "faithful map" ``docs/adr/0001-record-architecture-decisions.md``
promises and ships uncaught.

This module makes the integrity check part of the existing pytest gate
(ADR-0005). For every ``*.md`` under ``docs/`` it resolves each
``[label](relative)`` link against the filesystem, stripping ``#anchor``
fragments, and fails when the target does not exist. External ``http(s)`` /
``mailto`` links are out of scope (no network in CI). Pure ``pathlib`` +
``re`` — no new dependency, consistent with ADR-0002.

See ``docs/adr/0005-pytest-as-evaluation-framework.md``. Tracks issue #34.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repo root: tests/docs/test_doc_links.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

# A Markdown link: optional leading ``!`` (image), a [label], then (target).
# The target group keeps any ``url "title"`` suffix; callers split it.
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)]*)\)")

# A target that begins with a scheme (``http:``, ``https:``, ``mailto:``,
# ``ftp:`` ...) is external and is not resolved against the filesystem.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")

# Inline code spans (`` `...` ``) hold literal text, not links.
_INLINE_CODE_RE = re.compile(r"`[^`]*`")

# Opening marker of a fenced code block: at least three ````` or ``~~~```.
_FENCE_RE = re.compile(r"(`{3,}|~{3,})")


def _resolve_internal_target(source: Path, raw_target: str) -> Path | None:
    """Resolve ``raw_target`` relative to ``source``.

    Returns the absolute path the link points at, or ``None`` when the link is
    external (``http:``/``https:``/...) or is an intra-file anchor (``#sec``),
    both of which are intentionally not validated here.
    """
    tokens = raw_target.split()
    target = tokens[0] if tokens else ""
    target = target.split("#", 1)[0]  # drop any ``#anchor`` fragment
    if not target or _SCHEME_RE.match(target):
        return None
    return (source.parent / target).resolve()


def _collect_internal_links() -> list[tuple[str, int, str, str]]:
    """Return ``(file, lineno, label, target)`` for each internal link in docs/.

    Fenced code blocks (``` ``` ... ``` ```) and inline code spans are skipped
    so link-shaped literals inside them are not mistaken for references.
    External and anchor-only links are filtered out so every returned entry is
    a real filesystem cross-reference that the test must verify.
    """
    results: list[tuple[str, int, str, str]] = []
    for md in sorted(DOCS_DIR.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        in_fence = False
        fence_char = ""
        for lineno, line in enumerate(text.splitlines(), start=1):
            fence = _FENCE_RE.match(line.lstrip())
            if fence:
                char = fence.group(1)[0]
                if not in_fence:
                    in_fence = True
                    fence_char = char
                elif char == fence_char:
                    in_fence = False
                    fence_char = ""
                continue
            if in_fence:
                continue
            searchable = _INLINE_CODE_RE.sub("", line)
            for match in _LINK_RE.finditer(searchable):
                label, raw_target = match.group(1), match.group(2)
                if _resolve_internal_target(md, raw_target) is None:
                    continue
                results.append((str(md.relative_to(REPO_ROOT)), lineno, label, raw_target))
    return results


_LINKS = _collect_internal_links()


@pytest.mark.parametrize(
    ("file", "lineno", "label", "target"),
    _LINKS,
    ids=[f"{link[0]}:{link[1]}:{link[2]}" for link in _LINKS],
)
def test_internal_doc_link_resolves(file: str, lineno: int, label: str, target: str) -> None:
    """Every internal ``[label](relative)`` link must point at an existing path.

    Fails when a Markdown file under ``docs/`` references a file or directory
    that has been renamed, moved, or removed. The error names the source,
    line, label and the missing destination so a reviewer can fix it without
    grepping. Grounding: ``docs/adr/0001-record-architecture-decisions.md``
    (the ``docs/adr/`` index must stay a faithful map).
    """
    source = REPO_ROOT / file
    resolved = _resolve_internal_target(source, target)
    assert resolved is not None  # filtered at collection time; invariant hold
    assert resolved.exists(), (
        f"{file}:{lineno}: broken Markdown link [{label}]({target}) -> "
        f"{resolved}. Restore the target or update the link."
    )


def test_internal_doc_links_were_found() -> None:
    """Sanity guard: the collector must find at least one internal link.

    If this fails the link regex or the docs layout changed and
    ``test_internal_doc_link_resolves`` is silently asserting nothing. Update
    the regex or the scan root (ADR-0005).
    """
    assert _LINKS, (
        "No internal Markdown links found under docs/. Either docs/ lost its "
        "cross-references or _LINK_RE no longer matches the link syntax."
    )
