"""Detect documentation claims about CI gates that are not actually enforced.

The guardrail prose in ``docs/SECURITY.md`` (and, by extension, the ADRs)
states that the Critic, Test, and Lint gates are ``enforced in CI`` and that
``uv pip audit`` runs ``in CI on every PR``. Until the ``infra`` subsystem
lands real workflows under ``.github/workflows/``, those claims are
aspirational, not enforced. A contributor who trusts them may skip the local
``uv run pytest`` / ``uv run ruff check .`` gate and let a regression through.

This module makes the docs honest about that gap. Every CI-enforcement claim
must be backed by either (a) at least one workflow file under
``.github/workflows/`` or (b) an inline ``ci-status: planned`` marker
acknowledging the claim is aspirational. Removing a marker before the
matching workflow exists turns this test red.

See ``docs/SECURITY.md`` and ADR-0005 (pytest as the unified evaluation
framework). Tracks issue #35.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

GUARDRAIL_PHRASES = ("enforced in CI", "CI blocks merges", "in CI on every PR")
PLANNED_MARKER = "ci-status: planned"

_PHRASE_RE = re.compile(
    "|".join(
        r"\s+".join(re.escape(word) for word in phrase.split()) for phrase in GUARDRAIL_PHRASES
    ),
    re.IGNORECASE,
)


def _scan_targets() -> list[Path]:
    files = sorted((DOCS_DIR / "adr").glob("*.md"))
    security = DOCS_DIR / "SECURITY.md"
    if security.is_file():
        files.append(security)
    return files


def _ci_workflows_exist() -> bool:
    if not WORKFLOWS_DIR.is_dir():
        return False
    return bool(list(WORKFLOWS_DIR.glob("*.yml")) or list(WORKFLOWS_DIR.glob("*.yaml")))


def _bullet_span(lines: list[str], anchor_line_1based: int) -> tuple[int, int]:
    anchor = anchor_line_1based - 1
    start = anchor
    while start > 0 and not lines[start].lstrip().startswith("- "):
        start -= 1
    end = anchor
    nxt = anchor + 1
    while nxt < len(lines):
        if lines[nxt].lstrip().startswith("- "):
            break
        if lines[nxt].strip() and not lines[nxt][0].isspace():
            break
        end = nxt
        nxt += 1
    return start + 1, end + 1


def _claims() -> list[tuple[str, int, str, str]]:
    results: list[tuple[str, int, str, str]] = []
    for path in _scan_targets():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for match in _PHRASE_RE.finditer(text):
            start_line = text.count("\n", 0, match.start()) + 1
            bullet_start, bullet_end = _bullet_span(lines, start_line)
            span_text = "\n".join(lines[bullet_start - 1 : bullet_end])
            results.append(
                (
                    str(path.relative_to(REPO_ROOT)),
                    start_line,
                    match.group(0),
                    span_text,
                )
            )
    return results


_CLAIMS = _claims()


@pytest.mark.parametrize(
    ("file", "lineno", "phrase", "span"),
    _CLAIMS,
    ids=[f"{c[0]}:{c[1]}" for c in _CLAIMS],
)
def test_ci_enforcement_claim_is_backed(file: str, lineno: int, phrase: str, span: str) -> None:
    if _ci_workflows_exist():
        return
    assert PLANNED_MARKER in span, (
        f'{file}:{lineno} claims CI enforcement ("{phrase}") but no workflow '
        f"exists under .github/workflows/. Either add the matching workflow or "
        f"acknowledge the claim is aspirational with the inline "
        f"'{PLANNED_MARKER}' marker."
    )


def test_guardrail_phrases_still_appear_in_docs() -> None:
    assert _CLAIMS, (
        "No CI-enforcement claims were found in the docs. If the guardrail "
        "wording changed, update GUARDRAIL_PHRASES so claims stay honest "
        "(see ADR-0005)."
    )
