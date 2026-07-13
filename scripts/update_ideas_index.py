#!/usr/bin/env python3
"""Update docs/ideas/README.md with the current ideas index.

Run manually: python scripts/update_ideas_index.py
Install as pre-commit hook: scripts/update_ideas_index.py runs automatically
                            via the pre-commit configuration.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


IDEAS_DIR = Path("docs/ideas")
README_PATH = Path("docs/ideas/README.md")
INDEX_START = "<!-- entries inserted by scripts/update_ideas_index.py -->"
INDEX_END = "<!-- end of ideas index -->"


def extract_status(content: str) -> str:
    m = re.search(r"^## Status\s*\n\s*(.+?)\s*(?:\n|$)", content, re.MULTILINE)
    if m:
        value = m.group(1).strip()
        if value:
            return value
    return "Proposed"


def extract_title(content: str) -> str | None:
    m = re.search(r"^# Idea \d+: (.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def parse_idea_file(path: Path) -> dict | None:
    num_m = re.match(r"^(\d{4})-.+\.md$", path.name)
    if not num_m:
        return None
    content = path.read_text(encoding="utf-8")
    title = extract_title(content)
    if not title:
        return None
    return {
        "number": int(num_m.group(1)),
        "title": title,
        "status": extract_status(content),
        "file": path.name,
    }


def build_index(ideas: list[dict]) -> str:
    if not ideas:
        return ""
    lines = []
    for idea in ideas:
        slug = idea["file"]
        lines.append(f"| [{idea['number']:04d}](./{slug}) | {idea['title']} | {idea['status']} |")
    return "\n".join(lines) + "\n"


def update_readme(index_lines: str) -> None:
    content = README_PATH.read_text(encoding="utf-8")
    if INDEX_END in content:
        before = content[: content.find(INDEX_START)]
        after = content[content.find(INDEX_END) + len(INDEX_END) :]
        new_content = (
            before
            + INDEX_START
            + "\n"
            + index_lines
            + ("" if index_lines.endswith("\n") else "\n")
            + after
        )
    else:
        new_content = content
    README_PATH.write_text(new_content, encoding="utf-8")


def main() -> int:
    if not IDEAS_DIR.is_dir():
        print(f"ERROR: {IDEAS_DIR} does not exist", file=sys.stderr)
        return 1

    ideas: list[dict] = []
    for path in sorted(IDEAS_DIR.glob("[0-9][0-9][0-9][0-9]-*.md")):
        idea = parse_idea_file(path)
        if idea:
            ideas.append(idea)

    ideas.sort(key=lambda x: x["number"])
    index_lines = build_index(ideas)
    update_readme(index_lines)

    if ideas:
        print(f"Updated ideas index with {len(ideas)} idea(s).")
    else:
        print("No ideas found; index cleared.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
