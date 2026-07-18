"""Golden solution for the read_multiple_files_skill benchmark (issue #873).

This file mirrors the contract declared in
``harness/skills/read_multiple_files.json``: it accepts the same input
schema (``paths`` plus optional ``max_bytes`` / ``max_lines`` / ``offset``)
and returns the same output schema (``results`` ordered by ``paths``,
``truncated`` as the OR over per-file truncations, ``error`` populated only
when the call itself fails, e.g. when ``paths`` is empty). Each result entry
uses the ``read_file`` result shape declared in
``harness/skills/read_file.json``.

The benchmark is intentionally stdlib-only; the simplification that
truncation cuts at the byte boundary (not aligned to ``\\n``) matches the
sibling ``read_file_skill/solution.py``. What the benchmark pins is the
*interface surface* (per-file result shape, per-file ``error`` location,
aggregate ``truncated`` semantics) rather than the exact byte-alignment
policy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_file(
    path: Path,
    max_bytes: int | None = None,
    max_lines: int | None = None,
    offset: int = 0,
) -> dict[str, object]:
    """Mirror the read_file skill contract (per-file result entry shape)."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {
            "content": "",
            "sha256": _sha256(b""),
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": f"file not found: {path.name}",
        }
    except PermissionError:
        return {
            "content": "",
            "sha256": _sha256(b""),
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": f"permission denied: {path.name}",
        }

    bytes_total = len(raw)
    if offset >= bytes_total:
        return {
            "content": "",
            "sha256": _sha256(b""),
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": bytes_total,
            "error": None,
        }

    slice_ = raw[offset:]
    if max_bytes is not None and len(slice_) > max_bytes:
        slice_ = slice_[:max_bytes]
        truncated = True
    else:
        truncated = False

    content = slice_.decode("utf-8", errors="replace")
    if max_lines is not None:
        lines = content.split("\n")
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            truncated = True

    return {
        "content": content,
        "sha256": _sha256(content.encode("utf-8")),
        "truncated": truncated,
        "bytes_returned": len(content.encode("utf-8")),
        "bytes_total": bytes_total,
        "error": None,
    }


def read_multiple_files(
    paths: list[str],
    max_bytes: int | None = None,
    max_lines: int | None = None,
    offset: int = 0,
) -> dict[str, object]:
    """Mirror the read_multiple_files skill contract.

    Returns ``{results, truncated, error}`` where ``results`` is a list of
    per-file entries in the same order as ``paths``, ``truncated`` is the
    OR over per-file truncations, and ``error`` is non-null only when the
    call itself fails (e.g. empty ``paths``). Per-file errors live inside
    each result entry -- never at the top level.
    """
    if not paths:
        return {"results": [], "truncated": False, "error": "paths must not be empty"}

    results = [_read_file(Path(p), max_bytes, max_lines, offset) for p in paths]
    truncated = any(bool(r["truncated"]) for r in results)
    return {"results": results, "truncated": truncated, "error": None}


def _emit_case1(out_lines: list[str]) -> None:
    """multi_file_basic: 3 mixed-content files in one batch, all present."""
    out_lines.append("== multi_file_basic ==")
    response = read_multiple_files(
        paths=["app.py", "data.json", "notes.md"],
        max_bytes=65536,
    )
    assert response["error"] is None, f"call failed: {response['error']}"
    assert response["truncated"] is False, "unexpected aggregate truncation"
    for entry in response["results"]:
        out_lines.append(f"[{'ERR' if entry['error'] else 'OK'}]")
        out_lines.append(f"truncated={entry['truncated']}")
        out_lines.append(f"content={entry['content'].rstrip(chr(10))}")


def _emit_case2(out_lines: list[str]) -> None:
    """missing_file_in_batch: 3 paths, 2 present + 1 absent."""
    out_lines.append("== missing_file_in_batch ==")
    response = read_multiple_files(
        paths=["present.py", "missing.py", "settings.ini"],
        max_bytes=65536,
    )
    # The call itself succeeds; only the missing path carries a per-file error.
    assert response["error"] is None, f"call failed: {response['error']}"
    assert response["truncated"] is False, "unexpected aggregate truncation"
    assert len(response["results"]) == 3
    for entry in response["results"]:
        if entry["error"] is None:
            out_lines.append("RESULT: ok")
            out_lines.append(f"content={entry['content'].rstrip(chr(10))}")
        else:
            out_lines.append(f"RESULT: error={entry['error']}")


def _emit_case3(out_lines: list[str]) -> None:
    """truncation_aggregate: small file + large file → top-level truncated."""
    out_lines.append("== truncation_aggregate ==")
    # Tight per-file cap so the large module is truncated but the small
    # text file is read in full.
    response = read_multiple_files(
        paths=["small.txt", "large_module.py"],
        max_bytes=2048,
        max_lines=30,
    )
    assert response["error"] is None, f"call failed: {response['error']}"
    # The aggregate flag must be true because at least one file was truncated.
    assert response["truncated"] is True, "top-level truncated should be true"
    out_lines.append(f"truncated={response['truncated']}")
    for entry in response["results"]:
        out_lines.append(f"file_truncated={entry['truncated']}")
        out_lines.append(f"bytes_returned={entry['bytes_returned']}")
        out_lines.append(f"bytes_total={entry['bytes_total']}")


def main() -> None:
    out_lines: list[str] = []

    # Cases are mutually exclusive by the presence of their fixture files --
    # mirroring how the real agent would branch on the workspace state.
    if (
        Path("app.py").exists()
        and Path("data.json").exists()
        and Path("notes.md").exists()
    ):
        _emit_case1(out_lines)
    elif (
        Path("present.py").exists()
        and Path("settings.ini").exists()
        and not Path("missing.py").exists()
    ):
        _emit_case2(out_lines)
    elif Path("small.txt").exists() and Path("large_module.py").exists():
        _emit_case3(out_lines)

    Path("output.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
