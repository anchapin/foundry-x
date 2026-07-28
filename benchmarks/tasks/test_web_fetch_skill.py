"""Benchmark task: gate the web_fetch skill allowlist contract (issue #1054).

The ``web_fetch`` skill lets the agent retrieve documentation from
operator-allowlisted domains. Its security model is deny-by-default: an
empty ``FETCH_ALLOWED_DOMAINS`` blocks every fetch, and the
``WebFetchHook`` short-circuits blocked calls before any HTTP request is
issued. A regression that weakens the allowlist (e.g. implicit subdomain
matching, accepting a non-HTTP scheme, or letting the call through when
the allowlist is empty) would silently reopen the SSRF / data-exfil
channel documented in SECURITY.md threat #7 and pass every existing
benchmark.

This module closes that coverage gap with a deterministic, fully offline
benchmark: it exercises the pure allowlist logic and the hook's
``pre_tool`` blocking path directly, with no network access. It also pins
that the skill and hook are registered in ``manifest.json`` so a drift
between the manifest and disk surfaces at the Critic gate (issue #277,
issue #1010).

See ADR-0004 (Critic gate), ADR-0005 (pytest as evaluation framework),
and SECURITY.md threat #7.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from benchmarks.models import BenchmarkTask

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "harness"

TASK = BenchmarkTask(
    name="web_fetch_skill",
    description=(
        "Gate the web_fetch allowlist contract: deny-by-default, scheme "
        "guard, exact host match, hook blocking, and manifest registration."
    ),
    prompt=(
        "Verify the web_fetch skill's security contract: (1) an empty "
        "FETCH_ALLOWED_DOMAINS denies all URLs; (2) only http(s) schemes are "
        "allowed; (3) the host must exactly match an allowlist entry; (4) the "
        "WebFetchHook clears the URL and sets the sentinel on a blocked call "
        "and emits a fetch_blocked trace event; (5) the skill and hook are "
        "registered in harness/manifest.json."
    ),
    difficulty_tier="smoke",
    expected_outcome=(
        "All allowlist invariants hold and the hook's blocking path produces "
        "the fetch_blocked tracer event, the cleared URL, and the sentinel "
        "key; the manifest declares both the web_fetch skill and hook."
    ),
    requires_skills=["web_fetch"],
    tags=["web_fetch", "allowlist", "security", "ssrf", "hook"],
)


@pytest.mark.benchmark
def test_web_fetch_allowlist_contract() -> None:
    """Deterministic, offline gate for the web_fetch allowlist contract.

    Exercises the pure functions (``resolve_allowed_domains`` /
    ``is_url_allowed``) and the hook's ``pre_tool`` blocking path directly.
    No HTTP request is ever issued, so the check runs offline under the
    ``MockModelAdapter`` and is stable across environments.
    """
    import asyncio

    from harness.hooks.base import ToolCall
    from harness.hooks.web_fetch import (
        WebFetchHook,
        is_url_allowed,
        resolve_allowed_domains,
    )

    ALLOWED = frozenset({"docs.python.org", "man7.org"})

    # --- deny-by-default: empty allowlist denies everything -----------------
    empty = frozenset()
    assert is_url_allowed("https://docs.python.org/3/", empty) is False
    assert is_url_allowed("", ALLOWED) is False

    # --- scheme guard: only http(s) permitted -------------------------------
    assert is_url_allowed("file:///etc/passwd", ALLOWED) is False
    assert is_url_allowed("ftp://man7.org/linux/man-pages/", ALLOWED) is False

    # --- exact host match (no implicit subdomain matching) ------------------
    assert is_url_allowed("https://docs.python.org/3/library/os.html", ALLOWED) is True
    assert is_url_allowed("https://sub.docs.python.org/", ALLOWED) is False
    assert is_url_allowed("https://evil.com/", ALLOWED) is False

    # --- resolve_allowed_domains parsing ------------------------------------
    assert resolve_allowed_domains({}) == frozenset()
    assert (
        resolve_allowed_domains({"FETCH_ALLOWED_DOMAINS": "Docs.Python.Org, man7.org"}) == ALLOWED
    )

    # --- hook blocks a non-allowlisted call before any HTTP request ---------
    # The hook reads FETCH_ALLOWED_DOMAINS from os.environ, so drive both the
    # block path (host not in allowlist) and the pass-through path (host in
    # allowlist) under an explicit, populated allowlist.
    events: list[dict[str, object]] = []

    def _tracer(event_name: str, payload: dict[str, object]) -> None:
        events.append({"kind": event_name, **payload})

    hook = WebFetchHook(tracer=_tracer)
    with mock.patch.dict(os.environ, {"FETCH_ALLOWED_DOMAINS": "docs.python.org,man7.org"}):
        blocked_call = ToolCall(name="web_fetch", arguments={"url": "https://evil.com/exfil"})
        result = asyncio.run(hook.pre_tool(blocked_call))

        # URL is cleared and the sentinel is set so the executor short-circuits.
        assert result.arguments["url"] == ""
        assert result.arguments["__fetch_blocked"] is True
        # A fetch_blocked trace event was emitted carrying the block reason.
        blocked = [e for e in events if e.get("kind") == "fetch_blocked"]
        assert len(blocked) == 1
        assert blocked[0]["reason"] == "domain_not_in_allowlist"
        assert blocked[0]["url"] == "https://evil.com/exfil"

        # --- hook passes an allowed call through unchanged ------------------
        allowed_call = ToolCall(name="web_fetch", arguments={"url": "https://man7.org/"})
        passed = asyncio.run(hook.pre_tool(allowed_call))
        assert passed.arguments["url"] == "https://man7.org/"
        assert "__fetch_blocked" not in passed.arguments


@pytest.mark.benchmark
def test_web_fetch_skill_and_hook_registered_in_manifest() -> None:
    """The manifest must declare the web_fetch skill and hook (issue #277).

    The runner globs skills by filename and dispatches hooks by manifest
    declaration. A drift between ``manifest.json`` and disk silently drops
    the skill or disables the allowlist hook at runtime, so the benchmark
    pins both directions of the cross-reference.
    """
    manifest = json.loads((HARNESS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert "web_fetch" in manifest["hooks"], "web_fetch hook missing from manifest hooks"
    assert "web_fetch.json" in manifest["skills"], "web_fetch.json missing from manifest skills"
    assert (HARNESS_DIR / "skills" / "web_fetch.json").is_file(), (
        "harness/skills/web_fetch.json missing on disk"
    )
    assert (HARNESS_DIR / "hooks" / "web_fetch.py").is_file(), (
        "harness/hooks/web_fetch.py missing on disk"
    )
