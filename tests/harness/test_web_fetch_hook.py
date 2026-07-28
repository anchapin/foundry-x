"""Tests for ``WebFetchHook`` URL allowlist validation (issue #1054).

Covers:
- ``resolve_allowed_domains`` parsing from the ``FETCH_ALLOWED_DOMAINS`` env var
- ``is_url_allowed`` scheme, host, and allowlist checks
- ``WebFetchHook.pre_tool`` blocking behaviour (URL cleared, sentinel set, tracer fired)
- ``WebFetchHook.pre_tool`` pass-through for allowed domains
- ``WebFetchHook.post_tool`` is a no-op pass-through
"""

from __future__ import annotations

from unittest import mock

import pytest

from harness.hooks.base import ToolCall, ToolResult
from harness.hooks.web_fetch import (
    WebFetchHook,
    is_url_allowed,
    resolve_allowed_domains,
)

# ---------------------------------------------------------------------------
# resolve_allowed_domains
# ---------------------------------------------------------------------------


class TestResolveAllowedDomains:
    def test_empty_env_yields_empty_set(self) -> None:
        assert resolve_allowed_domains({}) == frozenset()

    def test_unset_env_yields_empty_set(self) -> None:
        assert resolve_allowed_domains({"OTHER_VAR": "x"}) == frozenset()

    def test_single_domain(self) -> None:
        result = resolve_allowed_domains({"FETCH_ALLOWED_DOMAINS": "docs.python.org"})
        assert result == frozenset({"docs.python.org"})

    def test_multiple_domains_comma_separated(self) -> None:
        result = resolve_allowed_domains(
            {"FETCH_ALLOWED_DOMAINS": "docs.python.org,man7.org,developer.mozilla.org"}
        )
        assert result == frozenset({"docs.python.org", "man7.org", "developer.mozilla.org"})

    def test_whitespace_is_stripped(self) -> None:
        result = resolve_allowed_domains(
            {"FETCH_ALLOWED_DOMAINS": "  docs.python.org , man7.org  "}
        )
        assert result == frozenset({"docs.python.org", "man7.org"})

    def test_case_is_lowered(self) -> None:
        result = resolve_allowed_domains({"FETCH_ALLOWED_DOMAINS": "Docs.Python.Org,MAN7.ORG"})
        assert result == frozenset({"docs.python.org", "man7.org"})

    def test_empty_entries_are_ignored(self) -> None:
        result = resolve_allowed_domains({"FETCH_ALLOWED_DOMAINS": "docs.python.org,, ,man7.org"})
        assert result == frozenset({"docs.python.org", "man7.org"})


# ---------------------------------------------------------------------------
# is_url_allowed
# ---------------------------------------------------------------------------


class TestIsUrlAllowed:
    def test_empty_allowlist_denies_all(self) -> None:
        assert not is_url_allowed("https://docs.python.org", frozenset())

    def test_empty_url_denied(self) -> None:
        assert not is_url_allowed("", frozenset({"docs.python.org"}))

    def test_allowed_domain(self) -> None:
        assert is_url_allowed("https://docs.python.org/3/", frozenset({"docs.python.org"}))

    def test_disallowed_domain(self) -> None:
        assert not is_url_allowed("https://evil.example.com/exfil", frozenset({"docs.python.org"}))

    def test_case_insensitive_host(self) -> None:
        assert is_url_allowed("https://DOCS.PYTHON.ORG/3/", frozenset({"docs.python.org"}))

    def test_http_scheme_allowed(self) -> None:
        assert is_url_allowed("http://man7.org/linux/man-pages/", frozenset({"man7.org"}))

    def test_file_scheme_denied(self) -> None:
        assert not is_url_allowed("file:///etc/passwd", frozenset({"etc"}))

    def test_ftp_scheme_denied(self) -> None:
        assert not is_url_allowed("ftp://man7.org/file", frozenset({"man7.org"}))

    def test_no_host_denied(self) -> None:
        assert not is_url_allowed("https:///path", frozenset({"docs.python.org"}))

    def test_subdomain_not_implicitly_allowed(self) -> None:
        """``docs.python.org`` in the allowlist does not allow ``sub.docs.python.org``."""
        assert not is_url_allowed("https://sub.docs.python.org/", frozenset({"docs.python.org"}))


# ---------------------------------------------------------------------------
# WebFetchHook.pre_tool
# ---------------------------------------------------------------------------


class TestWebFetchHookPreTool:
    @pytest.mark.asyncio
    async def test_non_fetch_call_passes_through(self) -> None:
        hook = WebFetchHook()
        call = ToolCall(name="bash", arguments={"command": "echo hi"})
        result = await hook.pre_tool(call)
        assert result is call
        assert result.name == "bash"

    @pytest.mark.asyncio
    async def test_allowed_domain_passes_through(self) -> None:
        with mock.patch.dict("os.environ", {"FETCH_ALLOWED_DOMAINS": "docs.python.org"}):
            hook = WebFetchHook()
            call = ToolCall(name="web_fetch", arguments={"url": "https://docs.python.org/3/"})
            result = await hook.pre_tool(call)
        assert result.name == "web_fetch"
        assert result.arguments["url"] == "https://docs.python.org/3/"
        assert "__fetch_blocked" not in result.arguments

    @pytest.mark.asyncio
    async def test_disallowed_domain_blocked(self) -> None:
        with mock.patch.dict("os.environ", {"FETCH_ALLOWED_DOMAINS": "docs.python.org"}):
            hook = WebFetchHook()
            call = ToolCall(name="web_fetch", arguments={"url": "https://evil.example.com/"})
            result = await hook.pre_tool(call)
        assert result.arguments["url"] == ""
        assert result.arguments["__fetch_blocked"] is True

    @pytest.mark.asyncio
    async def test_empty_allowlist_blocks_all(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            hook = WebFetchHook()
            call = ToolCall(name="web_fetch", arguments={"url": "https://docs.python.org/"})
            result = await hook.pre_tool(call)
        assert result.arguments["url"] == ""
        assert result.arguments["__fetch_blocked"] is True

    @pytest.mark.asyncio
    async def test_blocked_fetch_fires_tracer(self) -> None:
        tracer_calls: list[tuple[str, dict[str, object]]] = []

        def tracer(kind: str, payload: dict[str, object]) -> None:
            tracer_calls.append((kind, payload))

        with mock.patch.dict("os.environ", {"FETCH_ALLOWED_DOMAINS": "docs.python.org"}):
            hook = WebFetchHook(tracer=tracer)
            call = ToolCall(name="web_fetch", arguments={"url": "https://evil.example.com/"})
            await hook.pre_tool(call)

        assert len(tracer_calls) == 1
        kind, payload = tracer_calls[0]
        assert kind == "fetch_blocked"
        assert payload["url"] == "https://evil.example.com/"
        assert payload["reason"] == "domain_not_in_allowlist"
        assert payload["allowed_domains"] == ["docs.python.org"]

    @pytest.mark.asyncio
    async def test_allowed_fetch_does_not_fire_tracer(self) -> None:
        tracer_calls: list[tuple[str, dict[str, object]]] = []

        def tracer(kind: str, payload: dict[str, object]) -> None:
            tracer_calls.append((kind, payload))

        with mock.patch.dict("os.environ", {"FETCH_ALLOWED_DOMAINS": "docs.python.org"}):
            hook = WebFetchHook(tracer=tracer)
            call = ToolCall(name="web_fetch", arguments={"url": "https://docs.python.org/"})
            await hook.pre_tool(call)

        assert tracer_calls == []

    @pytest.mark.asyncio
    async def test_no_tracer_does_not_crash(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            hook = WebFetchHook()
            call = ToolCall(name="web_fetch", arguments={"url": "https://evil.example.com/"})
            # Should not raise even though tracer is None
            result = await hook.pre_tool(call)
        assert result.arguments["__fetch_blocked"] is True


# ---------------------------------------------------------------------------
# WebFetchHook.post_tool
# ---------------------------------------------------------------------------


class TestWebFetchHookPostTool:
    @pytest.mark.asyncio
    async def test_post_tool_pass_through(self) -> None:
        hook = WebFetchHook()
        call = ToolCall(name="web_fetch", arguments={"url": "https://docs.python.org/"})
        result = ToolResult(name="web_fetch", output={"content": "hello"})
        returned = await hook.post_tool(call, result)
        assert returned is result
