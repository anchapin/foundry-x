"""Controlled-fetch hook for agent documentation access (issue #1054).

Implements the ``Hook.pre_tool`` slot to validate every ``web_fetch`` tool
call's URL against the operator-configured ``FETCH_ALLOWED_DOMAINS``
allowlist before the HTTP request is issued. This is the enforcement point
for the SECURITY.md "Controlled fetch" threat model:

* The Docker sandbox isolates the agent from the host network. Without a
  controlled fetch path the agent cannot retrieve library documentation,
  man pages, or reference material during a session.
* An unrestricted fetch capability would violate the sandbox boundary
  (SECURITY.md threat #6, local privilege) and open an SSRF / data-exfil
  channel.

The hook follows the same self-registration pattern as
``InjectionFirewallHook`` and ``RateLimitHook``: importing this module
registers a singleton instance against the process-default registry.

Self-reference constraint (AGENTS.md section 7)
-----------------------------------------------
This module imports **nothing** from ``src/foundry_x/``. The harness is
the artifact being evolved; the foundry is the machinery that evolves
it. A harness-to-foundry import would close that loop in the wrong
direction.

Seed status (issue #1054)
-------------------------
This file is a **seed** created per issue #1054's explicit request. It is
a new file landing the controlled-fetch policy, not hand-edited DNA in
the sense of AGENTS.md section 2. Future evolution runs may modify it
through the normal ``Evolver`` -> ``Critic`` pipeline (ADR-0004).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from .base import HookRegistry, ToolCall, ToolResult, register_hook

# ---------------------------------------------------------------------------
# Environment variable consumed by the allowlist resolver
# ---------------------------------------------------------------------------
_FETCH_ALLOWED_DOMAINS_ENV = "FETCH_ALLOWED_DOMAINS"

# Sentinel key injected into ``call.arguments`` when the hook blocks a
# fetch. The runner's skill executor checks for this key and short-circuits
# to an error result without issuing the HTTP request.
_FETCH_BLOCKED_KEY = "__fetch_blocked"

# Allowed URL schemes for the fetch tool. Only HTTP(S) is permitted so
# the agent cannot use file://, ftp://, or other schemes to read local
# resources or pivot protocols.
_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def resolve_allowed_domains(
    env: Mapping[str, str] | None = None,
) -> frozenset[str]:
    """Return the set of allowed domains from ``FETCH_ALLOWED_DOMAINS``.

    The value is a comma-separated list of domain names, case-insensitive.
    Whitespace around each entry is stripped. An empty or unset variable
    yields an empty set, which means **deny all** — the secure default
    when no allowlist has been configured.
    """
    source = env if env is not None else os.environ
    raw = source.get(_FETCH_ALLOWED_DOMAINS_ENV, "")
    return frozenset(entry.strip().lower() for entry in raw.split(",") if entry.strip())


def is_url_allowed(
    url: str,
    allowed_domains: frozenset[str],
) -> bool:
    """Check whether ``url`` should be permitted by the allowlist.

    Rules:
    - Empty allowlist → deny all (secure default).
    - Non-HTTP(S) scheme → deny.
    - URL without a host → deny.
    - Host (case-insensitive) must be an exact match in ``allowed_domains``.
    """
    if not allowed_domains:
        return False
    if not url:
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    return host in allowed_domains


class WebFetchHook:
    """Hook that enforces the ``FETCH_ALLOWED_DOMAINS`` allowlist (issue #1054).

    Implements the ``Hook`` protocol. ``pre_tool`` inspects every
    ``web_fetch`` tool call, validates the URL's host against the allowlist,
    and — if the domain is not permitted — records a ``fetch_blocked``
    trace event (via the optional ``tracer`` callback) and replaces the
    URL in the call arguments with an empty string plus a sentinel key
    so the skill executor returns an error instead of issuing the HTTP
    request.

    ``post_tool`` is a pass-through; the hook does not modify fetch results.
    """

    _phase: int = 3

    __slots__ = ("_tracer",)

    def __init__(
        self,
        tracer: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._tracer = tracer

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        if call.name != "web_fetch":
            return call

        url: str = call.arguments.get("url", "")
        allowed = resolve_allowed_domains()

        if is_url_allowed(url, allowed):
            return call

        # Domain not allowed (or allowlist empty). Emit a ``fetch_blocked``
        # trace event so the Digester and operator can observe the block,
        # then mutate the call so the executor short-circuits.
        if self._tracer is not None:
            self._tracer(
                {
                    "kind": "fetch_blocked",
                    "url": url,
                    "reason": "domain_not_in_allowlist",
                    "allowed_domains": sorted(allowed),
                }
            )

        new_args: dict[str, Any] = dict(call.arguments)
        new_args["url"] = ""
        new_args[_FETCH_BLOCKED_KEY] = True
        return ToolCall(name=call.name, arguments=new_args)

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        return result


# ---------------------------------------------------------------------------
# Self-registration (mirrors injection_firewall.py / rate_limit.py pattern)
# ---------------------------------------------------------------------------

_web_fetch_hook_instance: WebFetchHook | None = None


def _get_hook() -> WebFetchHook:
    global _web_fetch_hook_instance
    if _web_fetch_hook_instance is None:
        _web_fetch_hook_instance = WebFetchHook()
    return _web_fetch_hook_instance


def register_into(registry: HookRegistry) -> WebFetchHook:
    hook = _get_hook()
    registry.register(hook)
    return hook


register_hook(_get_hook())


__all__ = [
    "WebFetchHook",
    "is_url_allowed",
    "register_into",
    "resolve_allowed_domains",
]
