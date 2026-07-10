"""Prompt-injection firewall hook (SECURITY.md "Prompt-input firewall").

Implements the ``Hook.post_tool`` slot to scan every ``ToolResult.output``
before it is re-injected into a model prompt. This is the enforcement point
for two SECURITY.md requirements:

* "Prompt-input firewall" (docs/SECURITY.md:47-50) — tool-call results that
  will be re-injected into a prompt must be checked for injection markers
  and either truncated or flagged for human review.
* "Prompt injection" (docs/SECURITY.md:88-89) — strip or escape role-tag
  sequences (``system:``, ``assistant:``, ``<|...|>``).

The detection patterns are a module-level constant so the Evolver can tune
them later without touching the hook contract.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .base import HookRegistry, ToolCall, ToolResult, register_hook

_log = logging.getLogger("harness.injection_firewall")

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------
# Ordered (name, source-pattern) tuples. Case-insensitive, compiled once at
# import. Module-level so the Evolver can tune them (issue #5 requirement).
# Categories mirror SECURITY.md threat #2 ("Prompt injection from traced
# content") and the markers enumerated at docs/SECURITY.md:48-49,88-89.
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    # Instruction-override phrases (docs/SECURITY.md:48).
    (
        "ignore_previous",
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    ),
    (
        "disregard_previous",
        r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    ),
    (
        "forget_previous",
        r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    ),
    (
        "new_instructions",
        r"(?:new|updated|real)\s+instructions\s*:",
    ),
    # Role-tag injection (docs/SECURITY.md:88-89).
    (
        "role_tag_colon",
        r"(?:^|\n|\r)\s*(?:system|assistant|developer|user)\s*:\s*",
    ),
    (
        "chatml_tag",
        r"<\|(?:im_start|im_end|system|assistant|user|begin_of_text|endoftext)\|>",
    ),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pat, re.IGNORECASE | re.MULTILINE)) for name, pat in INJECTION_PATTERNS
)

# Cap on how much of the original (now-suppressed) output we keep in the
# ``error`` field purely for human-review triage. The raw span is never
# placed back into ``output`` (which gets re-injected into the prompt).
_PREVIEW_LEN = 120


@dataclass(frozen=True)
class InjectionMatch:
    """A single detected injection marker within a tool result."""

    name: str
    pattern: str
    snippet: str


@dataclass(frozen=True)
class ScanResult:
    """Outcome of scanning one tool result for injection markers."""

    blocked: bool
    matches: tuple[InjectionMatch, ...]


def _coerce_text(output: Any) -> str:
    """Best-effort coercion of ``ToolResult.output`` to a scannable string."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return repr(output)
    except Exception:  # pragma: no cover - defensive, never swallow silently
        # Re-raise as RuntimeError so the trace pipeline surfaces it; bare
        # ``except: pass`` is forbidden by AGENTS.md §2.
        raise RuntimeError("injection_firewall: failed to coerce tool output") from None


def scan_for_injection(text: str) -> ScanResult:
    """Scan ``text`` for the markers in :data:`INJECTION_PATTERNS`.

    Returns every distinct match (a marker may fire more than once). This is
    a pure function so the Evolver and the test suite can exercise it
    independently of the hook machinery.
    """
    matches: list[InjectionMatch] = []
    for name, pattern in _COMPILED:
        source = ""
        for n, src in INJECTION_PATTERNS:
            if n == name:
                source = src
                break
        for m in pattern.finditer(text):
            matches.append(
                InjectionMatch(
                    name=name,
                    pattern=source,
                    snippet=m.group(0).strip()[:80],
                )
            )
    return ScanResult(blocked=bool(matches), matches=tuple(matches))


class InjectionFirewallHook:
    """Hook that truncates/flags tool results containing injection markers.

    Implements the :class:`~harness.hooks.base.Hook` Protocol. ``pre_tool``
    is an identity pass-through; ``post_tool`` is where the screening happens
    — it sees ``ToolResult.output`` before the caller re-injects it into a
    prompt (the interception point mandated by SECURITY.md).
    """

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        text = _coerce_text(result.output)
        scan = scan_for_injection(text)
        if not scan.blocked:
            return result

        names = ",".join(sorted({m.name for m in scan.matches}))
        _log.warning(
            "injection_firewall: blocked %d marker(s) [%s] from tool %r",
            len(scan.matches),
            names,
            result.name,
        )
        # ``output`` is re-injected into the prompt, so it must NOT carry the
        # adversarial span. ``error`` is the human-review flag channel and
        # keeps only a bounded preview for triage (never re-injected).
        preview = text[:_PREVIEW_LEN].replace("\n", " ")
        safe_output = (
            "[injection_firewall] output suppressed: injection markers "
            f"detected ({names}). Flagged for human review."
        )
        return ToolResult(
            name=result.name,
            output=safe_output,
            error=f"injection_detected:{names} preview={preview!r}",
        )


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------
# The firewall ships as a built-in (issue #5: it self-registers into the
# default registry on import). The Critic sandbox (ADR-0004, issue #22)
# needs to evaluate harness variants in isolation, so we also expose
# :func:`register_into` — pass any :class:`HookRegistry` to install the
# firewall into it without touching the process default. This keeps the
# global ``import harness.hooks`` behavior unchanged for the runner while
# letting the sandbox build a fresh, fully-loaded registry.


def register_into(registry: HookRegistry) -> InjectionFirewallHook:
    """Install a fresh :class:`InjectionFirewallHook` into ``registry``.

    Returns the registered hook so callers can introspect or detach it.
    Use this in the Critic sandbox: construct an empty ``HookRegistry``,
    call ``register_into(registry)`` for each built-in, then run the
    evaluation. Nothing leaks into the host process's default registry.
    """
    hook = InjectionFirewallHook()
    registry.register(hook)
    return hook


# ---------------------------------------------------------------------------
# Default-registry self-registration (issue #5)
# ---------------------------------------------------------------------------
# Unchanged behavior: ``import harness.hooks`` still wires the firewall into
# the process-default registry. The Critic sandbox opts out by constructing
# its own ``HookRegistry`` and calling :func:`register_into` instead.
register_hook(InjectionFirewallHook())
