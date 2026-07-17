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

Issue #122 extends the original ASCII-only regex set (issue #5) with three
evasion classes — Unicode confusables (full-width and zero-width forms),
base64-encoded payloads, and non-English marker phrases. Each new pattern
links back to SECURITY.md:48 so a future Evolver knows why it exists.
"""

from __future__ import annotations

import base64
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

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
    # --- Issue #122: Unicode + base64 + non-English evasion coverage -------
    # Spanish-language equivalent of "ignore previous instructions"
    # (docs/SECURITY.md:48). The literal imperative form is the attack
    # phrasing most commonly seen in Spanish-language tool outputs.
    (
        "ignore_spanish",
        r"ignora\s+(?:las\s+)?instrucciones\s+anteriores",
    ),
    # Issue #584: additional non-Spanish language evasion coverage.
    # French: "ignorer les instructions" / "oublier les consignes"
    (
        "ignore_french",
        r"(?:ignorer\s+(?:les\s+)?instructions|oublier\s+(?:les\s+)?consignes)",
    ),
    # German: "ignoriere vorherige Anweisungen" / "ignoriere die Anweisungen"
    (
        "ignore_german",
        r"ignoriere\s+(?:vorherige\s+)?(?:die\s+)?Anweisungen",
    ),
    # Portuguese: "ignore as instruções anteriores"
    (
        "ignore_portuguese",
        r"ignore\s+(?:as\s+)?instruções\s+anteriores",
    ),
    # Italian: "ignora le istruzioni precedenti"
    (
        "ignore_italian",
        r"ignora\s+(?:le\s+)?istruzioni\s+precedenti",
    ),
    # JSON-escaped role-tag form, e.g. {\"role\":\"system\",\"content\":...}.
    # The legitimate ``role_tag_colon`` regex above requires a literal
    # ``system:`` preceded by start-of-line; an attacker wrapping the role
    # in JSON quotes and escaping the wrapping quotes (``\"role\":\"system\"``)
    # sidesteps that. This pattern catches the escaped form directly so we
    # do not need an LLM call to JSON-decode every tool result.
    (
        "role_tag_json_escaped",
        r'\\"role\\":\\"(?:system|assistant|developer|user)',
    ),
    # Presence of zero-width / invisible format characters is a textbook
    # bypass for case-insensitive ASCII regex sets. NFKC normalization
    # preserves these characters (they are not combining marks), so the
    # firewall flags them as an explicit signal in addition to collapsing
    # them before the ASCII patterns run (see ``_ZERO_WIDTH_RE`` below).
    # The chars are scanned on the raw (pre-collapse) text in
    # ``scan_for_injection``; that is why this category is exempted from
    # the cleaned-text pass.
    (
        "unicode_confusable",
        r"[\u200B-\u200F\u2028-\u202F\u2060-\u2064\uFEFF]",
    ),
    # Candidate matcher for base64-encoded payloads. The candidate alone
    # is not an attack (URLs, hashes, and tokens all match the shape), so
    # the actual ``base64_payload`` match is only emitted when the decoded
    # content matches one of the ASCII markers above. See
    # ``_scan_base64_payloads``.
    (
        "base64_payload",
        r"[A-Za-z0-9+/]{16,}={0,2}",
    ),
)

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pat, re.IGNORECASE | re.MULTILINE)) for name, pat in INJECTION_PATTERNS
)
_PATTERN_SOURCE: dict[str, str] = {name: src for name, src in INJECTION_PATTERNS}
_COMPILED_BY_NAME: dict[str, re.Pattern[str]] = {name: pat for name, pat in _COMPILED}

# Zero-width / invisible format characters. NFKC preserves them (they are
# not combining marks), so the firewall strips them explicitly before the
# ASCII-marker pass: ``i\u200dg\u200dn\u200do\u200dr\u200de`` then collapses
# to ``ignore`` and the existing ``ignore_previous`` regex matches.
#
#   U+200B-U+200F  ZWSP, ZWNJ, ZWJ, LRM, RLM, embedding / pop controls
#   U+2028-U+202F  line / paragraph separators + bidi / NNBSP controls
#   U+2060-U+2064  word joiner + invisible math operators
#   U+FEFF         byte-order mark (also ZWNBSP)
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u2028-\u202F\u2060-\u2064\uFEFF]")

# Cap on how much of the original (now-suppressed) output we keep in the
# ``error`` field purely for human-review triage. The raw span is never
# placed back into ``output`` (which gets re-injected into the prompt).
_PREVIEW_LEN = 120

# Cap on the base64 candidate length we will try to decode. 4 KB of decoded
# text is well past anything a real tool emits by accident.
_BASE64_MAX_LEN = 4096


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
    """Best-effort coercion of ``ToolResult.output`` to a scannable string.

    Strings are NFKC-normalized here so full-width forms (e.g. ``Ｉｇｎｏｒｅ``)
    and combining-character sequences (e.g. ``e\u0301``) collapse to their
    ASCII canonical equivalents before the regex pass runs. Zero-width
    characters (ZWJ / ZWNJ / ZWSP / BOM) are *not* NFKC-collapsible — they
    are format characters, not combining marks — so the firewall strips them
    in ``scan_for_injection`` instead, which keeps this function cheap and
    preserves the original span for the ``unicode_confusable`` signal that
    intentionally scans the pre-strip text.

    The normalized text is only used for scanning and for the bounded
    human-review preview in ``error``; the firewall never places the
    normalized form back into the prompt (``post_tool`` truncates on
    detection, and on clean results the original output is returned
    unchanged).
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return unicodedata.normalize("NFKC", output)
    try:
        return repr(output)
    except Exception:  # pragma: no cover - defensive, never swallow silently
        # Re-raise as RuntimeError so the trace pipeline surfaces it; bare
        # ``except: pass`` is forbidden by AGENTS.md §2.
        raise RuntimeError("injection_firewall: failed to coerce tool output") from None


def _scan_base64_payloads(text: str) -> list[InjectionMatch]:
    """Decode each base64 candidate in ``text`` and rescan for ASCII markers.

    A candidate alone (a 16+ char run of base64 chars) is not an attack —
    legitimate URLs, hashes, and tokens match the same shape. We only emit
    a ``base64_payload`` match when the decoded bytes (UTF-8) match one of
    the ASCII marker regexes from :data:`INJECTION_PATTERNS`. We
    deliberately do not recurse into nested base64: the work is bounded
    and legitimate base64 almost never survives two decodings to land on
    a known phrase.
    """
    matches: list[InjectionMatch] = []
    candidate_pat = _COMPILED_BY_NAME["base64_payload"]
    for m in candidate_pat.finditer(text):
        candidate = m.group(0)
        if len(candidate) > _BASE64_MAX_LEN:
            continue
        try:
            decoded_bytes = base64.b64decode(candidate, validate=True)
        except Exception:
            continue
        try:
            decoded = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for name, pat in _COMPILED:
            if name == "base64_payload":
                continue  # no recursion into nested base64
            if pat.search(decoded):
                matches.append(
                    InjectionMatch(
                        name="base64_payload",
                        pattern=f"base64-of-{name}",
                        snippet=decoded.strip()[:80],
                    )
                )
                break
    return matches


def scan_for_injection(text: str) -> ScanResult:
    """Scan ``text`` for the markers in :data:`INJECTION_PATTERNS`.

    The pipeline is:

    1. Strip zero-width / format characters from a *copy* of ``text``. The
       original span is kept around for the ``unicode_confusable`` signal,
       which intentionally fires on the pre-strip text.
    2. Run every pattern in :data:`INJECTION_PATTERNS` against the cleaned
       text. The ``base64_payload`` candidate matcher is exempted — its
       only contribution is via ``_scan_base64_payloads`` below, so raw
       candidates that decode to junk do not generate false positives.
    3. Run the ``unicode_confusable`` pattern against the *original* text
       so the presence of zero-width characters is observable as a
       distinct signal even after they have been stripped for the ASCII
       pass.
    4. Decode every base64 candidate in the cleaned text and rescan for
       the ASCII markers. Emit a ``base64_payload`` match only when the
       decoded content actually contains one.

    Returns every distinct match (a marker may fire more than once). This
    is a pure function so the Evolver and the test suite can exercise it
    independently of the hook machinery.
    """
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    matches: list[InjectionMatch] = []
    for name, pattern in _COMPILED:
        if name == "base64_payload":
            # Candidate matchers do not, on their own, count as detection.
            # The real base64_payload matches are produced by
            # ``_scan_base64_payloads`` after a successful decode + rescan.
            continue
        target = text if name == "unicode_confusable" else cleaned
        for m in pattern.finditer(target):
            matches.append(
                InjectionMatch(
                    name=name,
                    pattern=_PATTERN_SOURCE[name],
                    snippet=m.group(0).strip()[:80],
                )
            )
    matches.extend(_scan_base64_payloads(cleaned))
    return ScanResult(blocked=bool(matches), matches=tuple(matches))


class InjectionFirewallHook:
    _phase: int = 1

    """Hook that truncates/flags tool results containing injection markers.

    Implements the :class:`~harness.hooks.base.Hook` Protocol. ``pre_tool``
    is an identity pass-through; ``post_tool`` is where the screening happens
    — it sees ``ToolResult.output`` before the caller re-injects it into a
    prompt (the interception point mandated by SECURITY.md).

    Parameters
    ----------
    policy:
        ``"block"`` (default) — replace the tool output with a suppression
        marker and flag the result for human review (legacy behavior).

        ``"warn"`` — log the detection and set ``error`` to an
        ``injection_warn:...`` marker so callers (notably tests) can assert
        that detection happened, but return the original output unchanged.
        The runner is never aborted in either mode — the firewall never
        raises — but ``"warn"`` makes the side effect observable without
        truncating output that may still be legitimate.
    tracer:
        Optional sink invoked on every block-policy detection. Receives the
        structured payload dict the caller should persist under the
        ``injection_blocked`` trace-event kind (issue #120). The payload
        shape is owned by the firewall — the hook knows the marker names,
        the tool name, and the bounded preview — and the caller's only job
        is to forward it to its trace store::

            payload = {
                "markers": [...],   # list[str], sorted unique marker names
                "tool":    "...",   # str, the originating tool name
                "preview": "...",   # str, first _PREVIEW_LEN chars of the
                                    # suppressed text with newlines folded
                                    # to spaces (never re-injected into a
                                    # prompt, safe to persist)
            }

        The firewall swallows any exception raised by ``tracer`` (logs and
        continues) so a misbehaving sink never aborts the agent run; per
        AGENTS.md §2 the failure is surfaced via the project logger rather
        than silently dropped. When ``tracer`` is ``None`` (the default,
        e.g. for the self-registered hook in ``harness/hooks/__init__``)
        blocks are still written to the module logger exactly as before —
        the legacy audit surface is preserved.
    """

    def __init__(
        self,
        policy: Literal["block", "warn"] = "block",
        tracer: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if policy not in ("block", "warn"):
            raise ValueError(f"injection_firewall: unknown policy {policy!r}")
        self._policy = policy
        self._tracer = tracer

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        try:
            text = _coerce_text(result.output)
            scan = scan_for_injection(text)
        except Exception as exc:
            _log.exception(
                "injection_firewall: scan of tool %r failed; treating result as "
                "potentially unsafe (fail-closed): %r",
                result.name,
                exc,
            )
            safe_output = (
                "[injection_firewall] output suppressed: injection scan failed. "
                "Treating as potentially unsafe."
            )
            return ToolResult(
                name=result.name,
                output=safe_output,
                error="injection_scan_error:coercion_or_scan_failed",
            )

        if not scan.blocked:
            return result

        names = ",".join(sorted({m.name for m in scan.matches}))
        _log.warning(
            "injection_firewall: %s %d marker(s) [%s] from tool %r",
            self._policy,
            len(scan.matches),
            names,
            result.name,
        )
        if self._policy == "warn":
            return ToolResult(
                name=result.name,
                output=result.output,
                error=f"injection_warn:{names}",
            )

        # ``output`` is re-injected into the prompt, so it must NOT carry the
        # adversarial span. ``error`` is the human-review flag channel and
        # keeps only a bounded preview for triage (never re-injected).
        preview = text[:_PREVIEW_LEN].replace("\n", " ")
        safe_output = (
            "[injection_firewall] output suppressed: injection markers "
            f"detected ({names}). Flagged for human review."
        )
        if self._tracer is not None:
            self._emit_block_event(result.name, scan.matches, preview)
        return ToolResult(
            name=result.name,
            output=safe_output,
            error=f"injection_detected:{names} preview={preview!r}",
        )

    def _emit_block_event(
        self,
        tool_name: str,
        matches: tuple[InjectionMatch, ...],
        preview: str,
    ) -> None:
        """Forward one block decision to the injected ``tracer`` sink.

        Emits two events on every block (issue #823):

        1. ``injection_blocked`` (issue #120) — the canonical block payload
           with ``markers``, ``tool``, ``preview``.
        2. ``firewall_exception`` (issue #823) — structured warning with
           ``hook_name``, ``pattern_matched``, and ``risk_score`` so the
           trace store carries a queryable firewall-audit log independent of
           the ``injection_blocked`` aggregation path.

        Kept tiny and synchronous so the hook contract stays an ``async``
        boundary at the hook level. A misbehaving tracer is logged at
        exception level and swallowed — the firewall never aborts the agent
        run on a sink failure (the same isolation contract the hook registry
        applies to its own hook exceptions; see
        ``harness/hooks/base.py::_isolate_failure``).
        """
        assert self._tracer is not None

        # Issue #120: injection_blocked event (unchanged contract).
        injection_payload: dict[str, Any] = {
            "markers": sorted({m.name for m in matches}),
            "tool": tool_name,
            "preview": preview,
        }

        # Issue #823: firewall_exception event.
        names = ",".join(sorted({m.name for m in matches}))
        risk_score = self._compute_risk_score(matches)
        firewall_payload: dict[str, Any] = {
            "hook_name": "InjectionFirewallHook",
            "pattern_matched": names,
            "risk_score": risk_score,
        }

        try:
            self._tracer("injection_blocked", injection_payload)
        except Exception:
            _log.exception(
                "injection_firewall: tracer raised injection_payload on block of tool %r; isolating and continuing",
                tool_name,
            )
        try:
            self._tracer("firewall_exception", firewall_payload)
        except Exception:
            _log.exception(
                "injection_firewall: tracer raised firewall_payload on block of tool %r; isolating and continuing",
                tool_name,
            )

    def _compute_risk_score(self, matches: tuple[InjectionMatch, ...]) -> int:
        """Compute a risk score from the detected injection markers.

        Higher scores indicate greater severity. The weighting is based on
        the attack class: role-tag injection (which can hijack the prompt
        context) scores higher than bare instruction-override phrases.
        """
        if not matches:
            return 0
        score = 0
        for match in matches:
            if match.name in ("role_tag_colon", "chatml_tag", "role_tag_json_escaped"):
                score += 2
            elif match.name in ("unicode_confusable", "base64_payload"):
                score += 2
            else:
                score += 1
        return score


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


def register_into(
    registry: HookRegistry,
    *,
    tracer: Callable[[str, dict[str, Any]], None] | None = None,
) -> InjectionFirewallHook:
    """Install a fresh :class:`InjectionFirewallHook` into ``registry``.

    Returns the registered hook so callers can introspect or detach it.
    Use this in the Critic sandbox: construct an empty ``HookRegistry``,
    call ``register_into(registry)`` for each built-in, then run the
    evaluation. Nothing leaks into the host process's default registry.

    Passing ``tracer=`` wires the new hook's block path to the supplied
    sink (typically a closure over ``TraceLogger.record``) so the runner
    or sandbox emits the ``injection_blocked`` and ``firewall_exception``
    trace events (issue #120, issue #823) without coupling the harness
    package to ``foundry_x``.
    """
    hook = InjectionFirewallHook(tracer=tracer)
    registry.register(hook)
    return hook


# ---------------------------------------------------------------------------
# Default-registry self-registration (issue #5)
# ---------------------------------------------------------------------------
# Unchanged behavior: ``import harness.hooks`` still wires the firewall into
# the process-default registry. The Critic sandbox opts out by constructing
# its own ``HookRegistry`` and calling :func:`register_into` instead.
register_hook(InjectionFirewallHook())
