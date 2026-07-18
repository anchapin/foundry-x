from .base import Hook, HookRegistry, get_registry, register_hook
from .context_pruning import (
    ContextPruningHook,
    DEFAULT_THRESHOLD,
    DEFAULT_TOKEN_THRESHOLD,
    Pruner,
    Tracer,
    TokenCounter,
    TokenAwarePruningHook,
    register_into,
    register_token_aware_into,
    resolve_context_tokens_threshold,
)
from . import token_aware_pruning
from .injection_firewall import InjectionFirewallHook, INJECTION_PATTERNS
from .rate_limit import (
    RateLimitHook,
    DEFAULT_MAX_DIFF_LINES,
    DEFAULT_MAX_PROPOSALS_PER_HOUR,
    DEFAULT_RATE_WINDOW_HOURS,
    get_default_max_diff_lines,
    get_default_max_proposals,
    get_default_rate_window_hours,
    register_into as rate_limit_register_into,
)

# Importing this package activates the prompt-injection firewall mandated by
# docs/SECURITY.md (the hook self-registers on import). The context_pruning
# hook is importable here but does NOT self-register: it needs a session_id
# and TraceLogger-backed closures that only the runner can supply (issue
# #106). The runner calls register_into(registry, ...) and
# register_token_aware_into(registry, ...) to install it.
__all__ = [
    "Hook",
    "HookRegistry",
    "get_registry",
    "register_hook",
    "ContextPruningHook",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOKEN_THRESHOLD",
    "Pruner",
    "Tracer",
    "TokenCounter",
    "TokenAwarePruningHook",
    "register_into",
    "register_token_aware_into",
    "resolve_context_tokens_threshold",
    "token_aware_pruning",
    "InjectionFirewallHook",
    "INJECTION_PATTERNS",
    "RateLimitHook",
    "DEFAULT_MAX_DIFF_LINES",
    "DEFAULT_MAX_PROPOSALS_PER_HOUR",
    "DEFAULT_RATE_WINDOW_HOURS",
    "get_default_max_diff_lines",
    "get_default_max_proposals",
    "get_default_rate_window_hours",
    "rate_limit_register_into",
]
