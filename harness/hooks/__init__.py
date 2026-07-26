from . import token_aware_pruning
from .base import Hook, HookRegistry, get_registry, register_hook
from .context_pruning import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOKEN_THRESHOLD,
    ContextPruningHook,
    Pruner,
    TokenAwarePruningHook,
    TokenCounter,
    Tracer,
    register_into,
    register_token_aware_into,
    resolve_context_tokens_threshold,
)
from .injection_firewall import INJECTION_PATTERNS, InjectionFirewallHook
from .rate_limit import (
    DEFAULT_MAX_DIFF_LINES,
    DEFAULT_MAX_PROPOSALS_PER_HOUR,
    DEFAULT_RATE_WINDOW_HOURS,
    RateLimitHook,
    get_default_max_diff_lines,
    get_default_max_proposals,
    get_default_rate_window_hours,
)
from .rate_limit import (
    register_into as rate_limit_register_into,
)

# Importing this package activates the prompt-injection firewall mandated by
# docs/SECURITY.md (the hook self-registers on import). The context_pruning
# hook is importable here but does NOT self-register: it needs a session_id
# and TraceLogger-backed closures that only the runner can supply (issue
# #106). The runner calls register_into(registry, ...) and
# register_token_aware_into(registry, ...) to install it.
__all__ = [
    "DEFAULT_MAX_DIFF_LINES",
    "DEFAULT_MAX_PROPOSALS_PER_HOUR",
    "DEFAULT_RATE_WINDOW_HOURS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOKEN_THRESHOLD",
    "INJECTION_PATTERNS",
    "ContextPruningHook",
    "Hook",
    "HookRegistry",
    "InjectionFirewallHook",
    "Pruner",
    "RateLimitHook",
    "TokenAwarePruningHook",
    "TokenCounter",
    "Tracer",
    "get_default_max_diff_lines",
    "get_default_max_proposals",
    "get_default_rate_window_hours",
    "get_registry",
    "rate_limit_register_into",
    "register_hook",
    "register_into",
    "register_token_aware_into",
    "resolve_context_tokens_threshold",
    "token_aware_pruning",
]
