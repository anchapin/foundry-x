from .base import Hook, HookRegistry, get_registry, register_hook
from .context_pruning import (
    ContextPruningHook,
    DEFAULT_THRESHOLD,
    Pruner,
    Tracer,
    register_into as context_pruning_register_into,
)
from .injection_firewall import InjectionFirewallHook, INJECTION_PATTERNS
from .rate_limit import (
    DEFAULT_MAX_DIFF_LINES,
    DEFAULT_MAX_PROPOSALS_PER_HOUR,
    DEFAULT_RATE_WINDOW_HOURS,
    RateLimitHook,
    get_default_max_diff_lines,
    get_default_max_proposals,
    get_default_rate_window_hours,
    register_into as rate_limit_register_into,
)

# Importing this package activates the prompt-injection firewall mandated by
# docs/SECURITY.md (the hook self-registers on import).
__all__ = [
    "Hook",
    "HookRegistry",
    "get_registry",
    "register_hook",
    "ContextPruningHook",
    "DEFAULT_THRESHOLD",
    "Pruner",
    "Tracer",
    "context_pruning_register_into",
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
