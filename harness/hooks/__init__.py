from .base import Hook, HookRegistry, get_registry, register_hook
from .context_pruning import (
    ContextPruningHook,
    DEFAULT_THRESHOLD,
    Pruner,
    Tracer,
    register_into,
)
from .injection_firewall import InjectionFirewallHook, INJECTION_PATTERNS

# Importing this package activates the prompt-injection firewall mandated by
# docs/SECURITY.md (the hook self-registers on import). The context_pruning
# hook is importable here but does NOT self-register: it needs a session_id
# and TraceLogger-backed closures that only the runner can supply (issue
# #106). The runner calls register_into(registry, ...) to install it.
__all__ = [
    "Hook",
    "HookRegistry",
    "get_registry",
    "register_hook",
    "ContextPruningHook",
    "DEFAULT_THRESHOLD",
    "Pruner",
    "Tracer",
    "register_into",
    "InjectionFirewallHook",
    "INJECTION_PATTERNS",
]
