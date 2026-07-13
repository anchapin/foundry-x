from .base import Hook, HookRegistry, get_registry, register_hook
from .injection_firewall import InjectionFirewallHook, INJECTION_PATTERNS
from .rate_limit import RateLimitHook, register_into

# Importing this package activates the prompt-injection firewall mandated by
# docs/SECURITY.md (the hook self-registers on import).
# RateLimitHook also self-registers on import.
__all__ = [
    "Hook",
    "HookRegistry",
    "get_registry",
    "register_hook",
    "InjectionFirewallHook",
    "INJECTION_PATTERNS",
    "RateLimitHook",
    "register_into",
]
