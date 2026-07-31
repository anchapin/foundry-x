"""API client for refactor_config_to_dataclass (seeded/stale caller).

Calls ``get_config()`` and accesses the result with dict-style
``cfg["key"]`` syntax.  Since ``get_config()`` now returns a ``Config``
dataclass (not a dict), this raises ``TypeError`` at call time.
"""

from cfglib.config import get_config


def fetch_with_timeout() -> tuple[int, int]:
    """Return (timeout, retries) from config — BROKEN: dict access on dataclass."""
    cfg = get_config()
    timeout = cfg["timeout"]     # TypeError: 'Config' is not subscriptable
    retries = cfg["retries"]     # TypeError
    return timeout, retries
