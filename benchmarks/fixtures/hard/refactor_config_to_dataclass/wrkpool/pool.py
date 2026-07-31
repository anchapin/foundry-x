"""Worker pool for refactor_config_to_dataclass (seeded/stale caller).

Calls ``get_config()`` and accesses the result with dict-style syntax.
Since ``get_config()`` now returns a ``Config`` dataclass, this raises
``TypeError`` at call time.
"""

from cfglib.config import get_config


def pool_settings() -> tuple[int, int]:
    """Return (timeout, batch_size) from config — BROKEN: dict access on dataclass."""
    cfg = get_config()
    timeout = cfg["timeout"]        # TypeError: 'Config' is not subscriptable
    batch_size = cfg["batch_size"]  # TypeError
    return timeout, batch_size
