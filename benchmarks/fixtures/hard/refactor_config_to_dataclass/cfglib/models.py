"""Models module using typed Config (already migrated — reference pattern).

This module demonstrates the CORRECT usage of the new ``Config`` dataclass:
attribute access (``cfg.timeout``) rather than dict-style access
(``cfg["timeout"]``).  The agent can use this file as a reference for the
expected calling convention.
"""

from cfglib.config import Config, get_config


def describe_config() -> str:
    """Return a human-readable config summary."""
    cfg: Config = get_config()
    return f"timeout={cfg.timeout}s retries={cfg.retries} limit={cfg.connection_limit}"
