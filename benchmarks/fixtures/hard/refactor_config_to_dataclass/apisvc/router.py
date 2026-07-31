"""API router for refactor_config_to_dataclass (seeded/stale caller).

Uses dict-style ``cfg["max_connections"]`` access AND the OLD field name.
``get_config()`` now returns a ``Config`` dataclass where the field was
renamed to ``connection_limit``.  This raises ``TypeError`` in the seeded
state; after migrating to attribute access it raises ``AttributeError``
because ``max_connections`` no longer exists on ``Config``.
"""

from cfglib.config import get_config


def max_connections() -> int:
    """Return the connection limit — BROKEN: dict access + stale field name."""
    cfg = get_config()
    return cfg["max_connections"]  # TypeError: not subscriptable
