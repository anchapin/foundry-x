"""Configuration library for refactor_config_to_dataclass (seeded/migrated).

The public ``get_config()`` function has been migrated from returning a
plain ``dict`` to returning a typed ``Config`` dataclass.  Callers that
still use dict-style ``cfg["key"]`` access will raise ``TypeError`` at
call time.

Note: the field formerly known as ``max_connections`` has been renamed to
``connection_limit`` as part of this migration.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Typed application configuration."""

    timeout: int = 30
    retries: int = 3
    connection_limit: int = 10
    batch_size: int = 100


def get_config() -> Config:
    """Return the application configuration."""
    return Config()
