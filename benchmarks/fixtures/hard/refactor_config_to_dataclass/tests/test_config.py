"""Config integration tests for refactor_config_to_dataclass (seeded/half-stale).

``test_config_usage`` exercises the API client, router, and worker pool,
AND contains a direct dict-style ``cfg["timeout"]`` access that must be
migrated to ``cfg.timeout``.  The constraint is that ``test_config_usage``
must STILL EXIST and pass after the refactor.

``test_config_defaults`` and ``test_config_types`` already use the correct
attribute-access pattern and serve as reference for the expected contract.
"""

from cfglib.config import get_config
from apisvc.client import fetch_with_timeout
from apisvc.router import max_connections
from wrkpool.pool import pool_settings


def test_config_usage() -> None:
    """Integration test exercising all config consumers."""
    timeout, retries = fetch_with_timeout()
    assert timeout == 30
    assert retries == 3

    assert max_connections() == 10

    pool_timeout, batch_size = pool_settings()
    assert pool_timeout == 30
    assert batch_size == 100

    # Direct stale call — must be migrated to attribute access
    cfg = get_config()
    assert cfg["timeout"] == 30  # stale dict access


def test_config_defaults() -> None:
    """Config dataclass has the expected default values."""
    cfg = get_config()
    assert cfg.timeout == 30
    assert cfg.retries == 3
    assert cfg.connection_limit == 10
    assert cfg.batch_size == 100


def test_config_types() -> None:
    """Config fields have the expected types."""
    cfg = get_config()
    assert isinstance(cfg.timeout, int)
    assert isinstance(cfg.connection_limit, int)
