"""Unit tests for ``ServerPool`` (issue #1046, ADR-0026).

Covers pool registration, lifecycle (``start_all`` / ``stop_all``), and
health-aware ``acquire()`` routing — all without spawning real
``llama-server`` processes by injecting fake managers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_x.infra.server_manager import (
    FoundryServerManager,
    ServerConfig,
    ServerPool,
)


def _config(
    *,
    host: str = "http://127.0.0.1:8080",
    model_path: str | None = "/tmp/x.gguf",
    n_gpu_layers: str = "0",
    ctx_size: str = "8192",
    autostart: bool = True,
    server_bin: str | None = None,
    health_ready_timeout_s: float = 0.1,
    **overrides: Any,
) -> ServerConfig:
    base: dict[str, Any] = {
        "host": host,
        "model_path": model_path,
        "n_gpu_layers": n_gpu_layers,
        "ctx_size": ctx_size,
        "autostart": autostart,
        "server_bin": server_bin,
        "health_ready_timeout_s": health_ready_timeout_s,
    }
    base.update(overrides)
    return ServerConfig(**base)


def _fake_manager(healthy: bool = True) -> MagicMock:
    """A MagicMock that quacks like a FoundryServerManager."""
    mgr = MagicMock(spec=FoundryServerManager)
    mgr.is_healthy = AsyncMock(return_value=healthy)
    mgr.start = AsyncMock(return_value=MagicMock())
    mgr.stop = AsyncMock(return_value=None)
    mgr.config = MagicMock()
    mgr.config.autostart = True
    mgr.config.health_ready_timeout_s = 0.1
    mgr._is_healthy_sync = MagicMock(return_value=healthy)
    return mgr


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


def test_register_adds_slot() -> None:
    pool = ServerPool()
    cfg = _config()
    pool.register("q4-km", cfg)
    assert "q4-km" in pool.slots


def test_register_duplicate_raises_keyerror() -> None:
    pool = ServerPool()
    pool.register("q4-km", _config())
    with pytest.raises(KeyError, match="already registered"):
        pool.register("q4-km", _config())


def test_get_returns_manager_for_slot() -> None:
    pool = ServerPool()
    pool.register("q4-km", _config())
    mgr = pool.get("q4-km")
    assert isinstance(mgr, FoundryServerManager)


def test_get_unregistered_slot_raises_keyerror() -> None:
    pool = ServerPool()
    with pytest.raises(KeyError, match="not registered"):
        pool.get("nonexistent")


def test_slots_returns_insertion_order() -> None:
    pool = ServerPool()
    pool.register("a", _config())
    pool.register("b", _config())
    pool.register("c", _config())
    assert pool.slots == ["a", "b", "c"]


def test_empty_pool_slots() -> None:
    pool = ServerPool()
    assert pool.slots == []


# ---------------------------------------------------------------------------
# ServerConfig.slot
# ---------------------------------------------------------------------------


def test_server_config_slot_defaults_none() -> None:
    cfg = ServerConfig(
        host="http://127.0.0.1:8080",
        model_path="/tmp/x.gguf",
        n_gpu_layers="0",
        ctx_size="8192",
        autostart=True,
        server_bin=None,
    )
    assert cfg.slot is None


def test_server_config_from_env_accepts_slot() -> None:
    cfg = ServerConfig.from_env(slot="q4-km-fast")
    assert cfg.slot == "q4-km-fast"


def test_server_config_from_env_slot_defaults_none() -> None:
    cfg = ServerConfig.from_env({})
    assert cfg.slot is None


# ---------------------------------------------------------------------------
# start_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_all_starts_every_manager() -> None:
    m1 = _fake_manager()
    m2 = _fake_manager()
    pool = ServerPool(manager_factory=lambda cfg: m1 if cfg.slot == "a" else m2)
    pool.register("a", _config(slot="a"))
    pool.register("b", _config(slot="b"))

    results = await pool.start_all()

    assert results == {"a": True, "b": True}
    m1.start.assert_called_once()
    m2.start.assert_called_once()


@pytest.mark.asyncio
async def test_start_all_returns_false_on_failure() -> None:
    m_ok = _fake_manager()
    m_fail = _fake_manager()
    m_fail.start = MagicMock(side_effect=Exception("boom"))
    pool = ServerPool(manager_factory=lambda cfg: m_fail if cfg.slot == "bad" else m_ok)
    pool.register("good", _config(slot="good"))
    pool.register("bad", _config(slot="bad"))

    results = await pool.start_all()

    assert results == {"good": True, "bad": False}


@pytest.mark.asyncio
async def test_start_all_empty_pool() -> None:
    pool = ServerPool()
    assert await pool.start_all() == {}


# ---------------------------------------------------------------------------
# stop_all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_all_stops_every_manager() -> None:
    m1 = _fake_manager()
    m2 = _fake_manager()
    pool = ServerPool(manager_factory=lambda cfg: m1 if cfg.slot == "a" else m2)
    pool.register("a", _config(slot="a"))
    pool.register("b", _config(slot="b"))

    await pool.stop_all()

    m1.stop.assert_called_once()
    m2.stop.assert_called_once()


@pytest.mark.asyncio
async def test_stop_all_empty_pool() -> None:
    pool = ServerPool()
    await pool.stop_all()


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acquire_returns_first_healthy_slot() -> None:
    m_unhealthy = _fake_manager(healthy=False)
    m_healthy = _fake_manager(healthy=True)
    pool = ServerPool(manager_factory=lambda cfg: m_unhealthy if cfg.slot == "a" else m_healthy)
    pool.register("a", _config(slot="a"))
    pool.register("b", _config(slot="b"))

    slot, mgr = await pool.acquire()

    assert slot == "b"
    assert mgr is m_healthy


@pytest.mark.asyncio
async def test_acquire_fallback_when_none_healthy_no_autostart() -> None:
    m1 = _fake_manager(healthy=False)
    m1.config.autostart = False
    pool = ServerPool(manager_factory=lambda cfg: m1)
    pool.register("a", _config(slot="a"))

    slot, mgr = await pool.acquire()

    assert slot == "a"
    assert mgr is m1


@pytest.mark.asyncio
async def test_acquire_waits_for_autostart_slot() -> None:
    """When autostart is on, acquire polls until the slot becomes healthy."""
    m = _fake_manager(healthy=False)

    call_count = 0

    async def _flipping_health() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count > 2

    m.is_healthy = _flipping_health
    m.config.autostart = True
    m.config.health_ready_timeout_s = 10.0

    pool = ServerPool(manager_factory=lambda cfg: m)
    pool.register("a", _config(slot="a"))

    # Patch the poll interval to make the test fast.
    import foundry_x.infra.server_manager as sm

    original = sm._POOL_HEALTH_POLL_INTERVAL_S
    sm._POOL_HEALTH_POLL_INTERVAL_S = 0.01
    try:
        slot, mgr = await pool.acquire()
    finally:
        sm._POOL_HEALTH_POLL_INTERVAL_S = original

    assert slot == "a"
    assert mgr is m
    assert call_count > 2


@pytest.mark.asyncio
async def test_acquire_falls_back_after_timeout() -> None:
    """When the autostart window expires, the first slot is returned."""
    m = _fake_manager(healthy=False)
    m.config.autostart = True
    m.config.health_ready_timeout_s = 0.02

    pool = ServerPool(manager_factory=lambda cfg: m)
    pool.register("a", _config(slot="a", health_ready_timeout_s=0.02))

    import foundry_x.infra.server_manager as sm

    original = sm._POOL_HEALTH_POLL_INTERVAL_S
    sm._POOL_HEALTH_POLL_INTERVAL_S = 0.01
    try:
        slot, mgr = await pool.acquire()
    finally:
        sm._POOL_HEALTH_POLL_INTERVAL_S = original

    assert slot == "a"
    assert mgr is m


@pytest.mark.asyncio
async def test_acquire_empty_pool_raises() -> None:
    pool = ServerPool()
    with pytest.raises(KeyError, match="empty pool"):
        await pool.acquire()


@pytest.mark.asyncio
async def test_acquire_all_healthy_returns_first() -> None:
    m1 = _fake_manager(healthy=True)
    m2 = _fake_manager(healthy=True)
    pool = ServerPool(manager_factory=lambda cfg: m1 if cfg.slot == "a" else m2)
    pool.register("a", _config(slot="a"))
    pool.register("b", _config(slot="b"))

    slot, _ = await pool.acquire()
    assert slot == "a"


# ---------------------------------------------------------------------------
# slot_health (non-blocking snapshot)
# ---------------------------------------------------------------------------


def test_slot_health_returns_snapshot() -> None:
    m1 = _fake_manager(healthy=True)
    m2 = _fake_manager(healthy=False)
    pool = ServerPool(manager_factory=lambda cfg: m1 if cfg.slot == "a" else m2)
    pool.register("a", _config(slot="a"))
    pool.register("b", _config(slot="b"))

    health = pool.slot_health()

    assert health == {"a": True, "b": False}


def test_slot_health_empty_pool() -> None:
    pool = ServerPool()
    assert pool.slot_health() == {}


# ---------------------------------------------------------------------------
# manager_factory injection
# ---------------------------------------------------------------------------


def test_custom_manager_factory_used() -> None:
    sentinel = MagicMock(spec=FoundryServerManager)
    factory = MagicMock(return_value=sentinel)
    pool = ServerPool(manager_factory=factory)
    pool.register("x", _config(slot="x"))

    factory.assert_called_once()
    assert pool.get("x") is sentinel
