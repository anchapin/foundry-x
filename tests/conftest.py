"""Shared pytest fixtures for the test suite.

``model_adapter`` fixture
-------------------------
Provides a ``ModelAdapter`` instance switched by the ``TEST_MODEL_MODE``
environment variable:

- ``TEST_MODEL_MODE=mock`` (default): ``MockModelAdapter`` — deterministic,
  no network, suitable for offline CI and local development.
- ``TEST_MODEL_MODE=real``: ``OpenAICompatibleAdapter`` backed by
  ``OPENCODE_SERVER_URL`` / ``LLAMACPP_HOST``. Suitable for integration
  testing against a live endpoint.

Example usage::

    async def test_my_feature(model_adapter):
        await run_task("do the task", harness_dir, logger, session_id, model_adapter=model_adapter)

For tests that need to configure the mock before use::

    async def test_with_tools(model_adapter: MockModelAdapter):
        model_adapter.set_response(MockModelAdapterConfig(
            response_content="here is the result",
            tool_calls=[...],
        ))
        await run_task("do the task", harness_dir, logger, session_id, model_adapter=model_adapter)

Note: the fixture is session-scoped for the mock (stateless and cheap)
and session-scoped for the real adapter (maintains a connection pool).
"""

from __future__ import annotations

import os

import pytest

from tests._model_fixtures import (
    MockModelAdapter,
    create_model_adapter,
)


@pytest.fixture(scope="session")
def model_adapter():
    """Return a ``ModelAdapter`` selected by ``TEST_MODEL_MODE``."""
    mode = os.environ.get("TEST_MODEL_MODE", "mock").strip().lower()
    adapter = create_model_adapter(mode)
    yield adapter
    if hasattr(adapter, "aclose") and callable(adapter.aclose):
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(adapter.aclose())
        else:
            loop.run_until_complete(adapter.aclose())


@pytest.fixture
def mock_adapter() -> MockModelAdapter:
    """Return a fresh ``MockModelAdapter`` (always mock, ignores TEST_MODEL_MODE).

    Use this when you need to configure per-test responses without sharing
    state across tests.
    """
    return MockModelAdapter()
