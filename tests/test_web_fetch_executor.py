"""Tests for the ``web_fetch`` skill executor (issue #1054).

Exercises ``_exec_web_fetch`` against:
- empty URL (error)
- non-HTTP(S) scheme rejection (security guard)
- successful fetch via mocked httpx (content, truncation, status code)
- timeout handling
- connection error handling

The ``WebFetchHook`` allowlist enforcement is tested separately in
``tests/harness/test_web_fetch_hook.py``.
"""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from foundry_x.execution.runner import _exec_web_fetch


class TestExecWebFetchEmptyUrl:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self) -> None:
        result = await _exec_web_fetch({"url": ""})
        assert result["error"] == "url is required"
        assert result["content"] == ""
        assert result["status_code"] == 0


class TestExecWebFetchSchemeGuard:
    @pytest.mark.asyncio
    async def test_file_scheme_rejected(self) -> None:
        result = await _exec_web_fetch({"url": "file:///etc/passwd"})
        assert "not allowed" in result["error"]
        assert "file" in result["error"]

    @pytest.mark.asyncio
    async def test_ftp_scheme_rejected(self) -> None:
        result = await _exec_web_fetch({"url": "ftp://example.com/file"})
        assert "not allowed" in result["error"]
        assert "ftp" in result["error"]


class TestExecWebFetchSuccess:
    @pytest.mark.asyncio
    async def test_successful_fetch(self) -> None:
        body = b"Python documentation\n" * 10
        mock_response = mock.MagicMock()
        mock_response.content = body
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.url = httpx.URL("https://docs.python.org/3/")

        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=None)
        mock_client.get = mock.AsyncMock(return_value=mock_response)

        with mock.patch("httpx.AsyncClient", return_value=mock_client):
            result = await _exec_web_fetch({"url": "https://docs.python.org/3/"})

        assert result["error"] is None
        assert result["status_code"] == 200
        assert result["content"].startswith("Python documentation")
        assert result["content_type"] == "text/html; charset=utf-8"
        assert result["truncated"] is False
        assert result["bytes_returned"] > 0

    @pytest.mark.asyncio
    async def test_truncation(self) -> None:
        body = b"A" * 50000
        mock_response = mock.MagicMock()
        mock_response.content = body
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.url = httpx.URL("https://docs.python.org/big")

        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=None)
        mock_client.get = mock.AsyncMock(return_value=mock_response)

        with mock.patch("httpx.AsyncClient", return_value=mock_client):
            result = await _exec_web_fetch(
                {"url": "https://docs.python.org/big", "max_bytes": 1024}
            )

        assert result["truncated"] is True
        assert result["bytes_returned"] <= 1024

    @pytest.mark.asyncio
    async def test_final_url_after_redirect(self) -> None:
        mock_response = mock.MagicMock()
        mock_response.content = b"redirected"
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.url = httpx.URL("https://docs.python.org/3/library/")

        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=None)
        mock_client.get = mock.AsyncMock(return_value=mock_response)

        with mock.patch("httpx.AsyncClient", return_value=mock_client):
            result = await _exec_web_fetch({"url": "https://docs.python.org/3"})

        assert result["url"] == "https://docs.python.org/3/library/"


class TestExecWebFetchErrors:
    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=None)
        mock_client.get = mock.AsyncMock(side_effect=httpx.ReadTimeout("read timed out"))

        with mock.patch("httpx.AsyncClient", return_value=mock_client):
            result = await _exec_web_fetch(
                {"url": "https://slow.example.com/", "timeout_seconds": 5}
            )

        assert "timeout" in result["error"].lower()
        assert result["status_code"] == 0
        assert result["content"] == ""

    @pytest.mark.asyncio
    async def test_connection_error(self) -> None:
        mock_client = mock.AsyncMock()
        mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mock.AsyncMock(return_value=None)
        mock_client.get = mock.AsyncMock(side_effect=httpx.ConnectError("connection refused"))

        with mock.patch("httpx.AsyncClient", return_value=mock_client):
            result = await _exec_web_fetch({"url": "https://nope.example.com/"})

        assert "ConnectError" in result["error"]
        assert result["status_code"] == 0
