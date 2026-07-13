"""Pytest entry point for the grep_search_fix benchmark.

Collection imports ``services``, which imports the stale symbol from
``models``; the resulting ImportError fails collection until
``services.py`` is fixed.
"""

from services import describe_user


def test_describe_user() -> None:
    """describe_user(42) returns the formatted stub record."""
    assert describe_user(42) == "user-42 (id=42)"
