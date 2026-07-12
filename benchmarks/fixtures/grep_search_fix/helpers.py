"""Misc helpers for the grep_search_fix benchmark.

Already migrated to the new ``fetch_user`` name. This file is the
decoy that forces the agent's grep to discriminate which file still
holds the stale reference.
"""

from models import fetch_user


def recent_users() -> list[dict]:
    """Return a couple of stub users via the renamed helper."""
    return [fetch_user(1), fetch_user(2)]
