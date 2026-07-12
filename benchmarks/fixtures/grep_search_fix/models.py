"""User model helpers for the grep_search_fix benchmark.

The lookup function has been renamed to ``fetch_user``. Callers that
still reference the previous name must be updated.
"""


def fetch_user(user_id: int) -> dict:
    """Return a stub user record for *user_id*."""
    return {"id": user_id, "name": f"user-{user_id}"}
