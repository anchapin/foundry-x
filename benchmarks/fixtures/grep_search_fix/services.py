"""Service layer for the grep_search_fix benchmark.

STALE REFERENCE: this module still imports the OLD name after models.py
renamed the lookup function to ``fetch_user``. Importing this module
raises ImportError until the reference is corrected.
"""

from models import get_user


def describe_user(user_id: int) -> str:
    """Return a human-readable description of *user_id*."""
    record = get_user(user_id)
    return f"{record['name']} (id={record['id']})"
