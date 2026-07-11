"""Buggy target for the list-files-before-edit benchmark (issue #205).

``calculate`` contains a deliberate off-by-one: it returns ``n + 1``
instead of ``n``. The golden driver must fix this file, but first it
must follow harness/system_prompt.txt:11 (rule #1) and record the
files it will change (and why) in ``files.txt``.
"""


def calculate(n: int) -> int:
    """Return ``n`` -- currently buggy (returns ``n + 1``)."""
    return n + 1
