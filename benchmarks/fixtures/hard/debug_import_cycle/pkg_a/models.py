"""DataModel for the debug_import_cycle hard-tier fixture (seeded/broken).

This module imports ``serialize`` from ``pkg_b.serializers`` at module top
level.  ``pkg_b.serializers`` in turn imports ``DataModel`` from this module
at module top level, forming a circular import: importing either package
first leaves the other partially initialised and raises ``ImportError``
during test collection.

The agent must trace the ``ImportError`` back to the cycle, then break it
by deferring one side of the import to function scope (the classic
deferred-import pattern).  See ``benchmarks/tasks/test_debug_import_cycle.py``.
"""

from pkg_b.serializers import serialize


class DataModel:
    """A simple value object carrying a single integer."""

    def __init__(self, value: int) -> None:
        self.value = value

    def to_serialized(self) -> dict[str, int]:
        """Serialise this model via the ``pkg_b`` serializer."""
        return serialize(self)
