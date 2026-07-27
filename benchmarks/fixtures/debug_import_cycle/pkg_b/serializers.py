"""Serializer for the debug_import_cycle hard-tier fixture (seeded/broken).

This module imports ``DataModel`` from ``pkg_a.models`` at module top
level, while ``pkg_a.models`` imports ``serialize`` from this module at
module top level.  The mutual top-level import is the circular-import
root cause: whichever package Python imports first, the other is only
partially initialised when the back-reference executes, so the import
fails with ``ImportError: cannot import name '...' from partially
initialised module``.

The golden fix moves this import inside ``serialize()`` (deferred import)
so that no import-time cycle exists.
"""

from pkg_a.models import DataModel


def serialize(model: "DataModel") -> dict[str, int]:
    """Return a dict representation of *model*."""
    return {"value": model.value}
