"""Integration test exercising both packages for debug_import_cycle (ADR-0028 H1).

Importing this module triggers the circular import at collection time in
the seeded (broken) state, so ``python -m pytest`` exits non-zero with an
``ImportError`` before any test body runs.  After the deferred-import fix,
collection succeeds and ``test_round_trip`` passes.
"""

from pkg_a.models import DataModel
from pkg_b.serializers import serialize


def test_round_trip() -> None:
    """Serialising a DataModel round-trips the value field."""
    model = DataModel(42)
    assert serialize(model) == {"value": 42}


def test_to_serialized_helper() -> None:
    """The DataModel.to_serialized helper delegates to pkg_b.serialize."""
    assert DataModel(7).to_serialized() == {"value": 7}
