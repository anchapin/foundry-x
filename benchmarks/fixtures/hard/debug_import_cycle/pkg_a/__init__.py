"""Package A for the debug_import_cycle hard-tier fixture (ADR-0028 H1).

Re-exports ``DataModel`` so callers can do ``from pkg_a import DataModel``.
"""
from pkg_a.models import DataModel

__all__ = ["DataModel"]
