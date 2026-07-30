"""Package B for the debug_import_cycle hard-tier fixture (ADR-0028 H1).

Re-exports ``serialize`` so callers can do ``from pkg_b import serialize``.
"""
from pkg_b.serializers import serialize

__all__ = ["serialize"]
