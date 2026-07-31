"""Re-export ``benchmark_workspace`` from ``benchmarks.conftest`` for tests under ``tests/benchmarks/``.

The fixture is defined in ``benchmarks/conftest.py`` but ``tests/benchmarks/`` does not
automatically load that conftest (pytest only walks upward to the nearest conftest ancestor,
which is ``tests/conftest.py`` for files in ``tests/benchmarks/``). Creating this proxy conftest
ensures the fixture is available without duplicating the implementation.
"""

from __future__ import annotations

from benchmarks.conftest import benchmark_workspace

__all__ = ["benchmark_workspace"]
