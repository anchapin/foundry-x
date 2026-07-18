"""External-eval slice directory (issue #900, ADR-0023).

This package marks ``benchmarks/external/`` as an importable location so
tooling under ``src/foundry_x/evaluation/`` can resolve slice paths
relative to it. The actual slice data lives as JSONL files alongside
this module (``humaneval_plus_sample.jsonl``).
"""
