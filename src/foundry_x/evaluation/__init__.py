"""External-eval validation machinery (issue #900, ADR-0023).

This package holds the *machinery* for validating the internal benchmark
suite under ``benchmarks/tasks/`` against external coding evals
(HumanEval+, SWE-bench). It is the data-science layer: it computes the
correlation between internal and external pass/fail signals so an ADR
can record whether ``kpi-improvement-rate`` (PRD §5, ADR-0005) is a
faithful proxy for real-world coding capability.

The package intentionally does **not** depend on pytest, the Runner, or
the TraceLogger, so it can be imported by both the offline unit tests
under ``tests/`` and the real-model orchestrator under
``infra/scripts/run_external_eval.sh``.

Members
-------
- :mod:`foundry_x.evaluation.correlation` — pure Pearson correlation on
  paired binary pass/fail observations, with input validation that the
  ADR-0023 methodology requires.
- :mod:`foundry_x.evaluation.humaneval_plus` — pydantic schema for the
  HumanEval+ JSONL shape, a deterministic JSONL loader, and a
  canonical-solution runner used to validate the plumbing offline.
"""
