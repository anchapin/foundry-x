# critic_security fixtures

This directory exists to satisfy the benchmark hygiene check in
`tests/benchmarks/test_hygiene.py`, which requires every non-smoke
`BenchmarkTask` to have a matching fixture directory. The
`critic_security` task (issue #333) does not consume fixture data -- its
tests use ``tmp_path`` to create isolated harness workspaces at runtime.

The tests exercise ``Critic.evaluate()`` security gates:
1. ``_contains_injection`` rejects diffs containing injection-like text
2. ``max_diff_lines`` rejects diffs exceeding the configured size cap

Both guards are evaluated against synthetic diff strings constructed
in-code; no external fixture files are required.
