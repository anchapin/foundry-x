# evolution_full_loop_real_model fixtures

This directory exists to satisfy the benchmark hygiene check in
`tests/benchmarks/test_hygiene.py`, which requires every non-smoke
`BenchmarkTask` to have a matching fixture directory. The
`evolution_full_loop_real_model` task (issue #484) does not consume
fixture data — it is a Phase-3 plumbing validation that drives
`Runner.run_task` against the live `llama-server` endpoint with the
`sort_a_list` prompt (inlined in
`benchmarks/tasks/test_evolution_full_loop_real_model.py`) and then
processes the trace through `run_evolution_step`.

The fixture directory is intentionally minimal. Adding data files here
would imply the live test consumes them, which it does not: the test
sets up its own workspace via `FOUNDRY_RUN_LIVE_LLM`-gated code paths.
