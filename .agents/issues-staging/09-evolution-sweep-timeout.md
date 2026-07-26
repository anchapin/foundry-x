## Motivation

`Critic._run_sweep_for_quant` (`critic.py:400-404`) spawns `pytest` via `subprocess.run` WITHOUT passing `timeout=self.gate_timeout_s`. The same class's `evaluate()` method enforces `gate_timeout_s` on all three of its subprocess calls (git apply at :668, load_check at :705, pytest at :739-746).

Issue #890 introduced `gate_timeout_s` specifically to prevent a hanging subprocess from running indefinitely, but its acceptance criterion 2 was scoped to "subprocess.run calls inside `evaluate()`" — the sweep path was never covered. A quantization sweep runs `pytest` serially for each quantization; a single hang blocks all subsequent quantizations indefinitely with no bound.

No test in `tests/evolution/test_quantization_sweep.py` references timeout.

This is a resource-safety / reliability gap: a stuck sweep subprocess can produce a false verdict (timeout masquerading as a pass/fail) or block the sweep CLI indefinitely, degrading regression-detection reliability.

## Evidence

- `src/foundry_x/evolution/critic.py:400-404` — `_run_sweep_for_quant` `subprocess.run` has no `timeout=` kwarg
- `src/foundry_x/evolution/critic.py:739-746` — `evaluate()` pytest `subprocess.run` passes `timeout=self.gate_timeout_s`
- `src/foundry_x/evolution/critic.py:222-242` — `gate_timeout_s` stored on Critic instance, available to `_run_sweep_for_quant` via `self`
- Issue #890 — acceptance criterion 2 explicitly scoped to `evaluate()` subprocesses; sweep uncovered
- `tests/evolution/test_quantization_sweep.py` — zero references to "timeout"

## Risk

Low. Default `gate_timeout_s=None` preserves unbounded behavior; operators opt in. When set, a `TimeoutExpired` is caught and the affected quantization gets `pass_rate=0.0` with a timeout note.

## Acceptance Criteria

1. `_run_sweep_for_quant` passes `timeout=self.gate_timeout_s` to `subprocess.run`
2. On `subprocess.TimeoutExpired` the sweep does not crash; the affected quantization gets `pass_rate=0.0` with a notes field indicating timeout
3. When `gate_timeout_s=None` (default) behavior is unchanged
4. `tests/evolution/test_quantization_sweep.py` has a test exercising the timeout path (monkeypatched short timeout produces zero-pass-rate result)

## ADR(s)

ADR-0004 — advances (Critic gate robustness)
ADR-0016 — advances (quantization sweep reliability)
