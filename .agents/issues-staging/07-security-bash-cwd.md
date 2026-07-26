## Motivation

All file-operation skills (`read_file`, `write_file`, `edit_file`, `list_dir`, `grep_search`) route their path argument through `_resolve_path` (`runner.py:993`), which rejects paths escaping `workspace_root`. The `bash` skill executor does not.

`_bash_skill_executor` (`runner.py:778`) accepts a `cwd` argument from the model and passes it directly to `subprocess.run` as `Path(cwd_arg)` — no workspace confinement check:

```python
cwd: Path | None = None
if cwd_arg:
    cwd = Path(cwd_arg)          # no _resolve_path
elif workspace_dir:
    cwd = workspace_dir
```

When running unsandboxed (the local dev path documented in SECURITY.md), an injected instruction can cause the agent to call `bash(command="ls -la", cwd="/home/user/.ssh")`, enumerating host files via a relative `cwd` outside the workspace.

This is a regression in the workspace-confinement contract: every file-op skill enforces the boundary, but the bash skill was missed. It addresses SECURITY.md threat vector #6 (local privilege: "a buggy hook could read files outside the workspace").

## Evidence

- `src/foundry_x/execution/runner.py:778-836` — `_bash_skill_executor` function signature and body
- `src/foundry_x/execution/runner.py:796-797` — `cwd = Path(cwd_arg)` — no `_resolve_path` call
- `src/foundry_x/execution/runner.py:804` — `subprocess.run` receives `cwd` directly without path validation
- `src/foundry_x/execution/runner.py:993-1008` — `_resolve_path` resolves against `workspace_root` and rejects escapes; used by all 5 file-op skills
- `docs/SECURITY.md` Threat #6: "a buggy hook could read files outside the workspace"

## Risk

Low-Medium. Any agent workflow that legitimately sets `cwd` outside the workspace would break. In practice, the Docker sandbox confines the entire container, so only the unsandboxed local dev path is affected. Mitigation: return an error result (same pattern as file-op skills) rather than crashing.

## Acceptance Criteria

1. `cwd` argument to bash skill is validated through `_resolve_path` before passing to `subprocess.run`
2. If `cwd` resolves outside `workspace_root`, an error result is returned (same pattern as file-op skills that reject path escapes)
3. Existing bash tool-call tests still pass with the new resolution layer
4. A test exercises a `cwd` pointing outside `workspace_root` and asserts the error result

## ADR(s)

ADR-0010 — advances (workspace confinement contract enforced uniformly across all skills)
