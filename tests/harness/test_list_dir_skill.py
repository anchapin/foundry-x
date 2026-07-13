"""Dispatch test for the ``list_dir`` skill and a synthetic-fixture walk
that exercises the contract on ``target.py`` (issue #105).

Issue #105 acceptance is two-pronged:

1. ``harness/skills/list_dir.json`` ships with the documented input/output
   schemas (covered parametrically by ``tests/harness/test_skills_load.py``
   and explicitly by ``test_list_dir_skill_exists_and_has_documented_contract``
   in the same module).
2. ``list_dir`` becomes a tool the agent can reach through ``HookRegistry``
   -- just like ``bash`` (issue #104, ``test_bash_skill_dispatch.py``).
3. A new test asserts list_dir on a workspace returns ``target.py`` among
   its entries, so the Phase 1 benchmark
   ``benchmarks/tasks/test_write_unit_test.py`` becomes discoverable.

This file is the smallest credible wiring for point (2) and the
behavioral sniff for point (3): ``ListDirDispatchHook`` is a test-side
hook that recognises ``ToolCall(name="list_dir", ...)``, walks the
requested directory using the same ``os.scandir``-shaped logic the
contract describes (sorted ``{name, kind, size}`` entries), and stuffs
the result into ``ToolResult.output``. The same hook is then exercised
against a synthetic fixture to prove the walk actually surfaces
``target.py``.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
from pathlib import Path

from harness.hooks.base import HookRegistry, ToolCall, ToolResult


REPO_ROOT = Path(__file__).resolve().parents[2]
LIST_DIR_SKILL_PATH = REPO_ROOT / "harness" / "skills" / "list_dir.json"

VALID_KINDS: tuple[str, ...] = ("file", "dir", "symlink", "other")


def _classify(entry: os.DirEntry[str]) -> str:
    """Map an ``os.DirEntry`` to one of the kind tags the contract declares.

    Mirrors the resolution order described in ``list_dir.json``:
    ``is_symlink`` (``"symlink"`` even when the link target is a regular
    file, so the agent can decide whether to follow), ``is_dir``
    (``"dir"``), ``is_file`` (``"file"``), else ``"other"`` (sockets,
    FIFOs, block/character devices).
    """
    if entry.is_symlink():
        return "symlink"
    if entry.is_dir(follow_symlinks=False):
        return "dir"
    if entry.is_file(follow_symlinks=False):
        return "file"
    return "other"


def perform_list_dir(
    path: Path,
    *,
    glob: str | None = None,
    include_hidden: bool = False,
    max_entries: int = 1000,
) -> dict:
    """Walk ``path`` and return a list_dir-shaped result.

    Pure-Python implementation of the ``list_dir`` contract, used by both
    the dispatch and the fixture walk tests. Sorting is by ``name``
    ascending (deterministic, stable); the entry cap mirrors
    ``list_dir.json`` so a directory containing ``> max_entries`` real
    entries sets ``truncated=True``.

    The implementation deliberately uses ``os.scandir`` (Python stdlib,
    the only introspection primitive the contract allows under
    SECURITY.md \u00a71 threat #3) and ``fnmatch.fnmatch`` (stdlib) for the
    optional ``glob`` filter so the test pinpoints contract violations
    if either guard rail is loosened.
    """
    if not path.exists() or not path.is_dir():
        return {"entries": [], "truncated": False}

    raw_entries: list[tuple[str, str, int]] = []
    for entry in os.scandir(path):
        if not include_hidden and entry.name.startswith("."):
            continue
        if glob is not None and not fnmatch.fnmatch(entry.name, glob):
            continue
        kind = _classify(entry)
        size = 0 if kind != "file" else entry.stat(follow_symlinks=False).st_size
        raw_entries.append((entry.name, kind, size))

    raw_entries.sort(key=lambda triple: triple[0])
    truncated = len(raw_entries) > max_entries
    visible = raw_entries[:max_entries]

    return {
        "entries": [{"name": name, "kind": kind, "size": size} for name, kind, size in visible],
        "truncated": truncated,
    }


class ListDirDispatchHook:
    """Records and serves ``list_dir`` ``ToolCall``s the registry routes to it.

    Implements the ``Hook`` protocol (pre/post) and *does* perform the
    list_dir walk in ``post_tool`` so the dispatch test exercises the
    full contract end-to-end against ``HookRegistry``. The future
    production implementation lives outside the harness (a hook bound
    by the runner at startup, per the issue #104 precedent); this stub
    exists to prove the JSON contract is reachable through the registry
    shape the runner uses.
    """

    def __init__(self) -> None:
        self.pre_calls: list[ToolCall] = []
        self.handled: list[ToolResult] = []

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        if call.name == "list_dir":
            self.pre_calls.append(call)
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        if call.name == "list_dir":
            args = call.arguments
            path = Path(args["path"])
            output = perform_list_dir(
                path,
                glob=args.get("glob"),
                include_hidden=bool(args.get("include_hidden", False)),
                max_entries=int(args.get("max_entries", 1000)),
            )
            new_result = ToolResult(name="list_dir", output=output)
            self.handled.append(new_result)
            return new_result
        return result


class HarnessStub:
    """Empty harness stand-in for dispatch tests (issue #104 pattern).

    Holds no hooks and exposes no surface; exists only so the test can
    assert that ``HookRegistry`` is the layer that fans a ``ToolCall``
    out to registered hooks regardless of the harness backing it. The
    class deliberately has no behaviour so a future edit cannot
    accidentally add a hook here and silently change the test's
    coverage.
    """


def _run(coro):
    return asyncio.run(coro)


def test_list_dir_skill_json_is_loadable() -> None:
    doc = json.loads(LIST_DIR_SKILL_PATH.read_text(encoding="utf-8"))
    assert doc["name"] == "list_dir"


def test_list_dir_skill_required_keys_match_issue_acceptance() -> None:
    """Acceptance-criteria snapshot from issue #105.

    Lives in addition to the parametrized tests in
    ``test_skills_load.py`` so a regression that weakens the
    ``list_dir`` input or output contract (the *exact* shape the issue
    acceptance criterion pins) is surfaced with a precise message.
    """
    doc = json.loads(LIST_DIR_SKILL_PATH.read_text(encoding="utf-8"))
    assert set(doc["input_schema"]["required"]) == {"path"}, (
        "issue #105 acceptance: list_dir input_schema.required must be exactly "
        "{'path'} (no over- or under-specifying the input surface)"
    )
    assert set(doc["output_schema"]["required"]) == {"entries", "truncated"}, (
        "issue #105 acceptance: list_dir output_schema.required must be exactly "
        "{'entries', 'truncated'} (no over- or under-specifying the output surface)"
    )
    entry_props = doc["output_schema"]["properties"]["entries"]["items"]["properties"]
    assert set(entry_props) == {
        "name",
        "kind",
        "size",
    }, "issue #105 acceptance: each entry must declare exactly {name, kind, size}"


def test_registry_routes_list_dir_call_through_dispatch_hook() -> None:
    """A ``list_dir`` ``ToolCall`` flows through ``HookRegistry`` cleanly.

    Companion to ``test_registry_dispatches_bash_call_without_raising``
    (issue #104): the registry delivers the call to
    ``ListDirDispatchHook`` (via ``run_pre``) and the hook computes a
    real ``list_dir`` result on its way through ``run_post``. If either
    side raises the test fails -- AGENTS.md \u00a72 forbids silently
    swallowing exceptions.
    """
    HarnessStub()

    registry = HookRegistry()
    hook = ListDirDispatchHook()
    registry.register(hook)

    call = ToolCall(
        name="list_dir",
        arguments={"path": str(REPO_ROOT / "harness" / "skills")},
    )
    out_call = _run(registry.run_pre(call))
    assert out_call is call
    assert hook.pre_calls == [call]

    seed_result = ToolResult(name="list_dir", output={"entries": [], "truncated": False})
    out_result = _run(registry.run_post(call, seed_result))

    assert out_result is not seed_result, (
        "post_tool must replace the result with the walk output, not pass the seed through"
    )
    assert out_result.name == "list_dir"
    out_entries = out_result.output["entries"]
    assert "bash.json" in {entry["name"] for entry in out_entries}, (
        "list_dir walk on harness/skills/ must surface the bash skill "
        "(acceptance: a directory walk against the skill directory works)"
    )
    for entry in out_entries:
        assert set(entry) == {"name", "kind", "size"}, entry
        assert entry["kind"] in VALID_KINDS, entry
        assert isinstance(entry["size"], int) and entry["size"] >= 0, entry
    assert isinstance(out_result.output["truncated"], bool)


def test_list_dir_walks_synthetic_fixture_and_finds_target(tmp_path: Path) -> None:
    """Acceptance: list_dir on a synthetic workspace returns ``target.py``.

    Mirrors the Phase 1 benchmark
    ``benchmarks/tasks/test_write_unit_test.py`` (issue #105 evidence:
    the agent must locate the target module the benchmark author did
    not name). Builds a deterministic workspace in ``tmp_path``,
    walks it through ``perform_list_dir``, and asserts ``target.py``
    appears with ``kind == "file"`` and a non-zero ``size``.

    Pulled out as its own test (rather than folded into the dispatch
    test above) so the fixture walk reads cleanly as the
    acceptance-criteria pin. The benchmark's actual
    ``benchmarks/fixtures/write_unit_test/target.py`` is *not* used as
    the target on purpose: this test is hermetic so it survives any
    future reorganisation of the benchmark fixtures tree (the contract
    is the contract, not the fixture's location).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "target.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (workspace / "helper.py").write_text("def helper():\n    return None\n", encoding="utf-8")
    (workspace / "README.md").write_text("fixture for issue #105\n", encoding="utf-8")

    result = perform_list_dir(workspace)

    names = [entry["name"] for entry in result["entries"]]
    assert "target.py" in names, (
        "list_dir on the synthetic workspace must surface target.py (issue #105 acceptance)"
    )
    assert "helper.py" in names
    assert "README.md" in names
    assert result["truncated"] is False, (
        f"workspace has 3 entries, well under max_entries=1000; "
        f"got truncated={result['truncated']!r}"
    )

    target_entry = next(entry for entry in result["entries"] if entry["name"] == "target.py")
    assert target_entry["kind"] == "file", target_entry
    assert target_entry["size"] > 0, (
        f"target.py was written with non-empty content; entry reports size={target_entry['size']}"
    )


def test_list_dir_glob_filter_only_returns_matching_entries(tmp_path: Path) -> None:
    """The optional ``glob`` input narrows the result; the agent composes with grep_search.

    Pins the contract's composition story: an agent that wants only
    Python files calls ``list_dir`` with ``glob='*.py'``; the hook must
    respect the filter and return only the matched entries (kind stays
    ``"file"`` because the walk still classifies by stat).
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "target.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "helper.py").write_text("y = 2\n", encoding="utf-8")
    (workspace / "data.txt").write_text("not py\n", encoding="utf-8")

    result = perform_list_dir(workspace, glob="*.py")

    names = sorted(entry["name"] for entry in result["entries"])
    assert names == ["helper.py", "target.py"], f"glob='*.py' must drop data.txt; got {names!r}"


def test_list_dir_max_entries_caps_result_and_sets_truncated(tmp_path: Path) -> None:
    """When a directory holds more entries than ``max_entries``, truncation is honest.

    The contract's per-call safety bound (SECURITY.md threat #5):
    dropping the excess deterministically and surfacing ``truncated``
    so the agent can re-call with a narrower ``glob``.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    for i in range(5):
        (workspace / f"file_{i:02d}.txt").write_text(f"content {i}\n", encoding="utf-8")

    result = perform_list_dir(workspace, max_entries=3)

    assert result["truncated"] is True, (
        "5 real entries against max_entries=3 must set truncated=True"
    )
    assert len(result["entries"]) == 3, (
        f"exactly max_entries items should be returned; got {len(result['entries'])}"
    )
    seen = {entry["name"] for entry in result["entries"]}
    assert len(seen) == 3, "entries must be unique after the cap"


def test_list_dir_excludes_hidden_entries_by_default(tmp_path: Path) -> None:
    """``include_hidden=False`` is the conservative default -- dotfiles do not leak.

    Pins the SECURITY.md threat #5 guardrail inside the contract:
    ``list_dir`` must not silently surface ``.env`` or ``.git`` into
    the agent's context without an explicit opt-in.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "visible.txt").write_text("ok\n", encoding="utf-8")
    (workspace / ".env").write_text("SECRET=top\n", encoding="utf-8")

    default_result = perform_list_dir(workspace)
    assert {entry["name"] for entry in default_result["entries"]} == {"visible.txt"}, (
        "default list_dir must exclude dotfiles"
    )

    revealed = perform_list_dir(workspace, include_hidden=True)
    assert {entry["name"] for entry in revealed["entries"]} == {
        ".env",
        "visible.txt",
    }, "include_hidden=True is the explicit opt-in to surface dotfiles"
