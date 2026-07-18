"""Post-write correction helpers for :class:`TraceLogger` (issue #157).

Acceptance from the issue body:

- ``TraceLogger.delete_session(session_id)`` removes the session and its
  events from the underlying store. Idempotent: a second call with the
  same (or never-existed) ``session_id`` returns ``True`` rather than
  raising, so an operator running it on a stale id does not see an
  error.
- ``TraceLogger.redact_event(session_id, event_index, key)`` rewrites
  ``payload[key]`` to ``"[REDACTED]"`` on the indexed event. The
  redaction is persisted; a fresh ``TraceLogger`` reads the redacted
  value back.

Both helpers operate on both the sqlite and jsonl backends.

Issue #193 extends this with regression coverage for multi-event
sessions: 50 events across three sessions, out-of-range redaction that
must NOT rewrite the store, and timestamp ordering preserved across
redaction. Same scenarios run against both backends so the rewrite
paths stay in lock-step.
"""

from __future__ import annotations

import json
import os
import sqlite3
from unittest import mock

import pytest

from foundry_x.trace.logger import TraceLogger, TraceEvent

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _suffix(backend: str) -> str:
    return ".db" if backend == "sqlite" else ".jsonl"


@_BACKENDS
def test_redact_event_replaces_payload_value_with_redacted(tmp_path, backend):
    """``redact_event`` overwrites ``payload[key]`` with ``"[REDACTED]"``."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "1234567890abcdef"
    bearer = "Bearer abcdefghij1234567890"
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_result",
            payload={
                "output": f"key={api_key}",
                "header": f"Authorization: {bearer}",
                "metadata": "keep-me",
            },
        )
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.redact_event(sid, event_index=0, key="output") is True
    assert logger.redact_event(sid, event_index=0, key="header") is True

    events = logger.load_session(sid)
    payload = events[0].payload
    assert payload["output"] == "[REDACTED]"
    assert payload["header"] == "[REDACTED]"
    assert payload["metadata"] == "keep-me"
    assert "[REDACTED]" in json.dumps(payload)
    assert api_key not in json.dumps(payload)
    assert bearer not in json.dumps(payload)
    assert events[1].payload == {"text": "hi"}


@_BACKENDS
def test_redact_event_persists_across_new_logger_instance(tmp_path, backend):
    """The redaction is written to disk, so a fresh ``TraceLogger``
    instance on the same path reads the redacted value back."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "abcdef0123456789"
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_call",
            payload={"secret": api_key, "name": "read_file"},
        )

    assert logger.redact_event(sid, event_index=0, key="secret") is True

    fresh = TraceLogger(path, backend=backend)
    events = fresh.load_session(sid)
    assert events[0].payload["secret"] == "[REDACTED]"
    assert events[0].payload["name"] == "read_file"
    assert api_key not in json.dumps(events[0].payload)


@_BACKENDS
def test_redact_event_out_of_range_returns_false(tmp_path, backend):
    """A stale ``event_index`` returns ``False`` without rewriting anything."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.redact_event(sid, event_index=99, key="text") is False
    assert logger.redact_event(sid, event_index=-1, key="text") is False

    events = logger.load_session(sid)
    assert events[0].payload == {"text": "hi"}


@_BACKENDS
def test_redact_event_unknown_session_returns_false(tmp_path, backend):
    """A non-existent ``session_id`` returns ``False`` without crashing."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert logger.redact_event("never-existed", event_index=0, key="text") is False


@_BACKENDS
def test_delete_session_removes_target_and_keeps_others(tmp_path, backend):
    """``delete_session`` removes the target session and its events, leaving
    every other session in the store untouched."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid_drop:
        logger.record(sid_drop, kind="user_prompt", payload={"text": "drop-me"})
        logger.record(sid_drop, kind="tool_call", payload={"name": "read"})
    with logger.session(harness_version="0.1.0") as sid_keep:
        logger.record(sid_keep, kind="user_prompt", payload={"text": "keep-me"})

    assert logger.delete_session(sid_drop) is True

    assert logger.load_session(sid_drop) == []
    surviving = [s.session_id for s in logger.list_sessions()]
    assert sid_drop not in surviving
    assert sid_keep in surviving
    assert len(logger.load_session(sid_keep)) == 1
    assert logger.load_session(sid_keep)[0].payload == {"text": "keep-me"}


@_BACKENDS
def test_delete_session_persists_across_new_logger_instance(tmp_path, backend):
    """The deletion is written to disk, so a fresh ``TraceLogger``
    instance on the same path sees the post-delete state."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "x"})

    assert logger.delete_session(sid) is True

    fresh = TraceLogger(path, backend=backend)
    assert fresh.load_session(sid) == []
    assert [s.session_id for s in fresh.list_sessions()] == []


@_BACKENDS
def test_delete_session_is_idempotent(tmp_path, backend):
    """A second ``delete_session`` on the same id, or one on a never-existed
    id, returns ``True`` rather than raising."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.delete_session(sid) is True
    assert logger.delete_session(sid) is True
    assert logger.delete_session("never-existed") is True


@_BACKENDS
def test_delete_session_on_empty_store_returns_true(tmp_path, backend):
    """Calling ``delete_session`` on a never-touched store is a no-op."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert logger.delete_session("anything") is True


# --- Issue #193: multi-event redact_event regression coverage -----------------
# The single-event tests above exercise redact_event against a one-event
# session. Production sessions carry tens of events interleaved across
# kinds and sessions. The jsonl backend rewrites the whole file on every
# redact (logger.py:603-631) so the corner cases -- wrong event_index
# ordering, ordering-stable output after rewrite, foreign session_id rows
# preserved -- need an explicit multi-event regression.


def _raw_payload_bytes_sqlite(path, session_id: str) -> dict[str, bytes]:
    """Return ``{event_id: raw_payload_text_bytes}`` for the session.

    The redact path targets ``events.payload`` as a single TEXT column;
    byte-stability of a survivor means this column is byte-equal to its
    pre-redact value. Loading via ``load_session`` and re-serializing
    would re-canonicalize whitespace and key order, so we read the raw
    TEXT instead.
    """
    out: dict[str, bytes] = {}
    with sqlite3.connect(path) as conn:
        for event_id, payload_text in conn.execute(
            "SELECT event_id, payload FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchall():
            out[event_id] = payload_text.encode("utf-8")
    return out


def _raw_event_lines_jsonl(path) -> dict[str, bytes]:
    """Return ``{event_id: line_bytes}`` from the jsonl trace file.

    Each event in a jsonl backend occupies one line. Byte-stability of
    a survivor means the entire line (including the trailing newline)
    is unchanged. Markers (``session_start`` / ``session_end``) are
    excluded because they have no ``event_id``.
    """
    out: dict[str, bytes] = {}
    raw = path.read_bytes().splitlines(keepends=True)
    for line in raw:
        if not line.strip():
            continue
        record = json.loads(line)
        event_id = record.get("event_id")
        if event_id is None:
            continue
        out[event_id] = line
    return out


def _plant_three_sessions(
    logger: TraceLogger, plan: list[tuple[str, int]]
) -> tuple[list[str], list[TraceEvent]]:
    """Plant ``plan`` events per session and return ``(session_ids, flat_events)``.

    Each event carries a payload whose ``secret`` field is unique per
    event so we can verify byte-level stability later. The session
    markers emitted by ``logger.session()`` are part of the jsonl file
    but are not events.
    """
    sids: list[str] = []
    flat: list[TraceEvent] = []
    for label, count in plan:
        with logger.session(harness_version="0.1.0") as sid:
            sids.append(sid)
            for i in range(count):
                ev = logger.record(
                    sid,
                    kind="tool_call" if i % 2 == 0 else "tool_result",
                    payload={
                        "label": label,
                        "i": i,
                        # Distinct value per event -- lets a survivor-vs-redacted
                        # confusion show up as a string mismatch.
                        "secret": f"sk-{label}{i:03d}abcdef",
                    },
                )
                flat.append(ev)
    return sids, flat


@_BACKENDS
def test_redact_event_preserves_byte_stability_of_survivors(tmp_path, backend):
    """50 events across three sessions; redacting one leaves the other 49
    events byte-stable in their underlying representation.

    - jsonl: every surviving event line (including the trailing newline)
      is byte-identical to its pre-redact counterpart. Foreign-session
      rows are also byte-stable -- the rewrite path must not touch them.
    - sqlite: every surviving row's ``payload`` TEXT column is byte-
      identical to its pre-redact value.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    plan = [("a", 18), ("b", 17), ("c", 15)]
    sids, flat = _plant_three_sessions(logger, plan)
    assert len(flat) == 50

    # Capture pre-redact raw bytes per event
    if backend == "jsonl":
        pre_bytes = _raw_event_lines_jsonl(path)
    else:
        pre_bytes_per_session = {sid: _raw_payload_bytes_sqlite(path, sid) for sid in sids}
        pre_bytes = {eid: b for d in pre_bytes_per_session.values() for eid, b in d.items()}
    assert len(pre_bytes) == 50

    # Redact one event in the middle of session B
    target_session = sids[1]
    target_index_in_session = 8
    target_event = [e for e in flat if e.session_id == target_session][target_index_in_session]
    assert (
        logger.redact_event(target_session, event_index=target_index_in_session, key="secret")
        is True
    )

    # Capture post-redact raw bytes
    if backend == "jsonl":
        post_bytes = _raw_event_lines_jsonl(path)
    else:
        post_bytes_per_session = {sid: _raw_payload_bytes_sqlite(path, sid) for sid in sids}
        post_bytes = {eid: b for d in post_bytes_per_session.values() for eid, b in d.items()}
    assert len(post_bytes) == 50

    redacted_ids = {target_event.event_id}
    surviving_ids = set(pre_bytes) - redacted_ids
    assert len(surviving_ids) == 49

    # Every survivor is byte-stable: same event_id set, same byte payload.
    assert set(post_bytes) == set(pre_bytes)
    for eid in surviving_ids:
        assert post_bytes[eid] == pre_bytes[eid], (
            f"survivor event {eid} changed bytes during redact: "
            f"{pre_bytes[eid]!r} -> {post_bytes[eid]!r}"
        )

    # The redacted event itself was rewritten (its payload changed).
    assert post_bytes[target_event.event_id] != pre_bytes[target_event.event_id]
    if backend == "jsonl":
        redacted_record = json.loads(post_bytes[target_event.event_id].decode("utf-8"))
        assert redacted_record["payload"]["secret"] == "[REDACTED]"
    else:
        # sqlite stores the payload dict as JSON TEXT in the ``events.payload``
        # column directly, so the post-redact bytes are the payload itself.
        redacted_payload = json.loads(post_bytes[target_event.event_id].decode("utf-8"))
        assert redacted_payload["secret"] == "[REDACTED]"

    # And the parsed view confirms only that key was rewritten.
    redacted_events = list(logger.load_session(target_session))
    redacted_event = next(e for e in redacted_events if e.event_id == target_event.event_id)
    assert redacted_event.payload["secret"] == "[REDACTED]"
    assert redacted_event.payload["label"] == target_event.payload["label"]
    assert redacted_event.payload["i"] == target_event.payload["i"]
    assert redacted_event.event_id == target_event.event_id
    assert redacted_event.timestamp == target_event.timestamp


@_BACKENDS
def test_redact_event_out_of_range_does_not_rewrite_store(tmp_path, backend):
    """In a multi-event session, an out-of-range ``event_index`` returns
    ``False`` and leaves the underlying store unchanged.

    - jsonl: the file's bytes are identical before and after the call.
    - sqlite: every row's ``payload`` TEXT is identical before and after.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        for i in range(20):
            logger.record(
                sid,
                kind="user_prompt",
                payload={"i": i, "msg": f"hello-{i:02d}"},
            )

    pre_events = list(logger.load_session(sid))
    assert len(pre_events) == 20
    if backend == "jsonl":
        pre_file_bytes = path.read_bytes()
        pre_line_bytes = _raw_event_lines_jsonl(path)
    else:
        pre_payload_bytes = _raw_payload_bytes_sqlite(path, sid)

    # Indices that must all be rejected without a rewrite.
    assert logger.redact_event(sid, event_index=99, key="msg") is False
    assert logger.redact_event(sid, event_index=20, key="msg") is False
    assert logger.redact_event(sid, event_index=-1, key="msg") is False

    if backend == "jsonl":
        post_file_bytes = path.read_bytes()
        post_line_bytes = _raw_event_lines_jsonl(path)
        # File must be byte-identical: a no-op redact must not touch disk.
        assert post_file_bytes == pre_file_bytes, "jsonl file was rewritten despite no-op redact"
        assert post_line_bytes == pre_line_bytes
    else:
        post_payload_bytes = _raw_payload_bytes_sqlite(path, sid)
        assert post_payload_bytes == pre_payload_bytes

    # And the parsed view is unchanged too -- a sanity check on top of the
    # byte-level check, in case a future refactor introduces a no-op write
    # that happens to round-trip.
    post_events = list(logger.load_session(sid))
    assert post_events == pre_events


@_BACKENDS
def test_redact_event_preserves_timestamp_ordering(tmp_path, backend):
    """After redacting several events, ``load_session`` returns the same
    events in the same order: same ``event_id`` at each position, same
    timestamp at each position. Ordering is part of the contract -- the
    jsonl rewrite path sorts lines by timestamp to recover the original
    order, and the sqlite path orders by timestamp in SQL.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        for i in range(20):
            logger.record(
                sid,
                kind="tool_call",
                payload={"i": i, "secret": f"sk-pre{i:04d}xx"},
            )

    pre_events = list(logger.load_session(sid))
    pre_ids = [e.event_id for e in pre_events]
    pre_timestamps = [e.timestamp for e in pre_events]
    pre_payloads = [e.payload for e in pre_events]
    # Sanity: timestamps are already non-decreasing on insertion.
    assert pre_timestamps == sorted(pre_timestamps)

    # Redact three non-adjacent events spread across the session.
    redacted_indices = [3, 9, 15]
    for idx in redacted_indices:
        assert logger.redact_event(sid, event_index=idx, key="secret") is True

    post_events = list(logger.load_session(sid))
    post_ids = [e.event_id for e in post_events]
    post_timestamps = [e.timestamp for e in post_events]

    # Same events at each position, same timestamps, same order.
    assert post_ids == pre_ids
    assert post_timestamps == pre_timestamps
    # Timestamp sequence is still non-decreasing after the rewrite.
    assert post_timestamps == sorted(post_timestamps)

    # Only the targeted positions got their payload rewritten.
    for idx, post in enumerate(post_events):
        if idx in redacted_indices:
            assert post.payload["secret"] == "[REDACTED]"
            assert pre_payloads[idx]["secret"] != "[REDACTED]"
            # Other payload keys untouched.
            assert post.payload["i"] == pre_payloads[idx]["i"]
        else:
            assert post.payload == pre_payloads[idx], (
                f"non-redacted event at index {idx} changed payload: "
                f"{pre_payloads[idx]} -> {post.payload}"
            )


# --- Issue #752: prune_sessions bulk delete ------------------------------------


@_BACKENDS
def test_prune_sessions_removes_target_and_keeps_others(tmp_path, backend):
    """``prune_sessions`` removes the target sessions and their events, leaving
    every other session in the store untouched."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid_drop1:
        logger.record(sid_drop1, kind="user_prompt", payload={"text": "drop-me-1"})
    with logger.session(harness_version="0.1.0") as sid_drop2:
        logger.record(sid_drop2, kind="user_prompt", payload={"text": "drop-me-2"})
    with logger.session(harness_version="0.1.0") as sid_keep:
        logger.record(sid_keep, kind="user_prompt", payload={"text": "keep-me"})

    deleted = logger.prune_sessions([sid_drop1, sid_drop2])
    assert deleted == 2

    surviving = [s.session_id for s in logger.list_sessions()]
    assert sid_drop1 not in surviving
    assert sid_drop2 not in surviving
    assert sid_keep in surviving
    assert len(logger.load_session(sid_keep)) == 1
    assert logger.load_session(sid_keep)[0].payload == {"text": "keep-me"}


@_BACKENDS
def test_prune_sessions_persists_across_new_logger_instance(tmp_path, backend):
    """The deletion is written to disk, so a fresh ``TraceLogger``
    instance on the same path sees the post-delete state."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid1:
        logger.record(sid1, kind="user_prompt", payload={"text": "x"})
    with logger.session(harness_version="0.1.0") as sid2:
        logger.record(sid2, kind="user_prompt", payload={"text": "y"})

    assert logger.prune_sessions([sid1, sid2]) == 2

    fresh = TraceLogger(path, backend=backend)
    assert fresh.list_sessions() == []


@_BACKENDS
def test_prune_sessions_empty_list_is_noop(tmp_path, backend):
    """Calling ``prune_sessions`` with an empty list is a no-op."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.prune_sessions([]) == 0
    assert len(logger.list_sessions()) == 1


@_BACKENDS
def test_prune_sessions_unknown_ids_returns_zero(tmp_path, backend):
    """Passing ids that do not exist returns 0 and touches nothing."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.prune_sessions(["never-existed-1", "never-existed-2"]) == 0
    assert len(logger.list_sessions()) == 1


# --- Issue #787: streaming + atomic rewrite of _delete_session_jsonl ---------
# Acceptance from the issue body:
# - _delete_session_jsonl uses O(1) peak memory (streaming read + write)
# - Uses atomic rename (temp file + os.replace) to avoid partial-write
#   corruption
# These tests target the jsonl backend specifically (the sqlite path is
# unaffected) and exercise the new file-rewrite contract directly.


def test_delete_session_jsonl_leaves_no_temp_file_on_success(tmp_path):
    """After a successful streaming delete, no ``.tmp`` file lingers in
    the trace directory. A leaked temp file would accumulate over many
    prune cycles and confuse operators (issue #787)."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid_drop:
        logger.record(sid_drop, kind="user_prompt", payload={"text": "drop"})
    with logger.session(harness_version="0.1.0") as sid_keep:
        logger.record(sid_keep, kind="user_prompt", payload={"text": "keep"})

    assert logger.delete_session(sid_drop) is True

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp file leaked after delete: {leftovers}"


def test_delete_session_jsonl_preserves_file_mode(tmp_path):
    """The rewritten file inherits the original's mode. The streaming
    path uses :func:`os.chmod` to copy the source mode onto the temp
    file before ``os.replace`` so a 0644 trace file does not silently
    become 0600 (the NamedTemporaryFile default)."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "x"})

    # Pin a non-default mode and verify it survives the rewrite.
    os.chmod(path, 0o640)
    pre_mode = path.stat().st_mode & 0o777
    assert pre_mode == 0o640

    assert logger.delete_session(sid) is True

    post_mode = path.stat().st_mode & 0o777
    assert post_mode == pre_mode, (
        f"file mode changed across streaming rewrite: {pre_mode:o} -> {post_mode:o}"
    )


def test_delete_session_jsonl_rolls_back_on_replace_failure(tmp_path):
    """If ``os.replace`` raises (e.g. the rename is interrupted), the
    original trace file must be byte-identical to its pre-delete state
    and the temp file must be cleaned up. The atomic-rename contract
    (issue #787) forbids partial writes from ever becoming visible."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid_drop:
        logger.record(sid_drop, kind="user_prompt", payload={"text": "drop"})
    with logger.session(harness_version="0.1.0") as sid_keep:
        logger.record(sid_keep, kind="user_prompt", payload={"text": "keep"})

    pre_bytes = path.read_bytes()

    def raising_replace(src, dst, *args, **kwargs):
        raise OSError("simulated rename failure")

    with mock.patch("foundry_x.trace.logger.os.replace", side_effect=raising_replace):
        with pytest.raises(OSError, match="simulated rename failure"):
            logger.delete_session(sid_drop)

    # Original file is untouched.
    assert path.read_bytes() == pre_bytes
    # Temp file is cleaned up even though the rename failed.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"temp file leaked after failed rename: {leftovers}"
    # The store still sees both sessions.
    sids = [s.session_id for s in logger.list_sessions()]
    assert sid_drop in sids
    assert sid_keep in sids


def test_delete_session_jsonl_streaming_handles_large_session(tmp_path):
    """The streaming path must correctly rewrite a file too large to
    hold in memory. We do not measure RSS here (flaky in CI), but we
    do verify correctness at a size that would OOM the old
    materialize-the-whole-file path on a constrained runner.

    Plants 5000 events across 5 sessions, deletes the middle one, and
    checks the survivor count and per-session integrity exactly."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    sids: list[str] = []
    for label in range(5):
        with logger.session(harness_version="0.1.0") as sid:
            sids.append(sid)
            for i in range(1000):
                logger.record(
                    sid,
                    kind="tool_call",
                    payload={"label": label, "i": i},
                )

    # Sanity: each session has 1000 events.
    assert all(len(logger.load_session(sid)) == 1000 for sid in sids)

    drop = sids[2]
    assert logger.delete_session(drop) is True

    # The dropped session is gone; the other four are intact.
    assert logger.load_session(drop) == []
    for sid in (sids[0], sids[1], sids[3], sids[4]):
        events = logger.load_session(sid)
        assert len(events) == 1000
        assert [e.payload["i"] for e in events] == list(range(1000))
    surviving = [s.session_id for s in logger.list_sessions()]
    assert drop not in surviving
    assert len(surviving) == 4
