from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from foundry_x.evolution.digester import Digester
from foundry_x.observability.render import render_failure_report
from foundry_x.observability.timeline import format_timeline
from foundry_x.trace.logger import TraceEvent, TraceLogger, TraceSession


def _render_failure(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.trace_path)
    events = logger.load_session(args.session_id)
    report = Digester().digest(args.session_id, events)
    markdown = render_failure_report(report)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown + "\n")
    return 0


def _format_session_row(session: TraceSession) -> str:
    fields = [
        session.session_id,
        session.started_at,
        session.harness_version,
        session.model_id or "-",
    ]
    if session.ended_at:
        fields.append(session.ended_at)
    return "  ".join(fields)


def _sessions(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.db)
    sessions = logger.list_sessions()
    if not sessions:
        sys.stdout.write("No sessions found.\n")
        return 0
    sys.stdout.write("session_id  started_at  harness_version  model_id\n")
    for session in sessions:
        sys.stdout.write(_format_session_row(session) + "\n")
    return 0


def _show(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    sys.stdout.write(f"Session: {args.session_id}\n")
    sys.stdout.write(f"Events: {len(events)}\n\n")
    sys.stdout.write(format_timeline(events) + "\n")
    return 0


def _serialize_event(event: TraceEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "session_id": event.session_id,
        "timestamp": event.timestamp,
        "kind": event.kind,
        "payload": event.payload,
    }


def _export(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    lines = [json.dumps(_serialize_event(event)) for event in events]
    output = "\n".join(lines)
    if lines:
        output += "\n"
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


# --- Issue #83: session-list / session-show / events-grep --------------------
# The three subcommands here are the named-developer-surface promised by
# ADR-0007 §Consequences ('Traces must be inspectable'). They co-exist with
# the ``sessions``/``show``/``export`` commands above; consolidating them is
# out of scope for #83 and tracked separately.


def _format_session_list_row(session: TraceSession) -> str:
    """Render one row of the ``session-list`` table.

    Column order matches the issue #83 acceptance criteria:
    ``session_id  started_at  ended_at  harness_version``. ``ended_at``
    may be empty while a session is still open, so the field is always
    emitted as a placeholder column rather than conditionally appended.
    """
    ended = session.ended_at if session.ended_at is not None else "-"
    return f"{session.session_id}  {session.started_at}  {ended}  {session.harness_version}"


def _session_list(args: argparse.Namespace) -> int:
    """Implement ``session-list`` (issue #83).

    Lists sessions ordered as ``TraceLogger.list_sessions`` returns them.
    ``--harness-version`` narrows to sessions recorded against a specific
    harness build; ``--limit`` truncates after N rows. The command exits 0
    even when the database is empty so it composes cleanly in shell pipes.
    """
    logger = TraceLogger(args.db)
    sessions = logger.list_sessions()
    if args.harness_version is not None:
        sessions = [s for s in sessions if s.harness_version == args.harness_version]
    if args.limit is not None:
        sessions = sessions[: args.limit]
    sys.stdout.write("session_id  started_at  ended_at  harness_version\n")
    for session in sessions:
        sys.stdout.write(_format_session_list_row(session) + "\n")
    return 0


def _session_show(args: argparse.Namespace) -> int:
    """Implement ``session-show`` (issue #83).

    Reuses ``observability.timeline.format_timeline`` so the rendered
    timeline stays consistent with ``show``/``render-failure``. An
    unknown session returns exit code 1 with a message on stderr,
    mirroring ``_show`` and the grep convention.
    """
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    sys.stdout.write(f"Session: {args.session_id}\n")
    sys.stdout.write(f"Events: {len(events)}\n\n")
    sys.stdout.write(format_timeline(events) + "\n")
    return 0


def _events_grep(args: argparse.Namespace) -> int:
    """Implement ``events-grep`` (issue #83).

    Scans every event payload in ``session_id`` for a regex match against
    the serialized JSON. Matching events are printed as one line each
    ``"<timestamp>  <kind>  <payload_json>"`` so the matched substring is
    visible in context and the output remains greppable. Exit codes
    follow the conventional grep semantics: 0 when at least one event
    matched, 1 when none matched. An invalid regex is re-raised after
    logging the parse error to stderr per the 'never silently swallow
    exceptions' rule in AGENTS.md.
    """
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    try:
        pattern = re.compile(args.pattern)
    except re.error as exc:
        sys.stderr.write(f"Invalid --pattern regex: {exc}\n")
        raise
    matches = 0
    for event in events:
        payload_text = json.dumps(event.payload, sort_keys=True)
        if pattern.search(payload_text):
            sys.stdout.write(f"{event.timestamp}  {event.kind}  {payload_text}\n")
            matches += 1
    return 0 if matches else 1


# --- Issue #192: redact-session / redact-key ---------------------------------
# These subcommands expose the post-write correction API (``delete_session``
# and ``redact_event`` from logger.py) so an on-call operator can respond to
# a secret leak within the SECURITY.md §Secrets response window without
# dropping into a Python REPL. Both accept ``--out`` to append a JSONL audit
# record so the remediation is traceable after the fact.


def _logger_for(db_path: str) -> TraceLogger:
    """Construct a :class:`TraceLogger` auto-detecting the backend.

    A ``--db`` path ending in ``.jsonl`` selects the JSONL backend; every
    other path (including the default ``logs/traces.db``) uses sqlite.
    Issue #192 requires both subcommands to work on either backend.
    """
    backend = "jsonl" if db_path.endswith(".jsonl") else "sqlite"
    return TraceLogger(db_path, backend=backend)


def _now_ts() -> str:
    return datetime.now(UTC).isoformat()


def _write_audit(args: argparse.Namespace, record: dict[str, Any]) -> None:
    """Append a JSONL audit line to ``args.out`` if it was provided."""
    if not args.out:
        return
    with Path(args.out).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _redact_session(args: argparse.Namespace) -> int:
    """Implement ``redact-session`` (issue #192).

    Counts the session's events (so the operator knows what was removed),
    deletes the session row and every event via ``TraceLogger.delete_session``
    (idempotent), prints the count, and optionally appends an audit record
    to ``--out``. Exits 0 even when the session did not exist, mirroring
    the idempotent contract of ``delete_session``.

    ``--dry-run`` (issue #1122) prints what would be removed without
    calling ``delete_session``.
    """
    logger = _logger_for(args.db)
    count = len(logger.load_session(args.session_id))
    if getattr(args, "dry_run", False):
        sessions = logger.list_sessions()
        session_meta = next((s for s in sessions if s.session_id == args.session_id), None)
        if session_meta:
            sys.stdout.write(
                f"  would redact {session_meta.session_id}"
                f"  started_at={session_meta.started_at}"
                f"  harness_version={session_meta.harness_version}\n"
            )
        sys.stdout.write(
            f"Dry run: would redact {count} event(s) from session {args.session_id}.\n"
        )
        return 0
    logger.delete_session(args.session_id)
    sys.stdout.write(f"Deleted session {args.session_id}: {count} event(s) removed.\n")
    _write_audit(
        args,
        {
            "action": "redact-session",
            "session_id": args.session_id,
            "events_deleted": count,
            "timestamp": _now_ts(),
        },
    )
    return 0


def _redact_key(args: argparse.Namespace) -> int:
    """Implement ``redact-key`` (issue #192).

    Rewrites ``payload[key]`` to ``"[REDACTED]"`` on the timestamp-ordered
    event at ``event_index`` via ``TraceLogger.redact_event``. An
    out-of-range index returns exit code 1 immediately so a stale index
    never silently rewrites the wrong row.
    """
    logger = _logger_for(args.db)
    ok = logger.redact_event(args.session_id, args.event_index, args.key)
    if not ok:
        sys.stderr.write(
            f"redact-key: event_index {args.event_index} is out of range "
            f"for session {args.session_id}.\n"
        )
        return 1
    sys.stdout.write(
        f"Redacted key '{args.key}' on event {args.event_index} of session {args.session_id}.\n"
    )
    _write_audit(
        args,
        {
            "action": "redact-key",
            "session_id": args.session_id,
            "event_index": args.event_index,
            "key": args.key,
            "timestamp": _now_ts(),
        },
    )
    return 0


# --- Issue #275: delete-session / prune --------------------------------------
# These subcommands expose retention management for the trace store.  ADR-0003
# flags unbounded growth as the trigger to "revisit" the store; Phase-3's
# mandate of many real benchmark runs/day makes a prune CLI essential for
# keeping ``logs/`` under control and Digester/KPI queries fast.


def _delete_session(args: argparse.Namespace) -> int:
    """Implement ``delete-session`` (issue #275).

    Removes one session and all its events via ``TraceLogger.delete_session``.
    Idempotent: exits 0 whether or not the session existed, mirroring the
    contract of the underlying primitive.

    ``--dry-run`` (issue #1122) prints what would be removed without
    calling ``delete_session``.
    """
    logger = _logger_for(args.db)
    count = len(logger.load_session(args.session_id))
    if getattr(args, "dry_run", False):
        sessions = logger.list_sessions()
        session_meta = next((s for s in sessions if s.session_id == args.session_id), None)
        if session_meta:
            sys.stdout.write(
                f"  would delete {session_meta.session_id}"
                f"  started_at={session_meta.started_at}"
                f"  harness_version={session_meta.harness_version}\n"
            )
        sys.stdout.write(
            f"Dry run: would delete {count} event(s) from session {args.session_id}.\n"
        )
        return 0
    logger.delete_session(args.session_id)
    sys.stdout.write(f"Deleted session {args.session_id}: {count} event(s) removed.\n")
    return 0


def _prune(args: argparse.Namespace) -> int:
    """Implement ``prune`` (issue #275).

    Two modes:
    * ``--keep-last N`` — retain only the N most recent sessions (by
      ``started_at``) and remove the rest.
    * ``--older-than DAYS`` — remove sessions whose ``started_at`` is
      older than the given number of days.

    ``--dry-run`` reports what *would* be removed without touching the
    store. Both modes work on sqlite and jsonl backends. Exits 0 on
    success, 1 on argument errors (neither / both flags given, or DAYS
    not a positive integer).

    ``--vacuum`` (sqlite only, issue #896) runs ``VACUUM`` plus
    ``PRAGMA wal_checkpoint(TRUNCATE)`` after the DELETEs so freed pages
    and WAL frames are returned to the filesystem; otherwise the
    ``-wal`` sidecar grows unboundedly across pruning cycles. The flag
    is a no-op on the jsonl backend.
    """
    logger = _logger_for(args.db)
    sessions = list(logger.list_sessions())

    if args.keep_last is not None and args.older_than is not None:
        sys.stderr.write("prune: use --keep-last OR --older-than, not both.\n")
        return 1
    if args.keep_last is None and args.older_than is None:
        sys.stderr.write("prune: specify --keep-last or --older-than.\n")
        return 1

    if args.keep_last is not None:
        if args.keep_last < 0:
            sys.stderr.write("prune: --keep-last must be >= 0.\n")
            return 1
        to_delete = sessions[: max(0, len(sessions) - args.keep_last)]
    else:
        assert args.older_than is not None
        if args.older_than <= 0:
            sys.stderr.write("prune: --older-than must be a positive integer.\n")
            return 1
        cutoff = datetime.now(UTC) - timedelta(days=args.older_than)
        to_delete = [s for s in sessions if datetime.fromisoformat(s.started_at) < cutoff]

    if not to_delete:
        sys.stdout.write("Nothing to prune.\n")
        return 0

    count = len(to_delete)
    if args.dry_run:
        for session in to_delete:
            sys.stdout.write(
                f"  would delete {session.session_id}  started_at={session.started_at}\n"
            )
        sys.stdout.write(f"Dry run: {count} session(s) would be deleted.\n")
        return 0

    logger.prune_sessions(
        [session.session_id for session in to_delete],
        vacuum=getattr(args, "vacuum", False),
    )
    sys.stdout.write(f"Deleted {count} session(s).\n")
    if getattr(args, "vacuum", False) and logger.backend == "sqlite":
        sys.stdout.write("Vacuumed SQLite database: WAL space reclaimed.\n")
    return 0


def _compact(args: argparse.Namespace) -> int:
    """Implement ``compact`` (issue #632).

    Rewrites the JSONL file removing orphaned ``session_end`` markers — those
    whose ``session_id`` has no corresponding ``session_start`` marker.
    This can happen after a session is deleted and a stale ``session_end``
    marker is written afterward.

    ``--dry-run`` reports what would be removed without touching the file.
    Only works on the JSONL backend; exits 0 on sqlite with a no-op message.
    """
    logger = _logger_for(args.db)

    if logger.backend != "jsonl":
        sys.stdout.write("compact: jsonl backend required; sqlite VACUUM is automatic.\n")
        return 0

    if not logger.path.exists():
        sys.stdout.write("No orphaned markers found.\n")
        return 0

    kept: list[str] = []
    seen_starts: set[str] = set()
    orphaned: list[str] = []

    with logger.path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                kept.append(line)
                continue

            kind = record.get("kind")
            session_id = record.get("session_id")

            if kind == "session_start":
                if session_id not in seen_starts:
                    seen_starts.add(session_id)
                kept.append(line)
            elif kind == "session_end":
                if session_id not in seen_starts:
                    orphaned.append(line)
                else:
                    kept.append(line)
            else:
                kept.append(line)

    orphaned_count = len(orphaned)

    if args.dry_run:
        if orphaned:
            for line in orphaned:
                sys.stdout.write(f"  would remove: {line.strip()}\n")
        sys.stdout.write(f"Dry run: {orphaned_count} orphaned marker(s) would be removed.\n")
        return 0

    if orphaned_count == 0:
        sys.stdout.write("No orphaned markers found.\n")
        return 0

    with logger.path.open("w", encoding="utf-8") as fh:
        fh.writelines(kept)
    sys.stdout.write(f"Removed {orphaned_count} orphaned marker(s).\n")
    return 0


# --- Issue #195: seed-sample-trace -------------------------------------------
# Plant a deterministic, secret-free session so the KPI / regression-report
# CLIs have something to chew on without standing up llama-server. Mirrors
# the event vocabulary the runner actually emits (issue #89, #91), so the
# Digester sees a faithful shape.


# Hard-coded payload constants. Every value is a literal placeholder — no
# real tokens, PEM blocks, or other secret-shaped substrings (docs/SECURITY.md
# §Secrets). Issue #195 acceptance criterion: "planted payloads contain no
# real secrets".
_SEED_PROMPT = "Refactor the auth helper to drop the legacy token cache."
_SEED_TOOL_NAME = "read_file"
_SEED_TOOL_ARGUMENTS: dict[str, str] = {"path": "src/foundry_x/auth.py"}
_SEED_TOOL_OUTPUT = "def authenticate(user, password):\n    return False\n"
_SEED_TOOL_CALL_ID = "call-seed-0001"
_SEED_DURATION_MS = 12
_SEED_MODEL_ID = "seeded-llama-sample"
# Issue #271: deterministic token-usage figures planted on the
# ``model_response`` event so the KPI summary and timeline CLIs have token
# data to surface without a live llama-server. All literal placeholders.
_SEED_USAGE = {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}
# Default harness version used when ``--harness-version`` is omitted. Picked
# to be obviously a synthetic seed (``seed-sample``) so real-run sessions
# don't collide with the planted one in regression reports.
_SEED_DEFAULT_HARNESS_VERSION = "seed-sample"
# Backend inferred from the ``--db`` suffix, mirroring ``observability/cli.py``.
_SQLITE_SUFFIX = ".db"


def _seed_sample_trace(args: argparse.Namespace) -> int:
    """Implement ``seed-sample-trace`` (issue #195).

    Plants one session in the trace store at ``args.db`` containing every
    event kind the :mod:`foundry_x.execution.runner` emits, so the Digester,
    KPI, and regression-report CLIs have realistic input without a live
    ``llama-server``. Returns 0 on success; emits the planted ``session_id``
    on stdout so a follow-up ``session-show`` or ``render-failure`` can pick
    it up. All planted payloads are literal placeholder text — never
    tokens, keys, or PEM blocks — per the redaction contract in
    ``docs/SECURITY.md`` §Secrets.
    """
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backend = "jsonl" if db_path.suffix.lower() != _SQLITE_SUFFIX else "sqlite"
    logger = TraceLogger(db_path, backend=backend)
    harness_version = args.harness_version

    with logger.session(
        harness_version=harness_version,
        model_id=_SEED_MODEL_ID,
        metadata={"seed": "seed-sample-trace", "issue": 195},
    ) as session_id:
        logger.record(
            session_id,
            kind="task_received",
            payload={"prompt": _SEED_PROMPT},
        )
        logger.record(
            session_id,
            kind="user_prompt",
            payload={"content": _SEED_PROMPT, "tool_count": 1},
        )
        logger.record(
            session_id,
            kind="model_request",
            payload={"step": 0, "message_count": 1, "tool_count": 1},
        )
        logger.record(
            session_id,
            kind="model_response",
            payload={
                "step": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Reading auth.py first to understand the legacy cache.",
                },
                "tool_calls": [
                    {
                        "id": _SEED_TOOL_CALL_ID,
                        "function": {
                            "name": _SEED_TOOL_NAME,
                            "arguments": json.dumps(_SEED_TOOL_ARGUMENTS),
                        },
                    }
                ],
                # Issue #271: token-usage accounting so the KPI summary and
                # timeline renderers have a token surface to exercise.
                "usage": _SEED_USAGE,
                "tokens_used": _SEED_USAGE["total_tokens"],
            },
        )
        logger.record(
            session_id,
            kind="tool_call",
            payload={
                "step": 0,
                "call_id": _SEED_TOOL_CALL_ID,
                "name": _SEED_TOOL_NAME,
                "arguments": _SEED_TOOL_ARGUMENTS,
                "duration_ms": _SEED_DURATION_MS,
            },
        )
        logger.record(
            session_id,
            kind="tool_result",
            payload={
                "step": 0,
                "call_id": _SEED_TOOL_CALL_ID,
                "name": _SEED_TOOL_NAME,
                "duration_ms": _SEED_DURATION_MS,
                "output": _SEED_TOOL_OUTPUT,
                "error": None,
            },
        )
        logger.record(
            session_id,
            kind="outcome",
            payload={"status": "success", "reason": "final_answer", "steps": 1},
        )

    sys.stdout.write(f"seeded session_id={session_id}\n")
    return 0


def _info(args: argparse.Namespace) -> int:
    """Implement ``info`` (issue #959).

    Prints WAL size, DB size, and session count for operators to detect
    WAL bloat before it becomes problematic.
    """
    logger = _logger_for(args.db)
    sessions = list(logger.list_sessions())

    if logger.backend == "sqlite":
        db_path = Path(args.db)
        wal_path = db_path.with_suffix(db_path.suffix + "-wal")
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        db_size = db_path.stat().st_size if db_path.exists() else 0
        session_count = len(sessions)

        sys.stdout.write("Backend: sqlite\n")
        sys.stdout.write(f"DB size: {db_size} bytes\n")
        sys.stdout.write(f"WAL size: {wal_size} bytes\n")
        sys.stdout.write(f"Sessions: {session_count}\n")

        if wal_size > 100 * 1024 * 1024:
            sys.stderr.write(
                f"WARNING: WAL size ({wal_size} bytes) exceeds 100 MB threshold. "
                f"Run `foundry-trace prune --vacuum` to reclaim WAL space.\n"
            )
    else:
        db_path = Path(args.db)
        db_size = db_path.stat().st_size if db_path.exists() else 0
        session_count = len(sessions)

        sys.stdout.write("Backend: jsonl\n")
        sys.stdout.write(f"File size: {db_size} bytes\n")
        sys.stdout.write(f"Sessions: {session_count}\n")

    return 0


# --- Issue #1044: diagnose — guided failure-mode triage ---------------------
# Implements the six-row "Common failure modes" table from
# docs/ARCHITECTURE.md §Common failure modes as a single automated pass.
# One row per failure mode; columns: Failure | Detected | Evidence. The
# table here and ARCHITECTURE.md stay in sync (acceptance criterion):
# adding a failure mode to ARCHITECTURE.md requires adding it here too.

# Max length of a payload JSON excerpt embedded in the Evidence column.
_DIAGNOSE_EVIDENCE_LIMIT = 160


def _payload_json(event: TraceEvent) -> str:
    """Sorted single-line JSON of *event*'s payload, truncated for the table."""
    text = json.dumps(event.payload, sort_keys=True)
    if len(text) > _DIAGNOSE_EVIDENCE_LIMIT:
        return text[: _DIAGNOSE_EVIDENCE_LIMIT - 1] + "\u2026"
    return text


def _payload_contains(event: TraceEvent, needle: str) -> bool:
    """True when *needle* appears in the serialized payload JSON (case-sensitive)."""
    return needle in json.dumps(event.payload, sort_keys=True)


def _check_no_proposed_edit(events: Sequence[TraceEvent]) -> tuple[bool, str]:
    """Row 1 — Evolver produces no ProposedEdit (ARCHITECTURE.md:136).

    Signature: session ends after ``task_completed`` with no
    ``critic_verdict`` event following it.
    """
    task_completed = [e for e in events if e.kind == "task_completed"]
    critic_verdicts = [e for e in events if e.kind == "critic_verdict"]
    if task_completed and not critic_verdicts:
        return True, "task_completed present; no critic_verdict event follows."
    if not task_completed:
        return False, "no task_completed event."
    return False, f"{len(critic_verdicts)} critic_verdict event(s)."


def _check_critic_timeout(events: Sequence[TraceEvent]) -> tuple[bool, str]:
    """Row 2 — Critic hangs / timeout (ARCHITECTURE.md:137).

    Signature: ``task_aborted`` with ``reason == "wall_clock"`` (or
    ``token_budget``). No ``critic_verdict`` follows.
    """
    aborted = [e for e in events if e.kind == "task_aborted"]
    if not aborted:
        return False, "no task_aborted event."
    ev = aborted[0]
    reason = ev.payload.get("reason", "?")
    return True, f"task_aborted reason={reason} ({_payload_json(ev)})"


def _check_oom(events: Sequence[TraceEvent]) -> tuple[bool, str]:
    """Row 3 — Runner OOM (ARCHITECTURE.md:138).

    Signature: ``task_failed`` / ``model_error`` whose payload carries
    ``MemoryError``. The confirmation grep is ``MemoryError``.
    """
    oom_events = [e for e in events if _payload_contains(e, "MemoryError")]
    if not oom_events:
        return False, "no MemoryError signal in any event payload."
    ev = oom_events[0]
    return True, f"{ev.kind}: {_payload_json(ev)}"


def _check_no_tool_calls(events: Sequence[TraceEvent]) -> tuple[bool, str]:
    """Row 4 — No tool calls emitted / tool surface missing (ARCHITECTURE.md:139).

    Signature: every ``model_response`` carries ``tool_calls: []``.
    """
    responses = [e for e in events if e.kind == "model_response"]
    if not responses:
        return False, "no model_response events."
    all_empty = all(not e.payload.get("tool_calls") for e in responses)
    if all_empty:
        return True, (f"{len(responses)} model_response(s), all with empty tool_calls.")
    return False, "at least one model_response has tool_calls."


def _check_hook_registry_error(
    events: Sequence[TraceEvent],
) -> tuple[bool, str]:
    """Row 5 — Hook registry error (ARCHITECTURE.md:140).

    Signature: ``hook_registry_error`` with non-null ``error_type``.
    Session continues in degraded mode with all hooks disabled.
    """
    hook_errors = [e for e in events if e.kind == "hook_registry_error"]
    if not hook_errors:
        return False, "no hook_registry_error event."
    ev = hook_errors[0]
    return True, f"hook_registry_error ({_payload_json(ev)})"


def _check_injection_attempt(events: Sequence[TraceEvent]) -> tuple[bool, str]:
    """Row 6 — Injection attempt (ARCHITECTURE.md:141).

    Signature: ``injection_blocked`` events with markers/tool/preview.
    Multiple blocks indicate an active adversarial attempt.
    """
    blocked = [e for e in events if e.kind == "injection_blocked"]
    if not blocked:
        return False, "no injection_blocked event."
    ev = blocked[0]
    return True, f"{len(blocked)} block(s); first: {_payload_json(ev)}"


# Ordered to match ARCHITECTURE.md row order so the rendered table is a
# faithful projection of the documentation.
_DIAGNOSE_CHECKS: tuple[tuple[str, Any], ...] = (
    ("Evolver produces no ProposedEdit", _check_no_proposed_edit),
    ("Critic hangs / timeout", _check_critic_timeout),
    ("Runner OOM", _check_oom),
    ("No tool calls emitted (tool surface missing)", _check_no_tool_calls),
    ("Hook registry error", _check_hook_registry_error),
    ("Injection attempt", _check_injection_attempt),
)


def build_diagnose_report(
    session_id: str,
    events: Sequence[TraceEvent],
) -> str:
    """Render the six-row failure-mode triage table as Markdown (issue #1044).

    Pure function: takes a session_id and an ordered event sequence and
    returns a single Markdown string. Each of the six
    ``ARCHITECTURE.md`` failure modes is one table row with columns
    ``Failure | Detected | Evidence``. A footer cross-references the
    Digester class taxonomy (ADR-0011) and the source-of-truth table in
    ``ARCHITECTURE.md`` so the reader can confirm the two are in sync.
    """
    lines = [f"# Diagnose: session `{session_id}`", ""]
    lines.append("| Failure | Detected | Evidence |")
    lines.append("|---|---|---|")
    for label, check in _DIAGNOSE_CHECKS:
        detected, evidence = check(events)
        mark = "yes" if detected else "no"
        lines.append(f"| {label} | {mark} | {evidence} |")
    lines.append("")
    lines.append(
        "_Six failure modes per docs/ARCHITECTURE.md §Common failure modes; "
        "class taxonomy per [ADR-0011](../../docs/adr/0011-failure-report-class-taxonomy.md)._"
    )
    return "\n".join(lines) + "\n"


def _diagnose(args: argparse.Namespace) -> int:
    """Implement ``diagnose`` (issue #1044).

    Loads a session, runs all six failure-mode checks, and prints the
    Markdown triage table. Unknown session_id exits 1 with a clear
    error message (acceptance criterion). Works on both sqlite and
    jsonl backends via :func:`_logger_for`.
    """
    logger = _logger_for(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"session {args.session_id} not found or empty.\n")
        return 1
    report = build_diagnose_report(args.session_id, events)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    return 0


# --- Issue #1036: graphical timeline visualization ----------------------------
# Renders a session's events as a graphical ASCII timeline with duration bars,
# timing offsets, and visual markers for different event categories.


def _extract_summary(payload: dict[str, Any] | None) -> str:
    """Pull a short human-readable summary from an event payload."""
    if not payload:
        return ""
    for key in ("kind", "message", "summary", "tool", "status", "phase", "text"):
        if key in payload:
            return str(payload[key])[:80]
    return ""


def _with_token_total(summary: str, payload: dict[str, Any] | None) -> str:
    """Append token counts to a model_response summary if available."""
    if not payload:
        return summary
    usage = payload.get("usage") or payload
    in_tok = usage.get("input_tokens") or usage.get("prompt_tokens")
    out_tok = usage.get("output_tokens") or usage.get("completion_tokens")
    if in_tok and out_tok:
        return f"{summary} ({in_tok}+{out_tok} tok)".strip()
    if out_tok:
        return f"{summary} ({out_tok} tok)".strip()
    return summary


# Visual category mapping: event kinds → display labels and marker characters.
_TIMELINE_CATEGORIES: dict[str, tuple[str, str]] = {
    "session_start": ("SESSION", ">>"),
    "session_end": ("SESSION", "<<"),
    "task_received": ("TASK", "->"),
    "task_completed": ("TASK", "OK"),
    "task_failed": ("TASK", "!!"),
    "task_aborted": ("TASK", "XX"),
    "user_prompt": ("PROMPT", ">>"),
    "model_request": ("MODEL", "->"),
    "model_response": ("MODEL", "<-"),
    "model_error": ("MODEL", "!!"),
    "tool_call": ("TOOL", "->"),
    "tool_result": ("TOOL", "<-"),
    "outcome": ("RESULT", "OK"),
    "critic_verdict": ("VERDICT", ">>"),
    "hook_registry_error": ("HOOK", "!!"),
    "injection_blocked": ("SECURITY", "!!"),
    "context_pruned": ("CONTEXT", "~~"),
}

# Error kinds get a distinct visual marker in the timeline.
_TIMELINE_ERROR_KINDS: frozenset[str] = frozenset(
    {"model_error", "task_failed", "task_aborted", "hook_registry_error", "injection_blocked"}
)

# Bar rendering: max bar width in characters, and the scale factor (ms → chars).
_TIMELINE_BAR_MAX = 40
_TIMELINE_BAR_CHAR = "#"
_TIMELINE_GAP_CHAR = "."


def _parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp string."""
    return datetime.fromisoformat(value)


def _timeline_category(kind: str) -> tuple[str, str]:
    """Return (category_label, marker) for an event kind."""
    return _TIMELINE_CATEGORIES.get(kind, ("OTHER", "??"))


def _timeline_is_error(kind: str) -> bool:
    """True when the kind is a failure/error category."""
    return kind in _TIMELINE_ERROR_KINDS


def _format_duration_ms(ms: float) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    if ms < 60_000:
        return f"{ms / 1000:.1f}s"
    minutes = int(ms // 60_000)
    seconds = (ms % 60_000) / 1000
    return f"{minutes}m{seconds:.0f}s"


def _render_bar(duration_ms: float, max_ms: float, width: int = _TIMELINE_BAR_MAX) -> str:
    """Render a proportional ASCII bar for a duration value.

    The bar's length is proportional to *duration_ms* relative to *max_ms*.
    A minimum of 1 character is shown for non-zero durations so the
    timeline stays readable for short-lived events.
    """
    if duration_ms <= 0 or max_ms <= 0:
        return ""
    filled = max(1, round((duration_ms / max_ms) * width))
    filled = min(filled, width)
    return _TIMELINE_BAR_CHAR * filled + _TIMELINE_GAP_CHAR * (width - filled)


def build_graphical_timeline(
    events: Sequence[TraceEvent],
    *,
    kind_filter: str | None = None,
    use_color: bool = False,
) -> str:
    """Render a session's events as a graphical ASCII timeline.

    Each event produces one line with:
      - A step number (zero-padded)
      - The wall-clock offset from the first event
      - The duration from the previous event as a proportional bar
      - The event kind (left-justified)
      - A one-line summary extracted from the payload

    Events matching *kind_filter* (if provided) are shown; all others
    are included but marked as filtered. Error events get a ``!`` prefix.

    Returns the complete timeline as a multi-line string. Empty event
    sequences produce a header-only output.
    """
    if not events:
        return "(no events)\n"

    base = _parse_ts(events[0].timestamp)
    lines: list[str] = []

    # Pre-compute inter-event durations to find the max for bar scaling.
    offsets: list[float] = []
    deltas: list[float] = []
    for i, event in enumerate(events):
        delta_s = (_parse_ts(event.timestamp) - base).total_seconds()
        offsets.append(delta_s)
        if i == 0:
            deltas.append(0.0)
        else:
            deltas.append(delta_s - offsets[i - 1])

    max_delta_ms = max((d * 1000 for d in deltas), default=0)

    # Header.
    total_s = offsets[-1] if offsets else 0
    lines.append(
        f"Timeline: {len(events)} event(s), total span {_format_duration_ms(total_s * 1000)}"
    )
    lines.append("-" * 78)

    for i, event in enumerate(events):
        step = f"#{i + 1:03d}"
        offset = offsets[i]
        delta_ms = deltas[i] * 1000
        cat_label, marker = _timeline_category(event.kind)
        is_err = _timeline_is_error(event.kind)

        # Duration bar.
        bar = _render_bar(delta_ms, max_delta_ms) if i > 0 else ""

        # Summary from payload.
        summary = _extract_summary(event.payload)
        if event.kind == "model_response":
            summary = _with_token_total(summary, event.payload)

        # Build the line.
        err_prefix = "!" if is_err else " "
        bar_part = f"[{bar}]" if bar else " " * (_TIMELINE_BAR_MAX + 2)
        ts_str = f"+{offset:.3f}s"

        kind_display = event.kind.ljust(22)
        cat_display = cat_label.ljust(8)

        line = (
            f"{err_prefix}{step} {ts_str:>10s}  {_format_duration_ms(delta_ms):>8s}  "
            f"{bar_part}  {cat_display} {kind_display} {marker} {summary}"
        )
        lines.append(line.rstrip())

    lines.append("-" * 78)
    return "\n".join(lines) + "\n"


def _doctor(args: argparse.Namespace) -> int:
    """Implement ``doctor`` (issue #1077).

    Scans a JSONL trace file for irrecoverably corrupted lines
    (those that raise JSONDecodeError on parse) and — with ``--apply`` —
    rewrites the file atomically via tempfile + os.replace, dropping the
    bad lines. Without ``--apply`` this is a dry-run that reports what
    would be dropped.
    """
    from foundry_x.trace.logger import TraceLogger

    if not args.db.endswith(".jsonl"):
        sys.stderr.write("doctor: jsonl backend required; sqlite is not supported.\n")
        return 1
    logger = TraceLogger(args.db, backend="jsonl")
    result = logger.doctor(apply=args.apply)
    if not result["applied"]:
        if result["skipped_lines"]:
            sys.stdout.write(
                f"doctor: dry-run — {len(result['skipped_lines'])} line(s) would be dropped:\n"
            )
            for lineno, sid in zip(result["skipped_lines"], result["skipped_session_ids"]):
                sid_str = sid if sid else "(no session_id)"
                sys.stdout.write(f"  line {lineno}  session_id={sid_str}\n")
            sys.stdout.write(
                f"doctor: dry-run — {result['kept_lines']} valid line(s) would be kept.\n"
            )
            sys.stdout.write("Run with --apply to rewrite the file.\n")
        else:
            sys.stdout.write("doctor: no corrupt lines found.\n")
    else:
        sys.stdout.write(
            f"doctor: applied — {len(result['skipped_lines'])} line(s) dropped, "
            f"{result['kept_lines']} line(s) kept.\n"
        )
        for lineno, sid in zip(result["skipped_lines"], result["skipped_session_ids"]):
            sid_str = sid if sid else "(no session_id)"
            sys.stdout.write(f"  dropped: line {lineno}  session_id={sid_str}\n")
    return 0


def _timeline(args: argparse.Namespace) -> int:
    """Implement ``timeline`` (issue #1036).

    Renders a session's events as a graphical ASCII timeline with
    duration bars, timing offsets, and visual markers for event
    categories. Supports ``--kind`` to filter by event kind and
    ``--no-color`` to disable TTY detection (colors are not used in
    the current implementation but the flag is reserved).
    """
    logger = _logger_for(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    kind_filter = getattr(args, "kind", None)
    use_color = not getattr(args, "no_color", False)

    # Apply kind filter if specified.
    if kind_filter is not None:
        events = [e for e in events if e.kind == kind_filter]
        if not events:
            sys.stderr.write(
                f"No events matching kind '{kind_filter}' in session {args.session_id}.\n"
            )
            return 1

    output = build_graphical_timeline(events, kind_filter=kind_filter, use_color=use_color)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


# --- Issue #1123: run-benchmark — single-task benchmark CLI --------------------


def _resolve_benchmark_task(task_name: str) -> tuple[Path, str]:
    """Resolve *task_name* to a (test_file_path, test_function_name) pair.

    Accepts both bare names (``sort_a_list``) and ``test_``-prefixed names
    (``test_sort_a_list``). Returns a tuple of the resolved pytest path and
    the canonical test function name.

    Raises:
        FileNotFoundError: the task file does not exist.
        ValueError: the task file exists but the corresponding test function
            is not found inside it.
    """
    normalized = task_name
    if task_name.startswith("test_"):
        normalized = task_name[5:]
    test_func_name = f"test_{normalized}"
    test_file = Path("benchmarks/tasks") / f"test_{normalized}.py"

    if not test_file.exists():
        raise FileNotFoundError(
            f"Benchmark task not found: '{task_name}' "
            f"(tried {test_file}). "
            f"Check the task name or browse benchmarks/tasks/ for available tasks."
        )

    source = test_file.read_text(encoding="utf-8")
    if f"def {test_func_name}(" not in source:
        raise ValueError(
            f"Task '{task_name}' resolves to {test_file} but "
            f"{test_func_name}() is not defined in that file. "
            f"Verify the task file contains the expected test function."
        )

    return test_file, test_func_name


def _run_benchmark(args: argparse.Namespace) -> int:
    """Implement ``run-benchmark`` (issue #1123).

    Resolves a named benchmark task to its pytest path and invokes it with
    the correct ``-m benchmark`` marker and workspace isolation.
    Accepts ``--fixture <name>`` to seed the benchmark workspace from
    ``benchmarks/fixtures/<name>/``. Exits with the underlying pytest
    exit code.

    Task name resolution is forgiving: both ``sort_a_list`` and
    ``test_sort_a_list`` are accepted and normalised internally.
    """
    try:
        test_file, test_func_name = _resolve_benchmark_task(args.task_name)
    except FileNotFoundError as exc:
        sys.stderr.write(f"run-benchmark: {exc}\n")
        return 1
    except ValueError as exc:
        sys.stderr.write(f"run-benchmark: {exc}\n")
        return 1

    cmd: list[str] = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file) + "::" + test_func_name,
        "-m",
        "benchmark",
    ]

    env: dict[str, str] = dict(__import__("os").environ)
    if getattr(args, "fixture", None) is not None:
        env["PYTEST_BENCHMARK_FIXTURE"] = args.fixture

    import subprocess

    result = subprocess.run(
        cmd,
        env=env,
        check=False,
    )
    return result.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry-trace",
        description="Inspect and render trace data (ADR-0007).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render_parser = sub.add_parser(
        "render-failure",
        help="Render a Digester FailureReport as Markdown.",
    )
    render_parser.add_argument("session_id", help="Trace session to digest.")
    render_parser.add_argument(
        "--trace-path",
        default="logs/traces.db",
        help="Path to the trace SQLite database.",
    )
    render_parser.add_argument(
        "--out",
        default=None,
        help="Write Markdown to this path instead of stdout.",
    )
    render_parser.set_defaults(func=_render_failure)

    sessions_parser = sub.add_parser(
        "sessions",
        help="List recorded trace sessions.",
    )
    sessions_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    sessions_parser.set_defaults(func=_sessions)

    show_parser = sub.add_parser(
        "show",
        help="Print the ordered events of one session.",
    )
    show_parser.add_argument("session_id", help="Session to display.")
    show_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    show_parser.set_defaults(func=_show)

    export_parser = sub.add_parser(
        "export",
        help="Export a session as newline-delimited JSON (ADR-0003 JSONL).",
    )
    export_parser.add_argument("session_id", help="Session to export.")
    export_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    export_parser.add_argument(
        "--out",
        default=None,
        help="Write JSONL to this path instead of stdout.",
    )
    export_parser.set_defaults(func=_export)

    # Issue #83 subcommands: session-list / session-show / events-grep.
    session_list_parser = sub.add_parser(
        "session-list",
        help="List trace sessions (session_id, started_at, ended_at, harness_version).",
    )
    session_list_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    session_list_parser.add_argument(
        "--harness-version",
        default=None,
        help="Filter to sessions recorded with this harness version.",
    )
    session_list_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Print at most N sessions after filtering.",
    )
    session_list_parser.set_defaults(func=_session_list)

    session_show_parser = sub.add_parser(
        "session-show",
        help="Print every event of a session via the timeline renderer.",
    )
    session_show_parser.add_argument("session_id", help="Session to display.")
    session_show_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    session_show_parser.set_defaults(func=_session_show)

    events_grep_parser = sub.add_parser(
        "events-grep",
        help="Print events whose payload JSON matches a regex.",
    )
    events_grep_parser.add_argument("session_id", help="Session to scan.")
    events_grep_parser.add_argument(
        "--pattern",
        required=True,
        help="Python regex applied to each event's serialized payload.",
    )
    events_grep_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    events_grep_parser.set_defaults(func=_events_grep)

    # Issue #192 subcommands: redact-session / redact-key.
    redact_session_parser = sub.add_parser(
        "redact-session",
        help="Delete a session and all its events (SECURITY.md \u00a7Secrets).",
    )
    redact_session_parser.add_argument("session_id", help="Session to delete.")
    redact_session_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    redact_session_parser.add_argument(
        "--out",
        default=None,
        help="Append a JSONL audit-log record to this path.",
    )
    redact_session_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be redacted without modifying the store (issue #1122).",
    )
    redact_session_parser.set_defaults(func=_redact_session)

    redact_key_parser = sub.add_parser(
        "redact-key",
        help="Overwrite a single payload field with [REDACTED] (SECURITY.md \u00a7Secrets).",
    )
    redact_key_parser.add_argument("session_id", help="Session containing the event.")
    redact_key_parser.add_argument(
        "event_index",
        type=int,
        help="Zero-based index of the event in timestamp order.",
    )
    redact_key_parser.add_argument("key", help="Payload key to overwrite.")
    redact_key_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    redact_key_parser.add_argument(
        "--out",
        default=None,
        help="Append a JSONL audit-log record to this path.",
    )
    redact_key_parser.set_defaults(func=_redact_key)

    # Issue #275: delete-session / prune — retention management.
    delete_session_parser = sub.add_parser(
        "delete-session",
        help="Remove a session and all its events (idempotent).",
    )
    delete_session_parser.add_argument("session_id", help="Session to delete.")
    delete_session_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    delete_session_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without modifying the store (issue #1122).",
    )
    delete_session_parser.set_defaults(func=_delete_session)

    # Issue #195: offline smoke subcommand. Plants a deterministic session
    # so the Digester, KPI, and regression-report CLIs have realistic input
    # without standing up llama-server. ``--harness-version`` lets the
    # caller simulate a candidate harness version for the compare-kpis
    # baseline-vs-candidate path (issue #100).
    seed_parser = sub.add_parser(
        "seed-sample-trace",
        help="Plant a deterministic sample session for offline smoke testing.",
    )
    seed_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    seed_parser.add_argument(
        "--harness-version",
        default=_SEED_DEFAULT_HARNESS_VERSION,
        help=(
            "Harness version to stamp on the seeded session "
            "(default: %(default)s). Set to a candidate version (e.g. "
            "1.0.0) so compare-kpis can treat the seed as the candidate."
        ),
    )
    seed_parser.set_defaults(func=_seed_sample_trace)

    prune_parser = sub.add_parser(
        "prune",
        help="Remove old sessions per retention policy (--keep-last or --older-than).",
    )
    prune_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    prune_parser.add_argument(
        "--keep-last",
        type=int,
        metavar="N",
        help="Keep only the N most recent sessions (by started_at).",
    )
    prune_parser.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Remove sessions older than DAYS days (by started_at).",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sessions that would be removed without deleting them.",
    )
    prune_parser.add_argument(
        "--vacuum",
        action="store_true",
        help=(
            "After pruning, run VACUUM and wal_checkpoint(TRUNCATE) on the "
            "SQLite database to reclaim WAL/free space (issue #896). "
            "Recommended when the trace store is not being written to, e.g. "
            "after a batch prune of old sessions. Without this, the WAL "
            "sidecar grows unboundedly and can reach several GB. "
            "No-op on the JSONL backend."
        ),
    )
    prune_parser.set_defaults(func=_prune)

    info_parser = sub.add_parser(
        "info",
        help="Show WAL size, DB size, and session count for the trace store (issue #959).",
    )
    info_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    info_parser.set_defaults(func=_info)

    compact_parser = sub.add_parser(
        "compact",
        help="Remove orphaned session_end markers from a JSONL trace file (issue #632).",
    )
    compact_parser.add_argument(
        "--db",
        default="logs/traces.jsonl",
        help="Path to the JSONL trace file (default: logs/traces.jsonl).",
    )
    compact_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orphaned markers without modifying the file.",
    )
    compact_parser.set_defaults(func=_compact)

    # Issue #1077: repair a JSONL trace file by dropping irrecoverably
    # corrupted lines. Dry-run by default; --apply to rewrite atomically.
    doctor_parser = sub.add_parser(
        "doctor",
        help="Drop irrecoverably corrupted JSONL lines (json_decode_error) from a trace file.",
    )
    doctor_parser.add_argument(
        "--db",
        required=True,
        help="Path to the JSONL trace file.",
    )
    doctor_parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help=(
            "Rewrite the file, dropping corrupt lines. Without this flag "
            "the command is a dry-run that reports what would be dropped."
        ),
    )
    doctor_parser.set_defaults(func=_doctor)

    # Issue #1044: guided failure-mode triage. Runs all six
    # ARCHITECTURE.md failure-mode checks in one pass and prints a
    # Markdown table (Failure | Detected | Evidence).
    diagnose_parser = sub.add_parser(
        "diagnose",
        help="Run all six ARCHITECTURE.md failure-mode checks (issue #1044).",
    )
    diagnose_parser.add_argument("session_id", help="Session to diagnose.")
    diagnose_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    diagnose_parser.add_argument(
        "--out",
        default=None,
        help="Write the report to this path instead of stdout.",
    )
    diagnose_parser.set_defaults(func=_diagnose)

    # --- timeline (issue #1036) ---
    timeline_parser = sub.add_parser(
        "timeline",
        help="Render a graphical ASCII timeline of a session's events with duration bars.",
    )
    timeline_parser.add_argument(
        "session_id",
        help="The session ID to render a timeline for.",
    )
    timeline_parser.add_argument(
        "--kind",
        default=None,
        help="Filter to a single event kind (e.g. 'tool_call', 'model_request').",
    )
    timeline_parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable TTY color detection (reserved for future use).",
    )
    timeline_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    timeline_parser.add_argument(
        "--out",
        default=None,
        help="Write the timeline to this path instead of stdout.",
    )
    timeline_parser.set_defaults(func=_timeline)

    # Issue #1123: single-task benchmark execution. Resolves a named benchmark
    # task to its pytest path and invokes it with correct isolation and the
    # benchmark marker. ``--fixture`` seeds the workspace from the named
    # fixture directory. Exit code mirrors pytest's exit code.
    run_benchmark_parser = sub.add_parser(
        "run-benchmark",
        help="Run a single benchmark task in isolation (issue #1123).",
    )
    run_benchmark_parser.add_argument(
        "task_name",
        help=(
            "Benchmark task name (e.g. sort_a_list or test_sort_a_list). "
            "The 'test_' prefix is optional and stripped automatically."
        ),
    )
    run_benchmark_parser.add_argument(
        "--fixture",
        metavar="NAME",
        default=None,
        help=(
            "Seed the benchmark workspace from benchmarks/fixtures/NAME/ "
            "before running the task. If omitted the workspace starts empty."
        ),
    )
    run_benchmark_parser.set_defaults(func=_run_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
