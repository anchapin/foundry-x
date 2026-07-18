"""CI guard for the new-contributor tutorial (issue #897).

The tutorial at ``docs/TUTORIAL.md`` walks a new contributor through
planting a deterministic offline trace with ``foundry-x-trace
seed-sample-trace`` and inspecting it with ``foundry-x-trace show``.
Acceptance criterion #5 requires that the tutorial commands be verified
by CI.

Rather than re-implementing a parallel code path, this module invokes
the **exact CLI subcommands** the tutorial documents and asserts the
invariants the tutorial promises a reader:

1. The trace store grew (a session was persisted and is loadable) —
   criterion #2.
2. The session is inspectable via the ``show`` subcommand (the code path
   behind ``foundry-x-trace show <session_id>``) — criterion #2.
3. The session contains the key events the tutorial teaches
   (``task_received``, ``model_request``, ``tool_call``, ``outcome``) —
   criterion #3.

Everything runs offline against the bundled ``seed-sample-trace``
deterministic seed; no live model or network access is required, so the
test is safe in CI without ``OPENCODE_SERVER_URL``.
"""

from __future__ import annotations

from pathlib import Path

from foundry_x.trace.cli import main as trace_main
from foundry_x.trace.logger import TraceLogger

# The four ``kind`` values the tutorial (docs/TUTORIAL.md §4) teaches a
# new contributor to recognize. Kept in lifecycle order so a regression
# that reorders events is visible in the failure diff. This is a subset
# of the seven events ``seed-sample-trace`` plants; the tutorial groups
# ``user_prompt`` / ``model_response`` / ``tool_result`` as supporting
# detail and teaches these four as the load-bearing lifecycle markers.
_TUTORIAL_KINDS: tuple[str, ...] = (
    "task_received",
    "model_request",
    "tool_call",
    "outcome",
)


def test_tutorial_seed_and_show_produces_trace_with_key_events(tmp_path: Path) -> None:
    """The CLI path docs/TUTORIAL.md documents must produce a loadable trace.

    Runs the three commands the tutorial teaches in Steps 2–3:

    1. ``foundry-x-trace seed-sample-trace --db <db>`` — plants one
       deterministic offline session.
    2. ``foundry-x-trace sessions --db <db>`` — lists sessions (the code
       path a reader uses to copy the ``session_id``).
    3. ``foundry-x-trace show <session_id> --db <db>`` — renders the
       timeline (the code path behind the ``show`` output the tutorial
       displays).

    Then asserts the trace store grew and the session carries every
    ``kind`` the tutorial promises.
    """
    db = tmp_path / "traces.db"

    # Step 2 of the tutorial: plant a deterministic offline trace.
    # ``seed-sample-trace`` emits the planted ``session_id`` on stdout;
    # we re-derive it from ``list_sessions`` instead of parsing stdout so
    # the assertion is robust to cosmetic output changes.
    seed_rc = trace_main(["seed-sample-trace", "--db", str(db)])
    assert seed_rc == 0, f"seed-sample-trace exited {seed_rc}"

    # Step 3 (part 1) of the tutorial: list sessions. The store must have
    # grown from zero to one session — acceptance criterion #2.
    sessions_rc = trace_main(["sessions", "--db", str(db)])
    assert sessions_rc == 0, f"sessions exited {sessions_rc}"
    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    assert len(sessions) == 1, f"expected one seeded session, got {len(sessions)}"
    session_id = sessions[0].session_id

    # Step 3 (part 2) of the tutorial: show the session timeline. The
    # ``show`` subcommand is the exact code path a reader follows; a
    # non-zero exit would mean the tutorial's displayed output is not
    # reproducible.
    show_rc = trace_main(["show", session_id, "--db", str(db)])
    assert show_rc == 0, f"show exited {show_rc} for session {session_id}"

    # Acceptance criterion #3: every kind the tutorial teaches is present
    # in the persisted session.
    events = logger.load_session(session_id)
    assert len(events) > 0, "seeded session has no events"
    kinds = [event.kind for event in events]
    missing = [kind for kind in _TUTORIAL_KINDS if kind not in kinds]
    assert not missing, (
        f"tutorial-promised event kinds missing from trace: {missing}. Observed kinds: {kinds}"
    )

    # The terminal ``outcome`` event must close the session on the success
    # path — this is the shape the tutorial's Step 4 flow diagram promises.
    outcome_events = [e for e in events if e.kind == "outcome"]
    assert len(outcome_events) == 1, f"expected exactly one outcome, got {len(outcome_events)}"
    assert outcome_events[0].payload["status"] == "success", (
        f"tutorial session should close with outcome.status='success', "
        f"got {outcome_events[0].payload['status']!r}"
    )
