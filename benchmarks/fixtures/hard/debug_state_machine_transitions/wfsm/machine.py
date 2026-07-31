"""Transition machine for debug_state_machine_transitions (seeded/buggy).

Defines the allowed state transitions via a ``TRANSITIONS`` dict.  The
table contains a bug: ``PENDING -> DONE`` is listed as an allowed
transition, which bypasses the ``RUNNING`` state entirely.

This bypass causes a downstream failure: when a workflow takes the
``PENDING -> DONE`` shortcut, the ``enter_running`` action never fires,
so ``ActionRegistry.duration`` stays ``None``.  The symptom
(``ValueError`` from ``wfrun/reporter.py``) points to the reporter, but
the root cause is the invalid transition in THIS module's table.

The agent must trace the error from the reporter back through the
workflow driver to this transition table, identify the invalid edge,
and remove it.
"""

from wfsm.states import State

#: Allowed state transitions.  BUG: ``State.DONE`` should NOT appear in
#: the ``PENDING`` set — only ``RUNNING -> DONE`` is a valid path to
#: completion.
TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.PENDING},
    State.PENDING: {State.RUNNING, State.DONE},  # BUG: DONE bypasses RUNNING
    State.RUNNING: {State.DONE, State.FAILED},
    State.DONE: set(),
    State.FAILED: {State.IDLE},
}


def can_transition(current: State, target: State) -> bool:
    """Return True if ``current -> target`` is an allowed transition."""
    return target in TRANSITIONS.get(current, set())
