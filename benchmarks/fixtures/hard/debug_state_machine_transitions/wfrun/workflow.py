"""Workflow runner for debug_state_machine_transitions (seeded).

Drives a sequence of state transitions through the ``TransitionMachine``
and fires action handlers via ``ActionRegistry``.

When ``fast_path=True``, the workflow attempts to skip directly from
``PENDING`` to ``DONE``.  If the transition table (in
``wfsm/machine.py``) incorrectly allows this, the ``RUNNING`` state is
bypassed and ``enter_running`` never fires, leaving ``duration`` as
``None``.  If the table correctly rejects the shortcut, the workflow
falls back to the normal ``PENDING -> RUNNING -> DONE`` path.
"""

from wfsm.states import State
from wfsm.machine import can_transition
from wfsm.actions import (
    ActionRegistry,
    enter_pending,
    enter_running,
    enter_done,
)


class Workflow:
    """A single workflow instance driven through the state machine."""

    def __init__(self, fast_path: bool = False) -> None:
        self.current_state = State.IDLE
        self.fast_path = fast_path
        self.registry = ActionRegistry()
        self.registry.register(State.PENDING, enter_pending)
        self.registry.register(State.RUNNING, enter_running)
        self.registry.register(State.DONE, enter_done)

    def advance(self) -> State:
        """Advance to the next state using the transition table."""
        if self.current_state == State.IDLE:
            self._transition(State.PENDING)
        elif self.current_state == State.PENDING:
            if self.fast_path:
                try:
                    self._transition(State.DONE)
                    return self.current_state
                except ValueError:
                    pass  # Transition rejected — fall back to normal path
            self._transition(State.RUNNING)
        elif self.current_state == State.RUNNING:
            self._transition(State.DONE)
        return self.current_state

    def _transition(self, target: State) -> None:
        """Transition to *target* and fire the corresponding action."""
        if not can_transition(self.current_state, target):
            raise ValueError(
                f"Invalid transition: {self.current_state.value} -> {target.value}"
            )
        self.current_state = target
        self.registry.fire(target)

    def run(self) -> None:
        """Run the workflow to completion (IDLE -> ... -> DONE)."""
        while self.current_state != State.DONE:
            self.advance()
