"""Action handlers for state transitions (seeded).

Each action fires when the workflow enters the corresponding state.
``enter_running`` records the ``duration`` attribute on the registry.
When the ``RUNNING`` state is bypassed (due to the ``PENDING -> DONE``
bug in ``wfsm/machine.py``), ``duration`` is never set.
"""

from wfsm.states import State


class ActionRegistry:
    """Registry of action handlers keyed by state.

    Carries the ``duration`` attribute that ``wfrun/reporter.py`` reads.
    """

    def __init__(self) -> None:
        self._handlers: dict[State, list] = {}
        self.duration: float | None = None

    def register(self, state: State, handler) -> None:
        """Register *handler* to fire when the workflow enters *state*."""
        self._handlers.setdefault(state, []).append(handler)

    def fire(self, state: State) -> None:
        """Fire all handlers registered for *state*."""
        for handler in self._handlers.get(state, []):
            handler(self)


def enter_pending(registry: ActionRegistry) -> None:
    """Reset duration when entering PENDING."""
    registry.duration = None


def enter_running(registry: ActionRegistry) -> None:
    """Record start — sets ``duration`` on the registry.

    This is the handler that the ``PENDING -> DONE`` bug bypasses,
    leaving ``duration`` as ``None``.
    """
    registry.duration = 0.0


def enter_done(registry: ActionRegistry) -> None:
    """Finalise the workflow."""
    if registry.duration is not None:
        registry.duration = max(registry.duration, 0.0)
