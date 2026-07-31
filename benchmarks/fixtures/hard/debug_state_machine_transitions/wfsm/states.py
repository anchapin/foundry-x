"""State definitions for debug_state_machine_transitions fixture."""

from enum import Enum


class State(Enum):
    """Workflow lifecycle states."""

    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
