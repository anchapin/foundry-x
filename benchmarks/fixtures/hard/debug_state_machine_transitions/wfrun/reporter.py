"""Reporter for debug_state_machine_transitions (seeded).

Reads the ``duration`` from the workflow's ``ActionRegistry`` and formats
a report string.  When the ``RUNNING`` state was bypassed (due to the
``PENDING -> DONE`` bug), ``duration`` is ``None`` and this function
raises ``ValueError`` — the symptom that the agent observes.
"""

from wfrun.workflow import Workflow


def report_duration(wf: Workflow) -> str:
    """Return a duration string for *wf*.

    Raises:
        ValueError: if ``duration`` is ``None`` (the ``RUNNING`` state
            was never entered, indicating an invalid transition bypass).
    """
    d = wf.registry.duration
    if d is None:
        raise ValueError(
            f"Cannot report duration for workflow in state {wf.current_state.value}: "
            f"duration is None (the RUNNING phase was never entered)"
        )
    return f"{d:.1f}s"
