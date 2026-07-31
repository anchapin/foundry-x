"""Workflow integration tests for debug_state_machine_transitions.

``test_fast_path_workflow_reports_duration`` exercises the fast-path
workflow.  When the transition table (in ``wfsm/machine.py``) incorrectly
allows ``PENDING -> DONE``, the ``RUNNING`` state is bypassed and
``report_duration`` raises ``ValueError`` because ``duration`` is ``None``.

The agent must trace the ``ValueError`` from the reporter, through the
workflow driver, back to the invalid transition in the machine, and
remove it so the fast-path workflow falls back to the normal lifecycle.
"""

from wfsm.states import State
from wfrun.workflow import Workflow
from wfrun.reporter import report_duration


def test_normal_workflow_completes() -> None:
    """A normal workflow reaches DONE with a recorded duration."""
    wf = Workflow()
    wf.run()
    assert wf.current_state == State.DONE
    assert wf.registry.duration is not None


def test_fast_path_workflow_reports_duration() -> None:
    """A fast-path workflow must still have a valid duration.

    The fast-path workflow attempts ``PENDING -> DONE`` directly.  If the
    transition table allows it, ``RUNNING`` is bypassed and
    ``report_duration`` raises ``ValueError``.  The correct behavior is
    for the table to reject the shortcut so the workflow falls back to
    ``PENDING -> RUNNING -> DONE``.
    """
    wf = Workflow(fast_path=True)
    wf.run()
    assert report_duration(wf) == "0.0s"


def test_reporter_format() -> None:
    """report_duration returns a properly formatted string."""
    wf = Workflow()
    wf.run()
    result = report_duration(wf)
    assert result.endswith("s")
