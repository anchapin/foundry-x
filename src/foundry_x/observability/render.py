from __future__ import annotations

from foundry_x.evolution.digester import FailureReport

_STEP_COLUMNS = ("step", "kind", "detail")


def render_failure_report(report: FailureReport) -> str:
    lines: list[str] = [
        f"# Failure Report \u2014 session `{report.session_id}`",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Suspected Causes",
        "",
    ]

    if report.suspected_causes:
        for index, cause in enumerate(report.suspected_causes, 1):
            lines.append(f"{index}. {cause}")
    else:
        lines.append("_None identified._")

    lines += ["", "## Failed Steps", ""]
    header = " | ".join(_STEP_COLUMNS)
    separator = " | ".join("---" for _ in _STEP_COLUMNS)
    lines += [f"| {header} |", f"| {separator} |"]

    if report.failed_steps:
        for step in report.failed_steps:
            cells = " | ".join(str(step.get(col, "")) for col in _STEP_COLUMNS)
            lines.append(f"| {cells} |")
    else:
        lines.append("| _none_ | \u2014 | \u2014 |")

    lines += ["", f"## Classification: {report.proposed_class}", ""]
    return "\n".join(lines)


def render_failure_report_json(report: FailureReport) -> str:
    return report.model_dump_json(indent=2) + "\n"
