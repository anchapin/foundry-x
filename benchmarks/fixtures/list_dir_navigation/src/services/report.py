"""Report generation helpers."""


def line(label: str, value: str) -> str:
    """Format a ``label=value`` report line."""
    return f"{label}={value}"
