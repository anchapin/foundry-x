def repeat(text, times):
    """Repeat a string `times` times, separated by commas."""
    if times < 0:
        raise ValueError("times must be non-negative")
    if times == 0:
        return ""
    return ", ".join([text] * times)
