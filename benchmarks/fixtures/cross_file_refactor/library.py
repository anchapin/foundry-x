"""Text statistics helpers for summarising prose.

This is the *seeded* (broken) library module for the cross_file_refactor
benchmark.  ``summarize`` references :func:`_format_line` which has not
been implemented yet -- the agent must add it.  The docstring of
``summarize`` documents the exact output format so the helper's contract
is discoverable without guessing.
"""


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*."""
    return len(text.split())


def char_count(text: str) -> int:
    """Return the number of non-whitespace characters in *text*."""
    return sum(1 for ch in text if not ch.isspace())


def summarize(text: str) -> str:
    """Return a one-line summary of *text*.

    Format: ``"words=<N>, chars=<M>"`` where *N* is the word count and
    *M* is the non-whitespace character count.
    """
    return _format_line(word_count(text), char_count(text))
