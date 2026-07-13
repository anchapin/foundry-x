"""Tests for the text statistics library.

This is the *seeded* test file (one passing case + one failing case).
``test_word_count`` exercises the working :func:`word_count` helper and
passes unconditionally.

``test_summarize`` exercises :func:`summarize`, which fails for **two**
independent reasons:

    1. ``summarize`` calls the missing ``_format_line`` helper, so the
       call raises ``NameError`` at runtime.
    2. Even after the helper is added, the expected string below uses the
       singular form ``"word=2, char=10"`` whereas the documented format
       (see ``summarize``'s docstring) is ``"words=2, chars=10"``.

The gold path therefore edits **both** the library (add ``_format_line``)
and this test file (correct the expected value to match the documented
format).
"""

from text_stats import char_count, summarize, word_count


def test_word_count():
    """word_count splits on whitespace (passing case)."""
    assert word_count("hello world") == 2
    assert word_count("one two three four") == 4


def test_summarize():
    """summarize produces the documented 'words=<N>, chars=<M>' format."""
    result = summarize("hello world")
    assert result == "word=2, char=10"
