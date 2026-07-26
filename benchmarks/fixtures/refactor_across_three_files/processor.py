"""Processor module for refactor_across_three_files benchmark (seeded/broken).

References MAX_CHUNK_SIZE -- the NEW constant name.  Since constants.py still
defines MAX_BUFFER_SIZE (the OLD name), running main.py raises NameError
until constants.py is updated to define the new name.

This file is the *seeded* (broken) state.  The golden state is identical
to this file -- only constants.py changes.
"""

from constants import MAX_CHUNK_SIZE


def process_data(data: str) -> int:
    """Return the byte size of *data* capped to MAX_CHUNK_SIZE."""
    size = len(data.encode("utf-8"))
    return min(size, MAX_CHUNK_SIZE)


def main() -> None:
    """Print the process_data result for a sample string."""
    result = process_data("hello world")
    print(f"processed_size={result}")
