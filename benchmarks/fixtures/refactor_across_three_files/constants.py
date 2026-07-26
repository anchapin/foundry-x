"""Constants definition for refactor_across_three_files benchmark (seeded/broken).

Defines MAX_BUFFER_SIZE -- the OLD constant name.  processor.py and
server.py already reference the NEW name MAX_CHUNK_SIZE, so running
main.py raises NameError until this file is updated to define the new name.

This file is the *seeded* (broken) state.  The golden rename lives in
benchmarks/tasks/test_refactor_across_three_files.py (GOLDEN_CONSTANTS).
"""

MAX_BUFFER_SIZE = 4096

DEFAULT_TIMEOUT = 30
