"""Server module for refactor_across_three_files benchmark (seeded/broken).

References MAX_CHUNK_SIZE -- the NEW constant name.  Since constants.py still
defines MAX_BUFFER_SIZE (the OLD name), running main.py raises NameError
until constants.py is updated to define the new name.

This file is the *seeded* (broken) state.  The golden state is identical
to this file -- only constants.py changes.
"""

from constants import MAX_CHUNK_SIZE

SERVER_PORT = 8080


def make_server_config() -> dict[str, int]:
    """Return a server config dict using MAX_CHUNK_SIZE as the max payload."""
    return {
        "port": SERVER_PORT,
        "max_payload": MAX_CHUNK_SIZE,
    }


def main() -> None:
    """Print the server config as a one-liner."""
    cfg = make_server_config()
    print(f"port={cfg['port']}, max_payload={cfg['max_payload']}")
