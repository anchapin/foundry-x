#!/usr/bin/env python3
"""Validate and run with config.json.

Exits 0 if the config is valid and port is a positive integer.
Exits 1 if config is missing, invalid JSON, or port is not a positive integer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found", file=sys.stderr)
        return 1

    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {e}", file=sys.stderr)
        return 1

    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port")

    if not isinstance(port, int):
        print(f"ERROR: port must be an integer, got {type(port).__name__}: {port!r}", file=sys.stderr)
        return 1

    if port <= 0 or port > 65535:
        print(f"ERROR: port must be 1-65535, got {port}", file=sys.stderr)
        return 1

    print(f"Config valid: server will listen on {host}:{port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
