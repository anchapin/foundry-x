"""Minimal Flask-style app used by the multi_file_basic benchmark case."""

from __future__ import annotations

from flask import Flask, jsonify

app = Flask(__name__)


@app.get("/health")
def health() -> tuple[dict[str, str], int]:
    """Liveness probe; returns a tiny JSON document."""
    return jsonify(status="ok"), 200


@app.get("/version")
def version() -> dict[str, str]:
    """Return the running build version for diagnostics."""
    return jsonify(version="1.0.0")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
