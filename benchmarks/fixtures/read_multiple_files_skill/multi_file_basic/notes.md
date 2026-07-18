# Benchmark Notes

This multi-file reading exercise covers three fixture files at once:

- `app.py` -- a tiny Flask app
- `data.json` -- the matching service config
- `notes.md` -- this file

The expected outcome is that all three are returned in order, none are
truncated, and the top-level `truncated` flag stays `false`. If any of those
properties break, the benchmark catches the regression.
