from configparser import ConfigParser
from pathlib import Path


def read_file(path, max_bytes=None, max_lines=None, offset=0):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except FileNotFoundError:
        return {"content": "", "sha256": _sha256(b""), "truncated": False,
                "bytes_returned": 0, "bytes_total": 0, "error": f"file not found: {path}"}
    except PermissionError:
        return {"content": "", "sha256": _sha256(b""), "truncated": False,
                "bytes_returned": 0, "bytes_total": 0, "error": f"permission denied: {path}"}

    bytes_total = len(raw)
    if offset >= bytes_total:
        return {"content": "", "sha256": _sha256(b""), "truncated": False,
                "bytes_returned": 0, "bytes_total": bytes_total, "error": None}

    slice_ = raw[offset:]
    if max_bytes is not None and len(slice_) > max_bytes:
        slice_ = slice_[:max_bytes]
        truncated = True
    else:
        truncated = False

    content = slice_.decode("utf-8", errors="replace")
    if max_lines is not None:
        lines = content.split("\n")
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            truncated = True

    return {
        "content": content,
        "sha256": _sha256(content.encode("utf-8")),
        "truncated": truncated,
        "bytes_returned": len(content.encode("utf-8")),
        "bytes_total": bytes_total,
        "error": None,
    }


def _sha256(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def main():
    out_lines = []

    if Path("config.ini").exists():
        result = read_file("config.ini", max_bytes=65536)
        config = ConfigParser(interpolation=None)
        config.read_string(result["content"])
        for section in config.sections():
            for key, value in config.items(section):
                out_lines.append(f"{key}={value}")

    if Path("source.txt").exists():
        result = read_file("source.txt", max_bytes=65536, max_lines=20, offset=0)
        lines = result["content"].split("\n")
        out_lines.append("-- LINE 10-20 --")
        out_lines.extend(lines[9:20])

    if Path("data.txt").exists():
        result = read_file("data.txt", max_bytes=65536)
        assert result["error"] is None, f"read_file returned error: {result['error']}"
        out_lines.append("-- LATIN1 CONTENT --")
        out_lines.append(repr(result["content"]))

    Path("output.txt").write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
