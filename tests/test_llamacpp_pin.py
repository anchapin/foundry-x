from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROCM_SETUP = REPO_ROOT / "infra" / "llama-cpp" / "rocm_setup.sh"
LLAMACPP_README = REPO_ROOT / "infra" / "llama-cpp" / "README.md"


def test_rocm_setup_pins_llamacpp_release_ref() -> None:
    script = ROCM_SETUP.read_text(encoding="utf-8")

    match = re.search(r'^LLAMACPP_REF="\$\{LLAMACPP_REF:-(b\d+)\}"$', script, re.MULTILINE)

    assert match is not None, "rocm_setup.sh must define a default llama.cpp release tag"
    assert 'git -C "$LLAMACPP_DIR" fetch --depth 1 origin "$LLAMACPP_REF"' in script
    assert 'git -C "$LLAMACPP_DIR" checkout --detach FETCH_HEAD' in script
    assert 'git clone https://github.com/ggerganov/llama.cpp "$LLAMACPP_DIR"' not in script


def test_llamacpp_readme_documents_same_pin() -> None:
    script = ROCM_SETUP.read_text(encoding="utf-8")
    readme = LLAMACPP_README.read_text(encoding="utf-8")
    tag = re.search(r'^LLAMACPP_REF="\$\{LLAMACPP_REF:-(b\d+)\}"$', script, re.MULTILINE)

    assert tag is not None
    assert f'LLAMACPP_REF="${{LLAMACPP_REF:-{tag.group(1)}}}"' in readme
    assert 'git -C llama.cpp fetch --depth 1 origin "$LLAMACPP_REF"' in readme
    assert "git -C llama.cpp checkout --detach FETCH_HEAD" in readme
