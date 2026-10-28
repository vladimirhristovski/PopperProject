import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFINITIONS = ["popper_qwen.def", "popper_modern.def"]
PROBE = (
    "from popper_upgrade.serving import rewrite\n"
    f"tokenizer = rewrite._tokenizer({MODEL!r}, {REVISION!r})\n"
    "assert tokenizer is not None, 'the pinned tokenizer was not found in the Hugging Face cache'\n"
    "print(len(tokenizer.encode('hello world', add_special_tokens=False)))\n"
)


def _environment_block(name):
    text = (ROOT / "cluster" / name).read_text(encoding="utf-8")
    return text.split("%environment", 1)[1].split("%runscript", 1)[0]


def _hf_home():
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


@pytest.mark.parametrize("name", DEFINITIONS)
def test_definitions_keep_the_transformers_cache_at_the_hub_default(name):
    block = _environment_block(name)

    assert "TRANSFORMERS_CACHE" not in block
    assert "export HF_HOME=" in block


@pytest.mark.parametrize("name", DEFINITIONS)
def test_definition_environment_finds_the_pinned_tokenizer_in_the_hub_cache(name, tmp_path):
    pytest.importorskip("transformers")
    home = _hf_home()
    if not (home / "hub" / f"models--{MODEL.replace('/', '--')}" / "snapshots" / REVISION / "tokenizer.json").exists():
        pytest.skip("the pinned tokenizer is not in the local Hugging Face cache")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "PYTHONPATH")
    }
    environment.update(POPPER_HF_HOME=str(home), PYTHONPATH=str(ROOT), PROBE=PROBE, PROBE_PYTHON=sys.executable)

    result = subprocess.run(
        ["sh", "-c", _environment_block(name) + '\nexec "$PROBE_PYTHON" -c "$PROBE"\n'],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert int(result.stdout.strip().splitlines()[-1]) > 0
