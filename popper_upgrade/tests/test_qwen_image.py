import re
from pathlib import Path

import pytest

from cluster import preflight

ROOT = Path(__file__).resolve().parents[2]
DEFINITION = (ROOT / "cluster" / "popper_qwen.def").read_text(encoding="utf-8")
DOCKERFILE = (ROOT / "popper_upgrade" / "Dockerfile.qwen").read_text(encoding="utf-8")
POPPER_SHA = "59611864e09dd15d7528e240ad87c7ddbb4a701c"
SHARED = [
    "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
    "uv==0.13.5",
    "uv venv --python 3.12 /opt/vllm-env",
    "uv pip install --python /opt/vllm-env/bin/python --torch-backend=cu128 vllm==0.19.1 torch==2.10.0",
    "uv venv --python 3.10 /opt/popper-env",
    "torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu",
    f"--no-deps git+https://github.com/snap-stanford/POPPER.git@{POPPER_SHA}",
    "-r /opt/popper/popper_upgrade/requirements.txt",
    "cuda-nvcc-12-8",
]


@pytest.mark.parametrize("text", [DEFINITION, DOCKERFILE], ids=["definition", "dockerfile"])
@pytest.mark.parametrize("fragment", SHARED)
def test_both_images_install_the_same_stack(text, fragment):
    assert fragment in text


def test_definition_selects_the_qwen_driver_floor_and_the_vllm_python():
    assert "export POPPER_IMAGE=qwen" in DEFINITION
    assert "POPPER_IMAGE=qwen" in DOCKERFILE
    assert "qwen" in preflight.DRIVER_FLOORS
    for text in (DEFINITION, DOCKERFILE):
        assert "POPPER_VLLM_PYTHON=/opt/vllm-env/bin/python" in text


def test_definition_bakes_only_the_qwen_large_job():
    assert "! -name qwen-large -exec rm -rf {} +" in DEFINITION


def test_definition_self_test_checks_the_installed_versions():
    pinned = re.search(r"vllm==([\d.]+)", DEFINITION).group(1)
    test_section = DEFINITION.split("%test", 1)[1]

    assert f"vllm.__version__ == '{pinned}'" in test_section
    assert "torch.version.cuda == '12.8'" in test_section
