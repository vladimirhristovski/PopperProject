import json
from collections import namedtuple
from pathlib import Path

import pytest

from cluster import preflight
from popper_upgrade.models import ServingError
from popper_upgrade.serving.usage import UsageCounter, UsageProxy
from popper_upgrade.tests.fakes import FakeEmbedder, FakeUpstream, write_job

Usage = namedtuple("Usage", "total used free")
ANSWER = json.dumps(
    {"choices": [{"message": {"content": "OK"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 1}}
).encode()
SILENT_ANSWER = json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()


def _check(result, name):
    return next(check for check in result["checks"] if check["name"] == name)


@pytest.fixture
def healthy(monkeypatch, tmp_path):
    job = write_job(tmp_path / "job", min_disk_gb=45)
    _, popper_common = preflight.load_baseline()
    monkeypatch.setattr(preflight, "_gpu_query", lambda: ("Test GPU", 80.0))
    monkeypatch.setattr(preflight, "_driver_version", lambda: "570.10")
    monkeypatch.delenv("POPPER_IMAGE", raising=False)
    for name in preflight.TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage(10**12, 0, 100 * preflight.GB))
    monkeypatch.setattr(popper_common, "check_data", lambda: [])
    monkeypatch.setattr(preflight, "Embedder", FakeEmbedder)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    return job


def _serve(monkeypatch, body):
    upstream = FakeUpstream(body)
    started = []

    class Serving:
        def __init__(self, model):
            self.counter = UsageCounter()
            self.proxy = UsageProxy(0, None, self.counter, upstream_url=f"http://127.0.0.1:{upstream.port}")

        def __enter__(self):
            started.append(1)
            self.proxy.start()
            self.counter.port = self.proxy.port
            return self.counter

        def __exit__(self, *exc):
            self.proxy.stop()
            return False

    monkeypatch.setattr(preflight, "serving_for", Serving)
    yield started
    upstream.close()


@pytest.fixture
def answering(monkeypatch):
    yield from _serve(monkeypatch, ANSWER)


@pytest.fixture
def silent(monkeypatch):
    yield from _serve(monkeypatch, SILENT_ANSWER)


def test_all_checks_pass(healthy, answering, tmp_path):
    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert result["ok"] is True
    assert [check["name"] for check in result["checks"]] == [
        "driver",
        "gpu",
        "memory",
        "token",
        "disk",
        "data",
        "snapshot",
        "model_answers",
    ]
    assert _check(result, "gpu")["detail"] == "Test GPU, 80 GB"
    assert "7 prompt tokens" in _check(result, "model_answers")["detail"]
    assert json.loads((tmp_path / "work" / "preflight.json").read_text(encoding="utf-8")) == result


def test_no_gpu_fails_without_starting_the_model(healthy, answering, monkeypatch, tmp_path):
    def missing():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(preflight, "_gpu_query", missing)

    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert result["ok"] is False
    assert _check(result, "gpu")["ok"] is False
    assert "--nv" in _check(result, "gpu")["detail"]
    assert _check(result, "model_answers")["detail"] == "not tried because an earlier check failed"
    assert answering == []
    assert (tmp_path / "work" / "preflight.json").exists()


def test_small_disk_names_needed_and_free_space(healthy, answering, monkeypatch, tmp_path):
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage(10**12, 0, 7 * preflight.GB))

    result = preflight.run_preflight(healthy, tmp_path / "work")

    disk = _check(result, "disk")
    assert disk["ok"] is False
    assert "45 GB" in disk["detail"]
    assert "only 7 GB is free" in disk["detail"]
    assert str(tmp_path / "hf") in disk["detail"]
    assert answering == []


def test_disk_check_uses_the_nearest_existing_parent(healthy, answering, monkeypatch, tmp_path):
    seen = []

    def usage(path):
        seen.append(Path(path))
        return Usage(10**12, 0, 100 * preflight.GB)

    monkeypatch.setattr(preflight.shutil, "disk_usage", usage)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing" / "deeper" / "hf"))

    preflight.run_preflight(healthy, tmp_path / "work")

    assert seen == [tmp_path.resolve()]


def test_model_failure_is_reported_in_plain_detail(healthy, monkeypatch, tmp_path):
    class Broken:
        def __init__(self, model):
            pass

        def __enter__(self):
            raise ServingError("vLLM did not start for demo-model")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(preflight, "serving_for", Broken)

    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert result["ok"] is False
    assert _check(result, "gpu")["ok"] is True
    answers = _check(result, "model_answers")
    assert answers["ok"] is False
    assert "vLLM did not start" in answers["detail"]


def test_model_that_hides_token_usage_fails(healthy, silent, tmp_path):
    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert _check(result, "model_answers")["ok"] is False
    assert "token usage" in _check(result, "model_answers")["detail"]


def test_missing_data_fails_before_the_model(healthy, answering, monkeypatch, tmp_path):
    _, popper_common = preflight.load_baseline()
    monkeypatch.setattr(popper_common, "check_data", lambda: ["winobias.csv"])

    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert "winobias.csv" in _check(result, "data")["detail"]
    assert answering == []


def test_snapshot_with_another_embedding_model_fails(healthy, answering, monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "Embedder", lambda: FakeEmbedder(model_name="other-embedder"))

    result = preflight.run_preflight(healthy, tmp_path / "work")

    snapshot = _check(result, "snapshot")
    assert snapshot["ok"] is False
    assert "other-embedder" in snapshot["detail"]


def test_model_without_a_disk_requirement_passes_the_disk_check(tmp_path, monkeypatch):
    job = write_job(tmp_path / "job")
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda path: Usage(10**12, 0, 3 * preflight.GB))

    assert preflight.check_disk(job)["ok"] is True


def test_modern_image_refuses_an_old_driver_and_names_the_fallback(healthy, answering, monkeypatch, tmp_path):
    monkeypatch.setenv("POPPER_IMAGE", "modern")
    monkeypatch.setattr(preflight, "_driver_version", lambda: "570.124.06")

    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert result["ok"] is False
    driver = _check(result, "driver")
    assert driver["ok"] is False
    assert "driver too old" in driver["detail"]
    assert "575.51" in driver["detail"]
    assert "popper_upgrade.sif llama-anchor" in driver["detail"]
    assert all(check["detail"].startswith("not tried") for check in result["checks"][1:])
    assert answering == []


def test_modern_image_accepts_a_new_enough_driver(healthy, answering, monkeypatch, tmp_path):
    monkeypatch.setenv("POPPER_IMAGE", "modern")
    monkeypatch.setattr(preflight, "_driver_version", lambda: "575.51.03")

    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert result["ok"] is True
    assert "575.51.03" in _check(result, "driver")["detail"]


def _qwen_large():
    path = Path(__file__).resolve().parents[1] / "registry" / "models.json"
    return next(model for model in json.loads(path.read_text(encoding="utf-8")) if model["name"] == "qwen-large")


def test_qwen_image_refuses_a_driver_below_the_cuda_12_floor_and_names_the_fallback(
    healthy, answering, monkeypatch, tmp_path
):
    monkeypatch.setenv("POPPER_IMAGE", "qwen")
    monkeypatch.setattr(preflight, "_driver_version", lambda: "520.61.05")

    result = preflight.run_preflight(healthy, tmp_path / "work")

    driver = _check(result, "driver")
    assert result["ok"] is False
    assert driver["ok"] is False
    assert "driver too old" in driver["detail"]
    assert "popper_qwen.sif needs 525.60 or newer" in driver["detail"]
    assert "popper_upgrade.sif llama-anchor" in driver["detail"]
    assert answering == []


@pytest.mark.parametrize("version", ["525.60.13", "535.104.05", "570.10"])
def test_qwen_image_accepts_any_driver_from_525_60(healthy, answering, monkeypatch, tmp_path, version):
    monkeypatch.setenv("POPPER_IMAGE", "qwen")
    monkeypatch.setattr(preflight, "_driver_version", lambda: version)

    result = preflight.run_preflight(healthy, tmp_path / "work")

    assert result["ok"] is True
    assert _check(result, "driver")["detail"] == f"driver {version} (needs 525.60 or newer)"


def test_qwen_image_fails_when_the_driver_cannot_be_read(healthy, monkeypatch):
    def broken():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setenv("POPPER_IMAGE", "qwen")
    monkeypatch.setattr(preflight, "_driver_version", broken)

    check = preflight.check_driver(healthy)

    assert check["ok"] is False
    assert "llama-anchor" in check["detail"]


def test_qwen_large_memory_check_fits_80_gb_and_names_the_fallback_on_40_gb(tmp_path, monkeypatch):
    job = write_job(tmp_path / "job", model=_qwen_large())

    monkeypatch.setattr(preflight, "_gpu_query", lambda: ("A100 80GB", 80.0))
    fits = preflight.check_gpu_memory(job)
    monkeypatch.setattr(preflight, "_gpu_query", lambda: ("A100 40GB", 40.0))
    small = preflight.check_gpu_memory(job)

    assert fits["ok"] is True
    assert "16384-token context" in fits["detail"]
    assert small["ok"] is False
    assert "40 GB" in small["detail"]
    assert "popper_upgrade.sif llama-anchor" in small["detail"]
    assert preflight.check_hf_token(job)["ok"] is True


def test_proven_image_has_no_driver_minimum(healthy, monkeypatch):
    monkeypatch.setattr(preflight, "_driver_version", lambda: "470.1")

    assert preflight.check_driver(healthy)["ok"] is True


def test_modern_image_fails_when_the_driver_cannot_be_read(healthy, monkeypatch):
    def broken():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setenv("POPPER_IMAGE", "modern")
    monkeypatch.setattr(preflight, "_driver_version", broken)

    check = preflight.check_driver(healthy)

    assert check["ok"] is False
    assert "llama-anchor" in check["detail"]


def test_gpu_memory_failure_names_weights_gpu_and_fallback(tmp_path, monkeypatch):
    serving = {"weights_gb": 61.3, "kv_kb_per_token": 6, "fallback": "ask Vladimir for the FP8 job"}
    job = write_job(tmp_path / "job", serving=serving)
    monkeypatch.setattr(preflight, "_gpu_query", lambda: ("Small GPU", 40.0))

    check = preflight.check_gpu_memory(job)

    assert check["ok"] is False
    assert "40 GB" in check["detail"]
    assert "ask Vladimir for the FP8 job" in check["detail"]


def test_gpu_memory_pass_reports_the_chosen_context(tmp_path, monkeypatch):
    job = write_job(tmp_path / "job", serving={"weights_gb": 60.4, "kv_kb_per_token": 192})
    monkeypatch.setattr(preflight, "_gpu_query", lambda: ("Big GPU", 80.0))

    check = preflight.check_gpu_memory(job)

    assert check["ok"] is True
    assert "16384-token context" in check["detail"]


def test_gated_model_without_token_fails_and_names_the_fallback(tmp_path, monkeypatch):
    serving = {"gated": True, "fallback": "use the ungated mirror job"}
    job = write_job(tmp_path / "job", serving=serving)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for name in preflight.TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)

    check = preflight.check_hf_token(job)

    assert check["ok"] is False
    assert "HF_TOKEN" in check["detail"]
    assert "use the ungated mirror job" in check["detail"]


def test_gated_model_passes_with_a_token_or_a_cached_copy(tmp_path, monkeypatch):
    job = write_job(tmp_path / "job", serving={"gated": True})
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for name in preflight.TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_TOKEN", "token")
    assert preflight.check_hf_token(job)["ok"] is True

    monkeypatch.delenv("HF_TOKEN")
    ref = tmp_path / "hf" / "hub" / f"models--{job_model(job)}" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc")
    assert preflight.check_hf_token(job)["ok"] is True


def test_ungated_model_needs_no_token(tmp_path):
    assert preflight.check_hf_token(write_job(tmp_path / "job"))["ok"] is True


def job_model(job):
    return preflight.model_row(job)["served_model_id"].replace("/", "--")
