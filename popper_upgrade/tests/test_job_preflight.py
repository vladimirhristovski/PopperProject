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
    assert [check["name"] for check in result["checks"]] == ["gpu", "disk", "data", "snapshot", "model_answers"]
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
