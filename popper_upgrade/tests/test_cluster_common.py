import json
import os
import signal
from collections import Counter
from pathlib import Path

import pytest

import popper_upgrade.models as models_module
import popper_upgrade.run as run_module
import popper_upgrade.runner as runner_module
from cluster import common
from popper_upgrade.tests.fakes import (
    HOSTED_MODEL,
    HYPOTHESES,
    VLLM_MODEL,
    FakeEmbedder,
    FakePopper,
    patch_agent_classes,
    write_snapshot,
)


class _FakeServer:
    instances = []
    start_result = True

    def __init__(self, model, port, host="localhost", proxy_port=None):
        self.calls = []
        self.proxy_port = proxy_port
        type(self).instances.append(self)

    def start(self):
        self.calls.append("start")
        return type(self).start_result

    def start_proxy(self):
        self.calls.append("start_proxy")

    def stop_proxy(self):
        self.calls.append("stop_proxy")

    def stop(self):
        self.calls.append("stop")


class _FakeUsageProxy:
    def __init__(self, listen_port, upstream_port, counter):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class _RecordingPopper(FakePopper):
    init_kwargs = []
    registered = []
    configured = []
    validations = []
    on_validate = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).init_kwargs.append(kwargs)

    def register_data(self, data_path, loader_type="bio", **kwargs):
        type(self).registered.append((data_path, loader_type))

    def configure(self, **kwargs):
        type(self).configured.append(kwargs)

    def validate(self, hypothesis):
        type(self).validations.append(hypothesis)
        if type(self).on_validate is not None:
            type(self).on_validate(len(type(self).validations))
        return super().validate(hypothesis)


@pytest.fixture
def env(monkeypatch, tmp_path):
    config, popper_common = common.load_baseline()
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(popper_common, "check_data", lambda: [])
    monkeypatch.setattr(common, "Embedder", FakeEmbedder)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    server = type("Server", (_FakeServer,), {"instances": [], "start_result": True})
    popper = type(
        "Popper",
        (_RecordingPopper,),
        {"init_kwargs": [], "registered": [], "configured": [], "validations": [], "on_validate": None},
    )
    monkeypatch.setattr(models_module, "VllmServer", server)
    monkeypatch.setattr(models_module, "UsageProxy", _FakeUsageProxy)
    monkeypatch.setattr(runner_module, "Popper", popper)
    patch_agent_classes(monkeypatch, run_module)
    snapshot = write_snapshot(
        tmp_path / "snapshot.json",
        [],
        FakeEmbedder(),
        models=[VLLM_MODEL, HOSTED_MODEL],
        hypothesis_sets={"gender_v1": HYPOTHESES},
    )

    class Env:
        pass

    result = Env()
    result.config = config
    result.server = server
    result.popper = popper
    result.snapshot = snapshot
    result.bundle = tmp_path / "bundle"
    return result


def _rows(bundle):
    path = bundle / "runs.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[row["id"]] = {**rows.get(row["id"], {}), **row}
    return rows


def test_vllm_start_failure_returns_1_and_stops_proxy(env):
    env.server.start_result = False

    code = common.run_model(VLLM_MODEL["name"], env.snapshot, env.bundle)

    assert code == 1
    assert env.server.instances[0].calls == ["start", "stop", "stop_proxy"]
    assert _rows(env.bundle) == {}
    assert env.popper.validations == []


def test_runs_every_hypothesis_for_every_arm(env):
    code = common.run_model(VLLM_MODEL["name"], env.snapshot, env.bundle)

    assert code == 0
    rows = _rows(env.bundle)
    assert len(rows) == 15
    assert Counter(row["arm"] for row in rows.values()) == {"baseline": 5, "skills": 5, "skills_reviewer": 5}
    assert {row["status"] for row in rows.values()} == {"completed"}
    assert {row["environment"] for row in rows.values()} == {"cluster"}
    assert {row["library_snapshot_id"] for row in rows.values()} == {"snap-1"}
    assert {row["human_review"] for row in rows.values()} == {False}
    assert Counter(row["hypothesis"] for row in rows.values()) == dict.fromkeys([h["text"] for h in HYPOTHESES], 3)
    assert env.server.instances[0].calls == ["start", "start_proxy", "stop", "stop_proxy"]


def test_serving_environment_matches_test3_local(env):
    seen = {}

    def capture(count):
        seen.update(key=os.environ["OPENAI_API_KEY"], url=os.environ["OPENAI_BASE_URL"])

    env.popper.on_validate = staticmethod(capture)

    common.run_model(VLLM_MODEL["name"], env.snapshot, env.bundle, arms=("baseline",))

    assert seen == {"key": "vllm", "url": "http://localhost:8001/v1"}


def test_uses_baseline_parameters(env):
    common.run_model(VLLM_MODEL["name"], env.snapshot, env.bundle, arms=("baseline",))

    config = env.config
    expected = {
        "alpha": config.ALPHA,
        "max_num_of_tests": config.MAX_TESTS,
        "max_retry": config.MAX_RETRY,
        "time_limit": config.TIME_LIMIT,
        "aggregate_test": "E-value",
        "relevance_checker": True,
        "use_react_agent": True,
    }
    assert env.popper.configured == [expected] * 5
    assert env.popper.registered == [(config.DATA_DIR, "custom")] * 5
    assert len(env.popper.init_kwargs) == 5
    assert env.popper.init_kwargs[0] == {
        "llm": VLLM_MODEL["name"],
        "is_locally_served": True,
        "server_port": 8001,
        "plot_agent_architecture": False,
        "domain": "social science",
        "api_key": "EMPTY",
    }


def test_sigterm_marks_current_run_error_and_stops_servers(env):
    def interrupt(count):
        if count == 2:
            signal.raise_signal(signal.SIGTERM)

    env.popper.on_validate = staticmethod(interrupt)
    previous = signal.getsignal(signal.SIGTERM)

    code = common.run_model(VLLM_MODEL["name"], env.snapshot, env.bundle)

    assert code == 128 + signal.SIGTERM
    rows = list(_rows(env.bundle).values())
    assert [row["status"] for row in rows] == ["completed", "error"]
    assert "received signal" in rows[1]["error"]
    assert env.server.instances[0].calls[-2:] == ["stop", "stop_proxy"]
    assert len(env.popper.validations) == 2
    assert signal.getsignal(signal.SIGTERM) is previous


def test_hosted_model_refused_before_any_llm_call(env):
    code = common.run_model(HOSTED_MODEL["name"], env.snapshot, env.bundle)

    assert code == 1
    assert env.server.instances == []
    assert env.popper.init_kwargs == []
    assert _rows(env.bundle) == {}


def test_unknown_model_returns_1(env):
    assert common.run_model("missing-model", env.snapshot, env.bundle) == 1
    assert env.server.instances == []


def test_missing_data_returns_1(env, monkeypatch):
    monkeypatch.setattr(common.load_baseline()[1], "check_data", lambda: ["winobias.csv"])

    assert common.run_model(VLLM_MODEL["name"], env.snapshot, env.bundle) == 1
    assert env.server.instances == []


def test_run_model_script_passes_the_chosen_model(monkeypatch):
    from cluster import run_model

    calls = []
    monkeypatch.setattr(common, "run_model", lambda *args: calls.append(args) or 0)

    assert run_model.main(["--model", "any-model", "--snapshot", "s.json", "--bundle-dir", "b"]) == 0
    assert calls == [("any-model", Path("s.json"), Path("b"))]


def test_experiment_job_exits_with_a_distinct_code_when_the_server_died(env, monkeypatch, capsys):
    from popper_upgrade.eval.run_experiment import ServerExited

    def dead(*args, **kwargs):
        raise ServerExited("the vLLM server is no longer responding; stopped before starting the next run")

    monkeypatch.setattr(common, "run_experiment", dead)

    code = common.run_experiment_job("spec.json", VLLM_MODEL["name"], env.snapshot, env.bundle)

    assert code == 4
    assert "no longer responding" in capsys.readouterr().out
