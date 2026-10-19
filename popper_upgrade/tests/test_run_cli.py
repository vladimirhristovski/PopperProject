import contextlib
import hashlib

import pytest

import popper_upgrade.run as run_module
from popper_upgrade.db.bundle import BundleSink
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.models import ServingError
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.fakes import (
    HOSTED_MODEL,
    HYPOTHESES,
    OLLAMA_MODEL,
    FakeEmbedder,
    FakeSink,
    patch_agent_classes,
    write_snapshot,
)


class FakeSession:
    instances = []

    def __init__(self, spec, source, sink, curator, reviewer, extractor, trajectory_dir, **kwargs):
        self.spec = spec
        self.sink = sink
        self.curator = curator
        self.reviewer = reviewer
        self.extractor = extractor
        self.kwargs = kwargs
        type(self).instances.append(self)

    def run(self):
        run_id = self.sink.create_run({"hypothesis": self.spec.hypothesis, "status": "running"})
        result = {"status": "completed", "res_stat": 3.5, "n_tests": 2, "tokens_in": 10, "tokens_out": 5}
        self.sink.finish_run(run_id, {**result, "wall_time_s": 1.5})
        return run_id


@pytest.fixture
def session_class(monkeypatch):
    cls = type("Session", (FakeSession,), {"instances": []})
    monkeypatch.setattr(run_module, "RunSession", cls)
    return cls


@pytest.fixture
def agents(monkeypatch):
    return patch_agent_classes(monkeypatch, run_module)


@pytest.fixture
def serving(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def fake(model):
        entered.append(model["name"])
        yield UsageCounter()

    monkeypatch.setattr(run_module, "serving_for", fake)
    return entered


@pytest.fixture
def snapshot_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "Embedder", FakeEmbedder)
    monkeypatch.setenv("POPPER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("POPPER_TRAJECTORY_DIR", str(tmp_path / "trajectories"))
    return write_snapshot(
        tmp_path / "snap.json",
        [],
        FakeEmbedder(),
        models=[OLLAMA_MODEL, HOSTED_MODEL],
        hypothesis_sets={"gender_v1": HYPOTHESES},
    )


def _args(snapshot_file, tmp_path, *extra, model="llama3.1:8b"):
    return [
        "--model",
        model,
        "--hypothesis-set",
        "gender_v1",
        "--source",
        f"snapshot:{snapshot_file}",
        "--sink",
        f"bundle:{tmp_path / 'bundle'}",
        *extra,
    ]


def _run(arm, tmp_path, **kwargs):
    return run_module.run_hypotheses(
        OLLAMA_MODEL,
        kwargs.pop("hypotheses", HYPOTHESES[:2]),
        arm,
        kwargs.pop("source", object()),
        kwargs.pop("sink", FakeSink()),
        "laptop",
        data_dir=tmp_path,
        trajectory_dir=tmp_path,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("arm", "skills", "reviewer"),
    [("baseline", False, False), ("skills", True, False), ("skills_reviewer", True, True)],
)
def test_arm_flags(session_class, agents, tmp_path, arm, skills, reviewer):
    _run(arm, tmp_path)

    assert len(session_class.instances) == 2
    for session in session_class.instances:
        assert (session.spec.skills_enabled, session.spec.reviewer_enabled) == (skills, reviewer)
        assert (session.curator is not None, session.reviewer is not None) == (skills, reviewer)
        assert session.extractor is None
    assert len(agents["SkillCurator"]) == int(skills)
    assert len(agents["Reviewer"]) == int(reviewer)


def test_agents_use_the_run_model_and_port(session_class, agents, tmp_path):
    _run("skills_reviewer", tmp_path, learn=True)

    expected = ("llama3.1:8b", {"port": 11435, "api_key": "EMPTY"})
    assert agents["SkillCurator"] == [expected]
    assert agents["Reviewer"] == [expected]
    assert agents["SkillExtractor"] == [expected]


def test_unknown_arm_is_rejected(session_class, agents, tmp_path):
    with pytest.raises(ValueError, match="unknown arm"):
        _run("other", tmp_path)

    assert session_class.instances == []


def test_cli_runs_each_hypothesis_once(session_class, agents, serving, snapshot_file, tmp_path, capsys):
    code = run_module.main(_args(snapshot_file, tmp_path, "--arm", "skills", "--seed", "7"))

    assert code == 0
    assert serving == ["llama3.1:8b"]
    assert [s.spec.hypothesis for s in session_class.instances] == [h["text"] for h in HYPOTHESES]
    assert {s.spec.seed for s in session_class.instances} == {7}
    assert {s.spec.environment for s in session_class.instances} == {"laptop"}
    assert {s.spec.library_snapshot_id for s in session_class.instances} == {"snap-1"}
    out = capsys.readouterr().out
    assert out.count("completed") == 5
    assert "10/5" in out
    assert "3.5" in out


def test_cli_hypothesis_index_selects(session_class, agents, serving, snapshot_file, tmp_path):
    code = run_module.main(
        _args(snapshot_file, tmp_path, "--arm", "baseline", "--hypothesis-index", "1", "--hypothesis-index", "3")
    )

    assert code == 0
    assert [s.spec.hypothesis_id for s in session_class.instances] == ["h1", "h3"]


def test_cli_rejects_bad_index(session_class, agents, serving, snapshot_file, tmp_path):
    code = run_module.main(_args(snapshot_file, tmp_path, "--arm", "baseline", "--hypothesis-index", "9"))

    assert code == 2
    assert session_class.instances == []
    assert serving == []


def test_cli_refuses_hosted_model_on_cluster_before_any_llm_call(
    session_class, agents, serving, snapshot_file, tmp_path, capsys
):
    code = run_module.main(
        _args(snapshot_file, tmp_path, "--arm", "baseline", "--environment", "cluster", model="claude-sonnet-4-5")
    )

    assert code == 2
    assert "model claude-sonnet-4-5 is not available on cluster" in capsys.readouterr().err
    assert serving == []
    assert session_class.instances == []
    assert agents["SkillCurator"] == []


def test_cli_bundle_sink_needs_snapshot_source(monkeypatch, tmp_path, capsys):
    class DummyRepository:
        def __init__(self, conn, embedder):
            pass

    monkeypatch.setattr(run_module, "_open_connection", lambda: object())
    monkeypatch.setattr(run_module, "SkillRepository", DummyRepository)
    monkeypatch.setattr(run_module, "Embedder", FakeEmbedder)

    code = run_module.main(
        ["--model", "m", "--hypothesis-set", "s", "--arm", "baseline", "--sink", f"bundle:{tmp_path}"]
    )

    assert code == 2
    assert "bundle sink needs a snapshot source" in capsys.readouterr().err


def test_cli_reports_serving_failure(session_class, agents, snapshot_file, tmp_path, monkeypatch, capsys):
    @contextlib.contextmanager
    def failing(model):
        raise ServingError("daemon down")
        yield

    monkeypatch.setattr(run_module, "serving_for", failing)

    code = run_module.main(_args(snapshot_file, tmp_path, "--arm", "baseline"))

    assert code == 1
    assert "daemon down" in capsys.readouterr().err
    assert session_class.instances == []


def test_snapshot_sessions_resolve_prompt_versions_without_usage_recorder(session_class, agents, tmp_path):
    digest = hashlib.sha256(b"curator\ncurator prompt").hexdigest()
    path = write_snapshot(
        tmp_path / "pv.json",
        [],
        FakeEmbedder(),
        prompt_versions=[{"id": "pv-1", "role": "curator", "text": "curator prompt", "hash": digest}],
    )
    snapshot = SnapshotSkillSource(path, FakeEmbedder())

    _run("skills", tmp_path, source=snapshot, sink=BundleSink(tmp_path / "b", snapshot), hypotheses=HYPOTHESES[:1])

    [session] = session_class.instances
    resolver = session.kwargs["prompt_version_resolver"]
    assert resolver("curator", "curator prompt") == "pv-1"
    assert resolver("curator", "other text") is None
    assert resolver("extractor", "curator prompt") is None
    assert session.kwargs["usage_recorder"] is None
    assert session.kwargs["usage"] is None


def test_stop_requested_ends_the_loop(session_class, agents, tmp_path):
    _run("baseline", tmp_path, hypotheses=HYPOTHESES, stop_requested=lambda: len(session_class.instances) >= 2)

    assert len(session_class.instances) == 2
