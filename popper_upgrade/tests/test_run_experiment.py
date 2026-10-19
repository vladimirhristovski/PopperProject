import contextlib
import json

import pytest

import popper_upgrade.eval.run_experiment as module
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.controls import RandomSkillSource, WrongDomainSkillSource
from popper_upgrade.eval.run_experiment import (
    apply_overrides,
    current_prompt_hashes,
    experiment_id_for,
    run_experiment,
)
from popper_upgrade.eval.spec import load_spec, run_identity
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import OLLAMA_MODEL, FakeEmbedder, FakeSink, write_snapshot

HYPOTHESES = [{"id": "h1", "text": "gender hypothesis one"}, {"id": "h2", "text": "gender hypothesis two"}]
DATASETS = [{"name": "bbq", "permutable_columns": ["question"]}]


class FakeSession:
    instances = []

    def __init__(self, spec, source, sink, curator, reviewer, extractor, trajectory_dir, **kwargs):
        self.spec = spec
        self.source = source
        self.curator = curator
        self.reviewer = reviewer
        self.kwargs = kwargs
        type(self).instances.append(self)

    def run(self):
        return "run"


@pytest.fixture
def sessions(monkeypatch):
    cls = type("Session", (FakeSession,), {"instances": []})
    monkeypatch.setattr(module, "RunSession", cls)
    return cls.instances


@pytest.fixture
def serving(monkeypatch):
    entered = []

    @contextlib.contextmanager
    def fake(model):
        entered.append(model["name"])
        counter = UsageCounter()
        counter.port = 4321
        yield counter

    monkeypatch.setattr(module, "serving_for", fake)

    def agents(kwargs, skills, reviewer, extractor):
        return (object() if skills else None, object() if reviewer else None, None)

    monkeypatch.setattr(module, "build_agents", agents)
    return entered


def _write_spec(tmp_path, **changes):
    data = {
        "name": "unit_v1",
        "models": ["llama3.1:8b"],
        "arms": ["baseline", "skills", "random_skills", "wrong_domain_skills"],
        "source": {"kind": "gender_custom", "hypothesis_set": "gender_v1", "task": "gender_bias"},
        "train_tasks": [],
        "test_tasks": ["gender_bias"],
        "replicates": 2,
        "permutation_seeds": [5, 6],
        "prompt_versions": current_prompt_hashes(),
    }
    data.update(changes)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _snapshot(tmp_path, skills=()):
    embedder = FakeEmbedder()
    path = write_snapshot(
        tmp_path / "snap.json",
        list(skills),
        embedder,
        models=[OLLAMA_MODEL],
        hypothesis_sets={"gender_v1": HYPOTHESES},
        datasets=DATASETS,
    )
    return SnapshotSkillSource(path, embedder)


def _run(tmp_path, source, sink, **kwargs):
    spec = kwargs.pop("spec", None) or _write_spec(tmp_path)
    return run_experiment(
        spec, "llama3.1:8b", source, sink, "laptop", data_dir=tmp_path, trajectory_dir=tmp_path / "traj", **kwargs
    )


def test_runs_all_expanded_specs(tmp_path, sessions, serving):
    sink = FakeSink()
    source = _snapshot(tmp_path)

    experiment_id = _run(tmp_path, source, sink)

    assert len(sessions) == 2 * (2 + 1) * 4
    assert serving == ["llama3.1:8b"]
    assert [e["id"] for e in sink.experiments] == [experiment_id]
    assert sink.experiments[0]["name"] == "unit_v1"
    assert {s.spec.experiment_id for s in sessions} == {experiment_id}
    assert {s.spec.library_snapshot_id for s in sessions} == {source.snapshot_id}
    assert {s.spec.split for s in sessions} == {"test"}
    assert {s.spec.popper_kwargs["server_port"] for s in sessions} == {4321}
    assert {s.spec.popper_kwargs["domain"] for s in sessions} == {"social science"}
    assert {s.spec.hypothesis_id for s in sessions} == {"h1", "h2"}
    assert {s.spec.answer for s in sessions} == {False, None}


def test_matched_seeds_across_arms(tmp_path, sessions, serving):
    _run(tmp_path, _snapshot(tmp_path), FakeSink())

    slots = {}
    for session in sessions:
        spec = session.spec
        slots.setdefault((spec.hypothesis, spec.permuted, spec.replicate), set()).add(spec.permute_seed)
    assert all(len(seeds) == 1 for seeds in slots.values())
    assert {seed for (_, permuted, _), seeds in slots.items() if permuted for seed in seeds} == {5, 6}


def test_control_arms_get_control_sources(tmp_path, sessions, serving):
    _run(tmp_path, _snapshot(tmp_path), FakeSink())

    kinds = {}
    for session in sessions:
        kinds.setdefault(session.spec.arm, set()).add(type(session.source))
    assert kinds["random_skills"] == {RandomSkillSource}
    assert kinds["wrong_domain_skills"] == {WrongDomainSkillSource}
    assert kinds["baseline"] == kinds["skills"] == {SnapshotSkillSource}
    arms = {s.spec.arm: (s.curator is not None, s.reviewer is not None) for s in sessions}
    assert arms == {
        "baseline": (False, False),
        "skills": (True, False),
        "random_skills": (True, False),
        "wrong_domain_skills": (True, False),
    }


def test_resume_skips_completed(tmp_path, sessions, serving):
    sink = FakeSink()
    source = _snapshot(tmp_path)
    _run(tmp_path, source, sink)
    first_runs = [s.spec for s in sessions]
    sessions.clear()
    sink.completed = {run_identity(spec) for spec in first_runs[:7]}

    _run(tmp_path, source, sink)

    assert len(sessions) == len(first_runs) - 7
    assert not {run_identity(s.spec) for s in sessions} & sink.completed


def test_isolation_checked_before_any_run(tmp_path, sessions, serving):
    leaked = Skill("s1", "a", "d", "", "social science", "Gender hypothesis ONE", approved=True)
    sink = FakeSink()

    with pytest.raises(ValueError, match="skill s1 derived from test hypothesis"):
        _run(tmp_path, _snapshot(tmp_path, [leaked]), sink)

    assert sessions == []
    assert serving == []
    assert sink.experiments == []


def test_stale_prompt_versions_are_refused(tmp_path, sessions, serving):
    hashes = {**current_prompt_hashes(), "reviewer": "0" * 64}
    spec = _write_spec(tmp_path, prompt_versions=hashes)

    with pytest.raises(ValueError, match="prompt versions differ from the spec for: reviewer"):
        _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=spec)

    assert serving == []


def test_model_outside_the_spec_is_refused(tmp_path, sessions, serving):
    spec = _write_spec(tmp_path, models=["other-model"])

    with pytest.raises(ValueError, match="not part of unit_v1"):
        _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=spec)


def test_unfrozen_source_is_refused(tmp_path, sessions, serving):
    with pytest.raises(ValueError, match="frozen library snapshot"):
        _run(tmp_path, object(), FakeSink())


def test_overrides_restrict_tasks_and_replicates(tmp_path):
    spec = load_spec(
        _write_spec(tmp_path, test_tasks=["gender_bias", "extra"], replicates=2, permutation_seeds=[5, 6])
    )

    narrowed = apply_overrides(spec, ["gender_bias"], 1)

    assert (narrowed.test_tasks, narrowed.replicates) == (["gender_bias"], 1)
    with pytest.raises(ValueError, match="not test tasks"):
        apply_overrides(spec, ["unknown"], None)


def test_experiment_id_is_shared_across_models_and_depends_on_snapshot(tmp_path):
    spec = load_spec(_write_spec(tmp_path))
    other_models = load_spec(_write_spec(tmp_path, models=["llama3.1:8b", "second-model"]))

    assert experiment_id_for(spec, "snap") == experiment_id_for(other_models, "snap")
    assert experiment_id_for(spec, "snap") != experiment_id_for(spec, "other-snap")
    assert experiment_id_for(spec, "snap") != experiment_id_for(spec, "snap", 1)
