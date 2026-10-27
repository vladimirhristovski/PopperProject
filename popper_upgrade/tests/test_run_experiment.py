import contextlib
import json
from collections import Counter

import pytest

import popper_upgrade.eval.run_experiment as module
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.controls import RandomSkillSource, WrongDomainSkillSource
from popper_upgrade.eval.learning import Learning
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
    def fake(model, log_dir=None):
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


TRAIN_HYPOTHESES = ["train hypothesis one", "train hypothesis two", "train hypothesis three"]


def _learning_spec(tmp_path, **changes):
    source = {
        "kind": "gender_custom",
        "hypothesis_set": "gender_v1",
        "task": "gender_bias",
        "train_task": "gender_bias_train",
        "train_hypotheses": TRAIN_HYPOTHESES,
    }
    return _write_spec(tmp_path, source=source, train_tasks=["gender_bias_train"], **changes)


def _frozen(tmp_path, skills=()):
    embedder = FakeEmbedder()
    path = write_snapshot(
        tmp_path / "frozen.json",
        list(skills),
        embedder,
        snapshot_id="frozen-1",
        models=[OLLAMA_MODEL],
        hypothesis_sets={"gender_v1": HYPOTHESES},
        datasets=DATASETS,
    )
    return SnapshotSkillSource(path, embedder)


class _Recorder:
    def __init__(self, frozen, events, stop_after_freeze=False):
        self.frozen = frozen
        self.events = events
        self.calls = 0

    def freeze(self, key=None):
        self.calls += 1
        self.events.append("freeze")
        return self.frozen


def _learning(tmp_path, events, skills=()):
    recorder = _Recorder(_frozen(tmp_path, skills), events)
    sink = FakeSink()
    live = object()
    return Learning(source=live, sink=sink, freeze=recorder.freeze), recorder, sink, live


@pytest.fixture
def ordered_sessions(monkeypatch, sessions, serving):
    events = []

    class Ordered(FakeSession):
        def run(self):
            events.append(("train" if self.spec.split == "train" else "test", self.spec.arm))
            return "run"

    monkeypatch.setattr(module, "RunSession", Ordered)
    agent_calls = []

    def agents(kwargs, skills, reviewer, extractor):
        agent_calls.append((skills, reviewer, extractor))
        return (object(), object(), object() if extractor else None)

    monkeypatch.setattr(module, "build_agents", agents)
    return events, agent_calls


def test_learning_runs_train_split_before_freezing_and_evaluating(tmp_path, serving, ordered_sessions):
    events, agent_calls = ordered_sessions
    learning, recorder, learn_sink, live = _learning(tmp_path, events)
    sink = FakeSink()
    source = _snapshot(tmp_path)

    experiment_id = _run(tmp_path, source, sink, spec=_learning_spec(tmp_path), learning=learning)

    count = len(TRAIN_HYPOTHESES)
    assert events[:count] == [("train", "skills_reviewer")] * count
    assert events[count] == "freeze"
    assert all(event[0] == "test" for event in events[count + 1 :])
    assert (True, True, True) in agent_calls
    assert recorder.calls == 1
    assert learn_sink.experiments[0]["spec"]["phase"] == "learn"
    assert sink.experiments[0]["spec"]["phase"] == "evaluate"
    assert sink.experiments[0]["id"] == experiment_id
    assert sink.experiments[0]["spec"]["learn_experiment_id"] == learn_sink.experiments[0]["id"]
    assert sink.experiments[0]["spec"]["base_snapshot_id"] == source.snapshot_id
    assert experiment_id != learn_sink.experiments[0]["id"]


def test_train_runs_learn_on_the_live_library_and_test_runs_use_the_frozen_one(tmp_path, sessions, serving):
    events = []
    learning, _, learn_sink, live = _learning(tmp_path, events)
    source = _snapshot(tmp_path)

    _run(tmp_path, source, FakeSink(), spec=_learning_spec(tmp_path), learning=learning)

    train = [s for s in sessions if s.spec.split == "train"]
    test = [s for s in sessions if s.spec.split == "test"]
    assert len(train) == len(TRAIN_HYPOTHESES)
    assert {s.spec.arm for s in train} == {"skills_reviewer"}
    assert {s.spec.learn for s in train} == {True}
    assert {s.spec.permuted for s in train} == {False}
    assert {s.spec.hypothesis for s in train} == set(TRAIN_HYPOTHESES)
    assert {s.spec.library_snapshot_id for s in train} == {source.snapshot_id}
    assert {s.source for s in train} == {live}
    assert {s.spec.library_snapshot_id for s in test} == {"frozen-1"}
    assert {type(s.source) for s in test if s.spec.arm == "baseline"} == {SnapshotSkillSource}
    assert not {s.spec.hypothesis for s in test} & set(TRAIN_HYPOTHESES)


def test_zero_approved_skills_still_evaluates(tmp_path, sessions, serving):
    learning, recorder, _, _ = _learning(tmp_path, [])

    _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_learning_spec(tmp_path), learning=learning)

    assert recorder.calls == 1
    assert len([s for s in sessions if s.spec.split == "test"]) == 2 * (2 + 1) * 4


def test_learning_resumes_across_phases(tmp_path, sessions, serving):
    learning, _, learn_sink, _ = _learning(tmp_path, [])
    sink = FakeSink()
    source = _snapshot(tmp_path)
    spec = _learning_spec(tmp_path)
    _run(tmp_path, source, sink, spec=spec, learning=learning)
    train = [s.spec for s in sessions if s.spec.split == "train"]
    tests = [s.spec for s in sessions if s.spec.split == "test"]
    sessions.clear()
    learn_sink.completed = {run_identity(spec) for spec in train}
    sink.completed = {run_identity(spec) for spec in tests[:5]}

    _run(tmp_path, source, sink, spec=spec, learning=learning)

    assert [s for s in sessions if s.spec.split == "train"] == []
    assert len(sessions) == len(tests) - 5


def test_stop_during_learning_skips_freeze_and_evaluation(tmp_path, sessions, serving):
    learning, recorder, learn_sink, _ = _learning(tmp_path, [])

    experiment_id = _run(
        tmp_path, _snapshot(tmp_path), FakeSink(), spec=_learning_spec(tmp_path), learning=learning,
        stop_requested=lambda: True,
    )

    assert recorder.calls == 0
    assert sessions == []
    assert experiment_id == learn_sink.experiments[0]["id"]


def test_specs_without_train_tasks_ignore_learning(tmp_path, sessions, serving):
    learning, recorder, learn_sink, _ = _learning(tmp_path, [])
    sink = FakeSink()

    _run(tmp_path, _snapshot(tmp_path), sink, learning=learning)

    assert recorder.calls == 0
    assert learn_sink.experiments == []
    assert "phase" not in sink.experiments[0]["spec"]
    assert len(sessions) == 2 * (2 + 1) * 4


def test_train_hypotheses_overlapping_a_test_hypothesis_are_refused_by_isolation(tmp_path, sessions, serving):
    leaked = Skill("s1", "a", "d", "", "social science", "Gender hypothesis ONE", approved=True)

    with pytest.raises(ValueError, match="derived from test hypothesis"):
        _run(tmp_path, _snapshot(tmp_path, [leaked]), FakeSink(), spec=_learning_spec(tmp_path))


class StubBudget:
    def __init__(self, allow):
        self.allow = allow
        self.calls = []
        self.arms = []
        self.plans = []
        self.exhausted = False

    def begin(self, plan, spec_path, model_row):
        self.plans.append((plan, model_row["name"]))

    def may_start(self, phase, arm=None):
        self.calls.append(phase)
        self.arms.append(arm)
        if len(self.calls) > self.allow:
            self.exhausted = True
            return False
        return True


def test_budget_gates_each_run_before_it_starts(tmp_path, sessions, serving):
    budget = StubBudget(5)

    _run(tmp_path, _snapshot(tmp_path), FakeSink(), budget=budget)

    assert len(sessions) == 5
    assert budget.calls == ["evaluate"] * 6
    assert budget.arms[:5] == [session.spec.arm for session in sessions]
    assert budget.exhausted is True


def test_budget_is_told_the_plan_and_experiment_rows_carry_it(tmp_path, sessions, serving):
    budget = StubBudget(100)
    sink = FakeSink()

    _run(tmp_path, _snapshot(tmp_path), sink, budget=budget)

    arms = {"baseline": 6, "skills": 6, "random_skills": 6, "wrong_domain_skills": 6}
    assert budget.plans == [({"evaluate": arms}, "llama3.1:8b")]
    assert sink.experiments[0]["spec"]["planned_runs"] == arms
    assert len(budget.calls) == len(sessions) == 24


def test_runs_already_completed_are_not_charged_to_the_budget(tmp_path, sessions, serving):
    source = _snapshot(tmp_path)
    sink = FakeSink()
    _run(tmp_path, source, sink)
    done = {run_identity(s.spec) for s in sessions[:7]}
    sessions.clear()
    sink.completed = done
    budget = StubBudget(100)

    _run(tmp_path, source, sink, budget=budget)

    assert len(budget.calls) == len(sessions) == 24 - 7


def test_stopped_and_resumed_runs_follow_the_same_order(tmp_path, sessions, serving):
    source = _snapshot(tmp_path)
    sink = FakeSink()
    _run(tmp_path, source, sink)
    order = [run_identity(s.spec) for s in sessions]
    sessions.clear()
    _run(tmp_path, source, sink, budget=StubBudget(9))
    first = [run_identity(s.spec) for s in sessions]
    sessions.clear()
    sink.completed = set(first)

    _run(tmp_path, source, sink)

    assert first == order[:9]
    assert [run_identity(s.spec) for s in sessions] == order[9:]


def test_every_prefix_of_the_schedule_is_balanced_across_arms(tmp_path, sessions, serving):
    _run(tmp_path, _snapshot(tmp_path), FakeSink())

    arms = [s.spec.arm for s in sessions]
    for size in range(len(arms) + 1):
        counts = [arms[:size].count(arm) for arm in set(arms)]
        assert max(counts) - min(counts) <= 1


def test_budget_stop_during_learning_skips_freeze_and_evaluation(tmp_path, sessions, serving):
    learning, recorder, learn_sink, _ = _learning(tmp_path, [])
    budget = StubBudget(1)

    experiment_id = _run(
        tmp_path, _snapshot(tmp_path), FakeSink(), spec=_learning_spec(tmp_path), learning=learning, budget=budget
    )

    assert [s.spec.split for s in sessions] == ["train"]
    assert recorder.calls == 0
    assert experiment_id == learn_sink.experiments[0]["id"]
    assert budget.plans[0][0] == {
        "learn": {"skills_reviewer": len(TRAIN_HYPOTHESES)},
        "evaluate": {"baseline": 6, "skills": 6, "random_skills": 6, "wrong_domain_skills": 6},
    }
    assert learn_sink.experiments[0]["spec"]["planned_runs"] == {"skills_reviewer": len(TRAIN_HYPOTHESES)}


def test_budget_stop_in_the_evaluation_after_learning_still_freezes(tmp_path, sessions, serving):
    learning, recorder, _, _ = _learning(tmp_path, [])
    budget = StubBudget(len(TRAIN_HYPOTHESES))

    _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_learning_spec(tmp_path), learning=learning, budget=budget)

    assert recorder.calls == 1
    assert [s.spec.split for s in sessions] == ["train"] * len(TRAIN_HYPOTHESES)
    assert budget.calls[-1] == "evaluate"


def _combined_spec(tmp_path, **gender):
    source = {
        "kind": "combined",
        "parts": [
            {
                "kind": "gender_custom",
                "hypothesis_set": "gender_v1",
                "task": "gender_bias",
                "train_task": "gender_bias_train",
                "train_hypotheses": TRAIN_HYPOTHESES,
                "replicates": {"original": 2, "permuted": 3},
                **gender,
            },
            {
                "kind": "discoverybench",
                "path": "bench",
                "sample": {"originals": 1, "seed": 9},
                "replicates": {"original": 1, "permuted": 1},
            },
        ],
    }
    return _write_spec(
        tmp_path,
        source=source,
        train_tasks=["gender_bias_train", "db_train"],
        test_tasks=["gender_bias", "db_test"],
        replicates=3,
        permutation_seeds=[5, 6, 7],
        arms=["baseline", "skills"],
    )


@pytest.fixture
def bench(monkeypatch):
    calls = []

    def examples(path, tasks, per_task=None, sample=None):
        calls.append((path, set(tasks), sample))
        base = {"domain": "economics", "factory": lambda seed: lambda: None, "datasets": []}
        return [
            {**base, "task": task, "hypothesis": f"{task} claim", "answer": answer, "permuted": not answer}
            for task in sorted(tasks & {"db_test", "db_train"})
            for answer in (True, False)
        ]

    monkeypatch.setattr(module, "discoverybench_examples", examples)
    return calls


def test_combined_source_gives_each_part_its_own_replicates(tmp_path, sessions, serving, bench):
    _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_combined_spec(tmp_path))

    gender = [s.spec for s in sessions if s.spec.task == "gender_bias"]
    discovery = [s.spec for s in sessions if s.spec.task == "db_test"]
    assert len(gender) == 2 * 2 * (2 + 3)
    assert len(discovery) == 2 * (1 + 1)
    assert {s.replicate for s in gender if not s.permuted} == {0, 1}
    assert {s.replicate for s in gender if s.permuted} == {0, 1, 2}
    assert {s.replicate for s in discovery} == {0}
    assert {s.popper_kwargs["domain"] for s in discovery} == {"economics"}
    assert bench[0][2] == {"originals": 1, "seed": 9}


def test_combined_source_learns_from_unsampled_train_tasks(tmp_path, sessions, serving, bench):
    learning, _, _, _ = _learning(tmp_path, [])

    _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_combined_spec(tmp_path), learning=learning)

    train = [s.spec for s in sessions if s.spec.split == "train"]
    assert Counter(s.task for s in train) == {"gender_bias_train": 3, "db_train": 1}
    assert bench[0][1:] == ({"gender_bias", "db_test"}, {"originals": 1, "seed": 9})
    assert bench[1] == (str(tmp_path / "bench"), {"gender_bias_train", "db_train"}, None)


def test_replicate_override_replaces_part_replicates(tmp_path):
    spec = load_spec(_combined_spec(tmp_path))

    narrowed = apply_overrides(spec, None, 1)

    assert narrowed.replicates == 1
    assert all("replicates" not in part for part in narrowed.source["parts"])
    assert "replicates" in spec.source["parts"][0]


def test_supplied_library_replaces_learning_and_the_live_library(tmp_path, sessions, serving, bench):
    learning, recorder, learn_sink, _ = _learning(tmp_path, [])
    library = _frozen(tmp_path)

    experiment_id = _run(
        tmp_path, _snapshot(tmp_path), FakeSink(), spec=_combined_spec(tmp_path), learning=learning, library=library
    )

    assert recorder.calls == 0
    assert learn_sink.experiments == []
    assert {s.spec.split for s in sessions} == {"test"}
    assert {s.spec.library_snapshot_id for s in sessions} == {"frozen-1"}
    assert experiment_id == experiment_id_for(load_spec(_combined_spec(tmp_path)), "frozen-1")


def test_supplied_library_is_still_checked_for_isolation(tmp_path, sessions, serving, bench):
    leaked = Skill("s1", "a", "d", "", "social science", "Gender hypothesis ONE", approved=True)
    library = SnapshotSkillSource(
        write_snapshot(
            tmp_path / "leaky.json",
            [leaked],
            FakeEmbedder(),
            snapshot_id="leaky",
            models=[OLLAMA_MODEL],
            hypothesis_sets={"gender_v1": HYPOTHESES},
            datasets=DATASETS,
        ),
        FakeEmbedder(),
    )

    with pytest.raises(ValueError, match="derived from test hypothesis"):
        _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_combined_spec(tmp_path), library=library)
    assert sessions == []


def test_supplied_library_must_be_frozen(tmp_path, sessions, serving, bench):
    with pytest.raises(ValueError, match="frozen snapshot"):
        _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_combined_spec(tmp_path), library=object())


def _bundle_learning(tmp_path):
    from popper_upgrade.db.bundle import BundleSink
    from popper_upgrade.eval.learning import bundle_learning, open_library

    embedder = FakeEmbedder()
    path = write_snapshot(
        tmp_path / "snap.json",
        [],
        embedder,
        models=[OLLAMA_MODEL],
        hypothesis_sets={"gender_v1": HYPOTHESES},
        datasets=DATASETS,
    )
    bundle = tmp_path / "bundle"
    library = open_library(path, bundle, embedder)
    sink = BundleSink(bundle, library)
    base = SnapshotSkillSource(path, embedder)
    return base, sink, bundle_learning(path, bundle, embedder, library, sink), bundle


def test_resume_after_the_freeze_keeps_the_frozen_library_and_experiment(tmp_path, monkeypatch, serving):
    state = {"retry": False, "stop": False, "tests": 0, "events": []}

    class Persisting:
        def __init__(self, spec, source, sink, curator, reviewer, extractor, trajectory_dir, **kwargs):
            self.spec = spec
            self.sink = sink

        def run(self):
            spec = self.spec
            run_id = self.sink.create_run(
                {
                    "experiment_id": spec.experiment_id,
                    "model_name": spec.model["name"],
                    "arm": spec.arm,
                    "task": spec.task,
                    "hypothesis": spec.hypothesis,
                    "replicate": spec.replicate,
                    "permuted": spec.permuted,
                    "library_snapshot_id": spec.library_snapshot_id,
                }
            )
            state["events"].append((spec.split, spec.hypothesis, spec.arm, spec.replicate, spec.permuted))
            if spec.split == "test":
                state["tests"] += 1
                state["stop"] = state["tests"] >= 4 and not state["retry"]
                self.sink.finish_run(run_id, {})
                return
            if spec.hypothesis == TRAIN_HYPOTHESES[1] and not state["retry"]:
                self.sink.finish_run(run_id, {"status": "error"})
                return
            skill = Skill(
                f"skill-{TRAIN_HYPOTHESES.index(spec.hypothesis)}",
                f"applicability of {spec.hypothesis}",
                "template",
                "",
                "social science",
                spec.hypothesis,
            )
            self.sink.add_candidate_skill(skill)
            self.sink.add_skill_decision(
                {"skill_id": skill.id, "trigger": "entry", "decider": "curator", "verdict": "approve", "reason": "r"}
            )
            self.sink.finish_run(run_id, {})

    monkeypatch.setattr(module, "RunSession", Persisting)
    monkeypatch.setattr(module, "build_agents", lambda kwargs, skills, reviewer, extractor: (object(),) * 3)
    spec = _learning_spec(tmp_path)

    base, sink, learning, bundle = _bundle_learning(tmp_path)
    first_id = _run(tmp_path, base, sink, spec=spec, learning=learning, stop_requested=lambda: state["stop"])
    frozen_before = (bundle / "frozen_snapshot.json").read_bytes()
    first_events = list(state["events"])
    state["events"].clear()
    state.update(retry=True, stop=False)

    base, sink, learning, bundle = _bundle_learning(tmp_path)
    second_id = _run(tmp_path, base, sink, spec=spec, learning=learning)

    assert second_id == first_id
    assert (bundle / "frozen_snapshot.json").read_bytes() == frozen_before
    assert ("train", TRAIN_HYPOTHESES[1], "skills_reviewer", 0, False) in state["events"]
    second_tests = [event for event in state["events"] if event[0] == "test"]
    assert second_tests
    assert not set(second_tests) & {event for event in first_events if event[0] == "test"}
    assert len(second_tests) + len([event for event in first_events if event[0] == "test"]) == 2 * (2 + 1) * 4
    experiments = [json.loads(line) for line in (bundle / "experiments.jsonl").read_text().splitlines()]
    assert len([row for row in experiments if row["spec"].get("phase") == "evaluate"]) == 1
    runs = [json.loads(line) for line in (bundle / "runs.jsonl").read_text().splitlines()]
    frozen_id = json.loads((bundle / "freeze.json").read_text())["snapshot_id"]
    assert {row["library_snapshot_id"] for row in runs if row["experiment_id"] == first_id} == {frozen_id}


VLLM_MODEL = {**OLLAMA_MODEL, "backend": "vllm", "environment": "cluster"}


def _dying_serving(monkeypatch, alive_for, detail=None):
    checks = []

    @contextlib.contextmanager
    def fake(model, log_dir=None):
        counter = UsageCounter()
        counter.port = 4321

        def alive():
            checks.append(1)
            return len(checks) <= alive_for

        counter.alive = alive
        if detail is not None:
            counter.upstream_error = lambda: detail
        yield counter

    monkeypatch.setattr(module, "serving_for", fake)
    return checks


def test_runs_stop_when_the_serving_process_has_exited(tmp_path, sessions, monkeypatch):
    checks = _dying_serving(monkeypatch, 3, "CUDA out of memory")

    with pytest.raises(module.ServerExited, match="no longer responding") as info:
        _run(tmp_path, _snapshot(tmp_path), FakeSink())

    assert len(sessions) == 3
    assert len(checks) == 4
    assert "after 3 runs" in str(info.value)
    assert "CUDA out of memory" in str(info.value)


def test_liveness_is_checked_before_the_first_learning_run_too(tmp_path, sessions, monkeypatch):
    _dying_serving(monkeypatch, 0)
    learning, recorder, _, _ = _learning(tmp_path, [])

    with pytest.raises(module.ServerExited):
        _run(tmp_path, _snapshot(tmp_path), FakeSink(), spec=_learning_spec(tmp_path), learning=learning)

    assert sessions == []
    assert recorder.calls == 0


def test_a_server_that_stays_up_does_not_interrupt_the_experiment(tmp_path, sessions, monkeypatch):
    _dying_serving(monkeypatch, 10**6)

    _run(tmp_path, _snapshot(tmp_path), FakeSink())

    assert len(sessions) == 2 * (2 + 1) * 4


def test_vllm_models_get_an_http_health_probe_by_default(tmp_path, sessions, monkeypatch):
    probed = []

    @contextlib.contextmanager
    def fake(model, log_dir=None):
        counter = UsageCounter()
        counter.port = 4321
        yield counter

    monkeypatch.setattr(module, "serving_for", fake)
    monkeypatch.setattr(module, "check_environment", lambda model, environment: None)
    monkeypatch.setattr(module, "server_responding", lambda port: probed.append(port) or False)
    embedder = FakeEmbedder()
    path = write_snapshot(
        tmp_path / "snap.json",
        [],
        embedder,
        models=[VLLM_MODEL],
        hypothesis_sets={"gender_v1": HYPOTHESES},
        datasets=DATASETS,
    )

    with pytest.raises(module.ServerExited):
        _run(tmp_path, SnapshotSkillSource(path, embedder), FakeSink())

    assert probed == [4321]
    assert sessions == []


def test_server_responding_tells_a_live_server_from_a_dead_one():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    assert module.server_responding(port, attempts=1, pause=0) is True
    server.shutdown()
    server.server_close()

    assert module.server_responding(port, attempts=2, pause=0) is False


def test_main_exits_with_a_distinct_code_and_a_plain_message(tmp_path, monkeypatch, capsys):
    from popper_upgrade.skills import embedding

    embedder = FakeEmbedder()
    snapshot = write_snapshot(tmp_path / "snap.json", [], embedder)
    monkeypatch.setattr(embedding, "Embedder", lambda: embedder)

    def dead(*args, **kwargs):
        raise module.ServerExited("the vLLM server is no longer responding; stopped before the next run")

    monkeypatch.setattr(module, "run_experiment", dead)

    code = module.main(
        ["--spec", "s.json", "--model", "m", "--source", f"snapshot:{snapshot}", "--sink", f"bundle:{tmp_path / 'b'}"]
    )

    assert code == module.EXIT_SERVER_EXITED == 4
    assert "no longer responding" in capsys.readouterr().err
