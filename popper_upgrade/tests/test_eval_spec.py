import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster.common import load_baseline
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.run_experiment import build_examples
from popper_upgrade.eval.spec import (
    check_snapshot_isolation,
    expand,
    expand_learning,
    load_spec,
    parse_spec,
    planned_counts,
    run_identity,
    schedule_order,
)
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import OLLAMA_MODEL, FakeEmbedder, write_snapshot

SPECS = Path(__file__).resolve().parents[1] / "eval" / "specs"
ARMS = ("baseline", "skills", "random_skills")


def _spec(**changes):
    data = {
        "name": "t",
        "models": ["llama3.1:8b"],
        "arms": list(ARMS),
        "source": {"kind": "gender_custom"},
        "train_tasks": ["train_task"],
        "test_tasks": ["test_task"],
        "replicates": 3,
        "permutation_seeds": [11, 12, 13],
        "prompt_versions": {"curator": "a", "reviewer": "b", "extractor": "c"},
    }
    data.update(changes)
    return parse_spec(data)


def _factory(seed):
    return lambda: ("loader", seed)


def _examples():
    base = {"task": "test_task", "domain": "social science", "factory": _factory}
    return [
        {**base, "hypothesis": "H1", "answer": False, "permuted": True},
        {**base, "hypothesis": "H1", "answer": None, "permuted": False, "replicates": 1},
    ]


def _expand(spec):
    return expand(spec, OLLAMA_MODEL, _examples(), {"llm": "llama3.1:8b"}, "laptop", "exp-1", "snap-1")


def test_expand_matches_arms_examples_replicates():
    runs = _expand(_spec())

    assert len(runs) == 3 * (3 + 1)
    expected = {(arm, True, replicate) for arm in ARMS for replicate in range(3)}
    expected |= {(arm, False, 0) for arm in ARMS}
    assert {(run.arm, run.permuted, run.replicate) for run in runs} == expected
    assert {run.split for run in runs} == {"test"}
    assert {run.learn for run in runs} == {False}
    assert {run.experiment_id for run in runs} == {"exp-1"}
    assert {run.library_snapshot_id for run in runs} == {"snap-1"}
    assert {run.popper_kwargs["domain"] for run in runs} == {"social science"}
    flags = {run.arm: (run.skills_enabled, run.reviewer_enabled) for run in runs}
    assert flags == {"baseline": (False, False), "skills": (True, False), "random_skills": (True, False)}


def test_permute_seed_identical_across_arms():
    runs = _expand(_spec())

    by_slot = {}
    for run in runs:
        by_slot.setdefault((run.permuted, run.replicate), set()).add(run.permute_seed)
        assert run.data_loader_factory() == ("loader", run.permute_seed)
    assert by_slot == {(True, 0): {11}, (True, 1): {12}, (True, 2): {13}, (False, 0): {None}}


def _many_examples(count=7):
    base = {"task": "test_task", "domain": "social science", "factory": _factory}
    examples = []
    for index in range(count):
        examples.append({**base, "hypothesis": f"H{index}", "answer": False, "permuted": True})
        examples.append({**base, "hypothesis": f"H{index}", "answer": True, "permuted": False, "replicates": 1})
    return examples


def _expand_many(spec, examples=None):
    examples = _many_examples() if examples is None else examples
    return expand(spec, OLLAMA_MODEL, examples, {"llm": "llama3.1:8b"}, "laptop", "exp-1", "snap-1")


def test_every_prefix_has_equal_arm_counts_and_items_are_visited_before_repeats():
    spec = _spec()
    runs = _expand_many(spec)
    arms = len(spec.arms)

    for size in range(len(runs) + 1):
        counts = [sum(1 for run in runs[:size] if run.arm == arm) for arm in spec.arms]
        assert max(counts) - min(counts) <= 1
    first_round = runs[: 14 * arms]
    assert {(run.hypothesis, run.permuted) for run in first_round} == {
        (f"H{i}", flag) for i in range(7) for flag in (True, False)
    }
    assert {run.replicate for run in first_round} == {0}
    assert {run.replicate for run in runs[14 * arms :]} == {1, 2}


def test_arms_run_back_to_back_for_each_item():
    spec = _spec()
    runs = _expand_many(spec)

    for start in range(0, len(runs[: 14 * 3]), 3):
        group = runs[start : start + 3]
        assert [run.arm for run in group] == list(spec.arms)
        assert len({(run.hypothesis, run.permuted, run.replicate) for run in group}) == 1


def test_item_order_is_seeded_stable_and_not_the_source_order():
    spec = _spec()
    examples = _many_examples()

    first = [run_identity(run) for run in _expand_many(spec, examples)]
    second = [run_identity(run) for run in _expand_many(spec, list(examples))]

    assert first == second
    natural = [(example["hypothesis"], example["permuted"]) for example in examples]
    shuffled = [(example["hypothesis"], example["permuted"]) for example in schedule_order(spec, examples)]
    assert shuffled != natural
    assert sorted(shuffled) == sorted(natural)
    renamed = [(e["hypothesis"], e["permuted"]) for e in schedule_order(_spec(name="other"), examples)]
    assert renamed != shuffled


def test_resuming_keeps_the_remaining_order():
    spec = _spec()
    full = [run_identity(run) for run in _expand_many(spec)]
    done = set(full[:23]) | set(full[40:45])

    remaining = [identity for identity in full if identity not in done]
    again = [run_identity(run) for run in _expand_many(spec) if run_identity(run) not in done]

    assert again == remaining
    assert len(set(full)) == len(full)


def test_planned_counts_follow_example_replicates():
    spec = _spec()

    assert planned_counts(spec, _many_examples(7)) == dict.fromkeys(ARMS, 7 * 3 + 7)
    assert planned_counts(spec, []) == dict.fromkeys(ARMS, 0)
    assert _expand_many(spec, []) == []


def test_snapshot_isolation_violation_raises(tmp_path):
    embedder = FakeEmbedder()
    skill = Skill("s1", "a", "d", "", "social science", "Some  Test hypothesis", approved=True)
    clean = Skill("s2", "b", "d", "", "social science", "train hypothesis", approved=False)
    snapshot = SnapshotSkillSource(write_snapshot(tmp_path / "s.json", [skill, clean], embedder), embedder)

    with pytest.raises(ValueError, match="skill s1 derived from test hypothesis"):
        check_snapshot_isolation(snapshot, {"some test hypothesis", "other"})
    check_snapshot_isolation(snapshot, {"unrelated"})


def test_rejected_skill_from_a_test_hypothesis_also_violates_isolation(tmp_path):
    embedder = FakeEmbedder()
    skill = Skill("s9", "b", "d", "", "social science", "T", approved=False)
    snapshot = SnapshotSkillSource(write_snapshot(tmp_path / "s.json", [skill], embedder), embedder)

    with pytest.raises(ValueError, match="skill s9"):
        check_snapshot_isolation(snapshot, {"T"})


@pytest.mark.parametrize(
    "changes",
    [
        {"arms": ["baseline", "nonsense"]},
        {"replicates": 4},
        {"permutation_seeds": [1, 1, 2]},
        {"train_tasks": ["test_task"]},
        {"source": {"kind": "other"}},
        {"prompt_versions": {"curator": "a"}},
        {"source": {"kind": "combined", "parts": []}},
        {"source": {"kind": "combined", "parts": [{"kind": "combined", "parts": [{"kind": "gender_custom"}]}]}},
        {"source": {"kind": "gender_custom", "replicates": {"original": 2}}},
        {"source": {"kind": "gender_custom", "sample": {"originals": 3, "seed": 1}}},
        {"source": {"kind": "discoverybench", "sample": {"originals": 0, "seed": 1}}},
        {"source": {"kind": "gender_custom", "replicates": {"original": 1, "permuted": 4}}},
    ],
)
def test_invalid_specs_are_rejected(changes):
    with pytest.raises(ValueError):
        _spec(**changes)


@pytest.mark.parametrize("name", ["gender_permutation_v1", "discoverybench_v1"])
def test_shipped_specs_are_valid(name):
    spec = load_spec(SPECS / f"{name}.json")

    assert spec.name == name
    assert not set(spec.train_tasks) & set(spec.test_tasks)
    assert len(spec.permutation_seeds) >= spec.replicates
    assert set(spec.arms) == {"baseline", "skills", "skills_reviewer", "random_skills", "wrong_domain_skills"}
    assert json.loads((SPECS / f"{name}.json").read_text(encoding="utf-8"))["popper"] == spec.popper


def test_part_replicates_must_be_covered_by_permutation_seeds():
    part = {"kind": "gender_custom", "replicates": {"original": 1, "permuted": 4}}

    assert _spec(source=part, replicates=3, permutation_seeds=[1, 2, 3, 4]).replicates == 3


def test_size_is_kept_only_when_set():
    assert "size" not in _spec().to_dict()
    assert _spec(size="lite").to_dict()["size"] == "lite"


REGISTRY = Path(__file__).resolve().parents[1] / "registry"
QWEN_ARMS = ["baseline", "skills", "skills_reviewer"]


def _qwen_round():
    spec = load_spec(SPECS / "qwen_round.json")
    datasets = json.loads((REGISTRY / "datasets.json").read_text(encoding="utf-8"))
    hypotheses = json.loads((REGISTRY / "hypotheses.json").read_text(encoding="utf-8"))
    registry = {
        "hypothesis_sets": hypotheses,
        "datasets": [{"name": d["name"], "permutable_columns": d["permutable_columns"]} for d in datasets],
    }
    test = build_examples(spec, SimpleNamespace(registry=registry), ".")
    train = build_examples(spec, SimpleNamespace(registry=registry), ".", tasks=spec.train_tasks, sampled=False)
    return spec, hypotheses["gender_v1"], test, [example for example in train if not example["permuted"]]


def test_qwen_round_spec_uses_the_baseline_hypotheses_and_parameters():
    config, _ = load_baseline()
    spec, hypotheses, _, _ = _qwen_round()
    models = {m["name"]: m for m in json.loads((REGISTRY / "models.json").read_text(encoding="utf-8"))}

    assert [h["text"] for h in hypotheses] == config.HYPOTHESES
    assert spec.models == ["qwen-large", "gpt-6-sol", "claude-sonnet-5-5"]
    assert all(name in models for name in spec.models)
    assert spec.arms == QWEN_ARMS
    assert spec.popper == {
        "alpha": config.ALPHA,
        "max_num_of_tests": config.MAX_TESTS,
        "max_retry": config.MAX_RETRY,
        "time_limit": config.TIME_LIMIT,
    }
    assert spec.source["domain"] == "social science"
    assert spec.test_tasks == ["gender_bias"]
    assert spec.train_tasks == ["gender_bias_train"]


def test_qwen_round_expands_to_90_evaluation_and_5_learning_runs():
    spec, hypotheses, test, train = _qwen_round()
    assert len(spec.permutation_seeds) == 5

    for model in spec.models:
        runs = expand(spec, {"name": model}, test, {}, "cluster", "exp", "snap")
        assert len(runs) == 90
        per_slot = Counter((run.hypothesis, run.arm, run.permuted) for run in runs)
        assert set(per_slot.values()) == {5, 1} and len(per_slot) == 5 * 3 * 2
        covered = {(run.hypothesis, run.arm) for run in runs}
        assert covered == {(h["text"], arm) for h in hypotheses for arm in QWEN_ARMS}
        assert Counter(run.permuted for run in runs) == {False: 75, True: 15}
        assert {run.replicate for run in runs if not run.permuted} == set(range(5))
        assert {run.replicate for run in runs if run.permuted} == {0}
        assert {run.permute_seed for run in runs if run.permuted} == {201}
        assert {run.popper_kwargs["domain"] for run in runs} == {"social science"}
        assert {run.learn for run in runs} == {False}
        assert {run.configure_kwargs["aggregate_test"] for run in runs} == {"E-value"}
    assert planned_counts(spec, test) == dict.fromkeys(QWEN_ARMS, 30)

    learning = expand_learning(spec, {"name": "qwen-large"}, train, {}, "cluster", "learn", "snap")
    assert len(learning) == 5
    assert {run.hypothesis for run in learning} == set(spec.source["train_hypotheses"])
    assert {run.split for run in learning} == {"train"}
    assert {run.arm for run in learning} == {"skills_reviewer"}
    assert {run.learn for run in learning} == {True}
    assert {run.popper_kwargs["domain"] for run in learning} == {"social science"}
    assert not {run.hypothesis for run in learning} & {h["text"] for h in hypotheses}


def test_qwen_round_shares_its_training_set_with_gender_permutation_v2():
    qwen = json.loads((SPECS / "qwen_round.json").read_text(encoding="utf-8"))
    v2 = json.loads((SPECS / "gender_permutation_v2.json").read_text(encoding="utf-8"))

    assert qwen["source"]["train_hypotheses"] == v2["source"]["train_hypotheses"]
    assert qwen["prompt_versions"] == v2["prompt_versions"]
