import json
from pathlib import Path

import pytest

from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.spec import check_snapshot_isolation, expand, load_spec, parse_spec
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
