import json

from popper_upgrade.db.bundle import BundleSink
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.learning import (
    FREEZE_INFO,
    FROZEN_SNAPSHOT,
    LearningSink,
    bundle_learning,
    freeze_bundle,
    open_library,
)
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, write_snapshot


def _skill(skill_id, text, hypothesis="train hypothesis", approved=None):
    return Skill(skill_id, text, "template", "", "social science", hypothesis, approved=approved)


def _setup(tmp_path, base_skills=()):
    embedder = FakeEmbedder()
    path = write_snapshot(tmp_path / "snapshot.json", list(base_skills), embedder)
    bundle = tmp_path / "bundle"
    library = open_library(path, bundle, embedder)
    sink = BundleSink(bundle, library)
    return path, bundle, embedder, library, sink


def _decision(skill_id, verdict, trigger="entry"):
    return {"skill_id": skill_id, "trigger": trigger, "decider": "curator", "verdict": verdict, "reason": "r"}


def _learn(sink, skill, verdict):
    learned = LearningSink(sink, sink._snapshot)
    assert learned.add_candidate_skill(skill) == skill.id
    learned.add_skill_decision(_decision(skill.id, verdict))
    return learned


def test_approved_candidate_joins_the_library_immediately(tmp_path):
    _, _, _, library, sink = _setup(tmp_path)

    _learn(sink, _skill("c1", "compare groups"), "approve")

    assert [skill.id for skill in library.list_skills(True)] == ["c1"]
    assert library.list_skills(True)[0].approved is True
    assert library.retrieve_scored("compare groups", 3)[0][0].id == "c1"


def test_rejected_candidate_stays_out_of_the_library(tmp_path):
    _, _, _, library, sink = _setup(tmp_path)

    _learn(sink, _skill("c1", "compare groups"), "reject")

    assert library.list_skills(True) == []


def test_library_is_rebuilt_from_the_bundle_on_resume(tmp_path):
    path, bundle, embedder, _, sink = _setup(tmp_path)
    _learn(sink, _skill("c1", "compare groups"), "approve")
    _learn(sink, _skill("c2", "test a trend over time"), "reject")

    reopened = open_library(path, bundle, embedder)

    assert [skill.id for skill in reopened.list_skills(True)] == ["c1"]
    assert reopened.snapshot_id == "snap-1"


def test_freeze_adds_learned_skills_with_their_own_id_and_hash(tmp_path):
    base = _skill("b1", "base skill", hypothesis="other", approved=True)
    path, bundle, embedder, _, sink = _setup(tmp_path, [base])
    _learn(sink, _skill("c1", "compare groups"), "approve")

    frozen = freeze_bundle(path, bundle, embedder)

    assert isinstance(frozen, SnapshotSkillSource)
    assert frozen.snapshot_id != "snap-1"
    assert {skill.id for skill in frozen.list_skills(True)} == {"b1", "c1"}
    info = json.loads((bundle / FREEZE_INFO).read_text(encoding="utf-8"))
    assert info["snapshot_id"] == frozen.snapshot_id
    assert info["base_snapshot_id"] == "snap-1"
    assert (info["candidates"], info["learned_skills"], info["approved_skills"]) == (1, 1, 2)
    assert info["empty_library"] is False
    assert len(info["content_hash"]) == 64
    assert sorted(info["skill_ids"]) == ["b1", "c1"]
    assert (bundle / FROZEN_SNAPSHOT).exists()


def test_freeze_is_deterministic_for_the_same_learned_skills(tmp_path):
    path, bundle, embedder, _, sink = _setup(tmp_path)
    _learn(sink, _skill("c1", "compare groups"), "approve")

    first = freeze_bundle(path, bundle, embedder)
    second = freeze_bundle(path, bundle, embedder)

    assert first.snapshot_id == second.snapshot_id


def test_zero_approved_skills_flags_an_empty_library(tmp_path):
    path, bundle, embedder, _, sink = _setup(tmp_path)
    _learn(sink, _skill("c1", "compare groups"), "reject")

    frozen = freeze_bundle(path, bundle, embedder)

    assert frozen.list_skills(True) == []
    info = json.loads((bundle / FREEZE_INFO).read_text(encoding="utf-8"))
    assert (info["candidates"], info["learned_skills"], info["approved_skills"]) == (1, 0, 0)
    assert info["empty_library"] is True


def test_freeze_with_no_candidates_at_all(tmp_path):
    path, bundle, embedder, _, _ = _setup(tmp_path)
    bundle.mkdir(exist_ok=True)

    frozen = freeze_bundle(path, bundle, embedder)

    assert frozen.list_skills(True) == []
    assert json.loads((bundle / FREEZE_INFO).read_text(encoding="utf-8"))["empty_library"] is True


def test_bundle_learning_exposes_source_sink_and_freeze(tmp_path):
    path, bundle, embedder, library, sink = _setup(tmp_path)

    learning = bundle_learning(path, bundle, embedder, library, sink)

    assert learning.source is library
    assert isinstance(learning.sink, LearningSink)
    assert learning.freeze().snapshot_id
