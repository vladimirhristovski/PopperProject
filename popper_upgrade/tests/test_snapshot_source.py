import pytest

from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, write_snapshot


def _skill(skill_id, applicability, approved=True):
    return Skill(
        id=skill_id,
        applicability=applicability,
        design_template="t",
        preconditions="",
        domain="d",
        source_hypothesis=applicability,
        approved=approved,
    )


def _source(tmp_path, skills, decisions=()):
    embedder = FakeEmbedder()
    path = write_snapshot(tmp_path / "snap.json", skills, embedder, decisions)
    return SnapshotSkillSource(path, embedder)


def test_snapshot_rejects_embedding_mismatch(tmp_path):
    embedder = FakeEmbedder()
    path = write_snapshot(tmp_path / "snap.json", [_skill("s1", "A")], embedder, embedding_model="other-model")

    with pytest.raises(ValueError, match="other-model") as info:
        SnapshotSkillSource(path, embedder)

    assert "fake-embedder" in str(info.value)


def test_snapshot_source_excludes_unapproved(tmp_path):
    source = _source(
        tmp_path,
        [_skill("s1", "A"), _skill("s2", "B", approved=False), _skill("s3", "C", approved=None)],
    )

    assert [s.id for s, _ in source.retrieve_scored("A", 5)] == ["s1"]
    assert [s.id for s in source.list_skills()] == ["s1"]
    assert [s.id for s in source.list_skills(approved=False)] == ["s2"]
    assert [s.id for s, _ in source.retrieve_scored("A", 5, allowed_ids={"s2"})] == ["s2"]


def test_snapshot_source_orders_by_similarity_and_handles_empty(tmp_path):
    source = _source(tmp_path, [_skill("s1", "A"), _skill("s2", "B")])

    results = source.retrieve_scored("B", 2)

    assert [s.id for s, _ in results] == ["s2", "s1"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert source.retrieve_scored("B", 2, allowed_ids=set()) == []
    assert source.retrieve_scored("B", 2, min_similarity=0.99)[0][0].id == "s2"
    assert len(source.retrieve_scored("B", 2, min_similarity=0.99)) == 1
    assert _source(tmp_path, []).retrieve_scored("B", 2) == []


def test_snapshot_similar_decisions(tmp_path):
    decisions = [
        {"skill_id": "s1", "verdict": "reject", "reason": "bad column", "decider": "curator"},
        {"skill_id": "s2", "verdict": "approve", "reason": "ok", "decider": "curator"},
    ]
    source = _source(tmp_path, [_skill("s1", "A", approved=False), _skill("s2", "B")], decisions)

    examples = source.similar_decisions("A", k=1)

    assert [e.skill.id for e in examples] == ["s1"]
    assert examples[0].reason == "bad column"
    assert source.snapshot_id == "snap-1"
    assert source.registry["models"] == []
    assert source.max_similarity("A") == pytest.approx(1.0, abs=1e-5)
