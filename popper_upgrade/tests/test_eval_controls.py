import pytest

from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.controls import (
    RandomSkillSource,
    WrongDomainSkillSource,
    flags_for_arm,
    source_for_arm,
)
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, write_snapshot


def _skill(skill_id, domain, approved=True):
    return Skill(skill_id, f"applicability {skill_id}", "design", "", domain, f"source {skill_id}", approved=approved)


def _base(tmp_path, skills):
    embedder = FakeEmbedder()
    return SnapshotSkillSource(write_snapshot(tmp_path / "s.json", skills, embedder), embedder)


def _library(tmp_path):
    skills = [_skill(f"s{i}", "social science" if i % 2 else "biology") for i in range(8)]
    return _base(tmp_path, [*skills, _skill("rejected", "biology", approved=False)])


def test_random_source_deterministic_per_seed(tmp_path):
    base = _library(tmp_path)

    first = RandomSkillSource(base, 5).retrieve_scored("query", 3)
    again = RandomSkillSource(base, 5).retrieve_scored("query", 3)
    other = RandomSkillSource(base, 6).retrieve_scored("query", 3)

    assert [skill.id for skill, _ in first] == [skill.id for skill, _ in again]
    assert [skill.id for skill, _ in first] != [skill.id for skill, _ in other]
    assert all(similarity is None for _, similarity in first)
    assert len(first) == 3
    assert "rejected" not in {skill.id for skill, _ in first}


def test_random_source_respects_allowed_ids(tmp_path):
    source = RandomSkillSource(_library(tmp_path), 1)

    picked = source.retrieve_scored("query", 5, allowed_ids={"s1", "s2"})

    assert {skill.id for skill, _ in picked} == {"s1", "s2"}
    assert source.retrieve_scored("query", 5, allowed_ids=set()) == []
    assert source.list_skills(True)
    assert source.embedding_model == "fake-embedder"


def test_wrong_domain_excludes_run_domain(tmp_path):
    source = WrongDomainSkillSource(_library(tmp_path), "Social Science")

    assert {skill.domain for skill in source.list_skills(True)} == {"biology"}
    scored = source.retrieve_scored("query", 10, min_similarity=-1.0)
    assert scored
    assert {skill.domain for skill, _ in scored} == {"biology"}
    assert source.retrieve_scored("query", 10, allowed_ids={"s1", "s3"}, min_similarity=-1.0) == []
    assert source.degenerate is False


def test_wrong_domain_degenerate_flag(tmp_path):
    base = _base(tmp_path, [_skill("s1", "social science"), _skill("s2", "Social Science ")])
    source = WrongDomainSkillSource(base, "social science")

    assert source.degenerate is True
    assert source.retrieve_scored("query", 5, min_similarity=-1.0) == []
    assert source.list_skills(True) == []


@pytest.mark.parametrize(
    ("arm", "flags"),
    [
        ("baseline", (False, False)),
        ("skills", (True, False)),
        ("skills_reviewer", (True, True)),
        ("random_skills", (True, False)),
        ("wrong_domain_skills", (True, False)),
    ],
)
def test_flags_for_arm(arm, flags):
    assert flags_for_arm(arm) == flags


def test_source_for_arm_wraps_only_controls(tmp_path):
    base = _library(tmp_path)

    assert source_for_arm("skills", base, 1, "d") is base
    assert isinstance(source_for_arm("random_skills", base, 1, "d"), RandomSkillSource)
    assert isinstance(source_for_arm("wrong_domain_skills", base, 1, "d"), WrongDomainSkillSource)
