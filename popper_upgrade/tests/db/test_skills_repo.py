import uuid

import pytest

from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder

pytestmark = pytest.mark.db


def _skill(applicability, template="template", domain="social science"):
    return Skill(
        id=str(uuid.uuid4()),
        applicability=applicability,
        design_template=template,
        preconditions="",
        domain=domain,
        source_hypothesis=applicability,
    )


@pytest.fixture
def repo(db):
    return SkillRepository(db, FakeEmbedder())


def _decisions(db, skill_id):
    return db.execute(
        "select verdict, decider, trigger from skills.skill_decisions where skill_id = %s order by created_at",
        (skill_id,),
    ).fetchall()


def _approve(repo, text):
    sid = repo.add_candidate_skill(_skill(text))
    repo.set_approval(sid, True, "curator", "fine", "entry")
    return sid


def test_retrieve_returns_only_approved_by_default(repo):
    approved = _approve(repo, "A")
    rejected = repo.add_candidate_skill(_skill("B"))
    repo.set_approval(rejected, False, "curator", "bad", "entry")
    repo.add_candidate_skill(_skill("C"))

    results = repo.retrieve_scored("A", 5)

    assert [skill.id for skill, _ in results] == [approved]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)


def test_retrieve_respects_allowed_ids_and_empty_set(repo):
    first = _approve(repo, "A")
    second = _approve(repo, "B")

    assert repo.retrieve_scored("q", 2, allowed_ids=set()) == []
    assert [s.id for s, _ in repo.retrieve_scored("q", 5, allowed_ids={first}, min_similarity=-1.0)] == [first]
    both = repo.retrieve_scored("q", 5, allowed_ids={first, second}, min_similarity=-1.0)
    assert {s.id for s, _ in both} == {first, second}


def test_retrieve_on_empty_library_returns_empty(repo):
    assert repo.retrieve_scored("q", 3) == []


def test_retrieve_respects_min_similarity(repo):
    _approve(repo, "A")

    assert repo.retrieve_scored("unrelated", 3, min_similarity=0.9) == []


def test_rejected_skill_blocks_near_duplicate_candidate(repo):
    sid = repo.add_candidate_skill(_skill("A"))
    repo.set_approval(sid, False, "curator", "bad", "entry")

    assert repo.add_candidate_skill(_skill("A")) is None


def test_pending_skill_blocks_duplicate_candidate(repo):
    assert repo.add_candidate_skill(_skill("A")) is not None
    assert repo.add_candidate_skill(_skill("A")) is None


def test_set_approval_writes_history(repo, db):
    sid = repo.add_candidate_skill(_skill("A"))

    repo.set_approval(sid, False, "curator", "bad", "entry")
    repo.set_approval(sid, True, "human", "actually fine", "admin")

    assert [d["verdict"] for d in _decisions(db, sid)] == ["reject", "approve"]
    skill = repo.get(sid)
    assert skill.approved is True
    assert skill.decided_by == "human"
    assert skill.decision_reason == "actually fine"


def test_ordering_ignores_times_passed(repo):
    low = _approve(repo, "A")
    high = _approve(repo, "B")
    before = [s.id for s, _ in repo.retrieve_scored("A", 2)]

    for _ in range(100):
        repo.record_usage(low, passed=True)
    after = [s.id for s, _ in repo.retrieve_scored("A", 2)]

    assert before == after == [low, high]
    assert repo.get(low).times_passed == 100


def test_similar_decisions_returns_rejected_with_reasons(repo):
    rejected = repo.add_candidate_skill(_skill("A"))
    repo.set_approval(rejected, False, "curator", "uses a column that does not exist", "entry")
    _approve(repo, "B")

    examples = repo.similar_decisions("A", k=5)

    assert examples[0].skill.id == rejected
    assert examples[0].verdict == "reject"
    assert examples[0].reason == "uses a column that does not exist"
    assert examples[0].decider == "curator"
    assert {e.verdict for e in examples} == {"reject", "approve"}


def test_search_filters(repo):
    pending = repo.add_candidate_skill(_skill("alpha topic", domain="biology"))
    approved = _approve(repo, "beta topic")

    assert {s.id for s in repo.search()} == {pending, approved}
    assert [s.id for s in repo.search(approved=None)] == [pending]
    assert [s.id for s in repo.search(approved=True)] == [approved]
    assert [s.id for s in repo.search(text="alpha")] == [pending]
    assert [s.id for s in repo.search(domain="biology")] == [pending]
    assert repo.search(flagged=True) == []


def test_update_content_reembeds_and_rehashes(repo):
    sid = _approve(repo, "A")
    before = repo.get(sid).content_hash

    repo.update_content(sid, applicability="B")

    assert repo.get(sid).content_hash != before
    assert [s.id for s, _ in repo.retrieve_scored("B", 1)] == [sid]
    with pytest.raises(ValueError):
        repo.update_content(sid, approved=True)


def test_flag_marks_skill(repo, db):
    sid = _approve(repo, "A")
    user = str(uuid.uuid4())
    db.execute("insert into auth.users (id, username, password_hash, role) values (%s, 'u', 'x', 'user')", (user,))

    repo.flag(sid, user, "looks wrong")

    assert repo.search(flagged=True)[0].id == sid
    assert db.execute("select count(*) as n from skills.skill_flags").fetchone()["n"] == 1
