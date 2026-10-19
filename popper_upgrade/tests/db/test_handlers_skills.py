import uuid

import pytest

from popper_upgrade.app.handlers import skills as handlers
from popper_upgrade.app.handlers.admin import upsert_model
from popper_upgrade.skills.proposal_agent import SkillAugmentedProposalAgent
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeCurator, FakeProposalAgent, run_config

pytestmark = pytest.mark.db


def _add(ctx, db, applicability, approved=True, domain="social science"):
    skill = Skill(
        id=str(uuid.uuid4()),
        applicability=applicability,
        design_template="compare group means",
        preconditions="two groups",
        domain=domain,
        source_hypothesis="",
    )
    repo = ctx.skills(db)
    skill_id = repo.add_candidate_skill(skill)
    if approved is not None:
        repo.set_approval(skill_id, approved, "curator", "seed", "entry")
    return skill_id


def _skill_events(db, skill_id):
    return db.execute("select action from audit.events where target_id = %s", (skill_id,)).fetchall()


def _curator_ctx(make_ctx, db, admin_user, curator):
    ctx = make_ctx(curator_factory=lambda name: curator, default_curator_model="claude-sonnet-4-5")
    upsert_model(ctx, admin_user, "claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5", "large", "laptop")
    return ctx


def test_user_cannot_delete_or_set_library_approval(ctx, db, plain_user):
    skill_id = _add(ctx, db, "approved skill")
    with pytest.raises(PermissionError):
        handlers.delete_skill(ctx, plain_user, skill_id)
    with pytest.raises(PermissionError):
        handlers.set_library_approval(ctx, plain_user, skill_id, False, "no")
    with pytest.raises(PermissionError):
        handlers.edit_skill(ctx, plain_user, skill_id, applicability="changed")
    with pytest.raises(PermissionError):
        handlers.restore_skill(ctx, plain_user, skill_id)
    assert ctx.skills(db).get(skill_id).approved is True
    assert ctx.skills(db).get(skill_id).applicability == "approved skill"
    assert _skill_events(db, skill_id) == []


def test_admin_delete_writes_full_before_to_audit(ctx, db, admin_user, plain_user):
    skill_id = _add(ctx, db, "to delete")
    handlers.flag_skill(ctx, plain_user, skill_id, "looks wrong")
    handlers.delete_skill(ctx, admin_user, skill_id)
    assert ctx.skills(db).get(skill_id) is None
    event = db.execute(
        "select user_id::text as user_id, before from audit.events where action = 'skill.delete' and target_id = %s",
        (skill_id,),
    ).fetchone()
    assert event["user_id"] == admin_user.id
    assert event["before"]["id"] == skill_id
    assert event["before"]["applicability"] == "to delete"
    assert event["before"]["design_template"] == "compare group means"
    assert event["before"]["approved"] is True
    assert event["before"]["flagged"] is True
    assert "embedding" not in event["before"]


def test_admin_edit_restore_and_approval_are_audited(ctx, db, admin_user):
    skill_id = _add(ctx, db, "original", approved=False)
    handlers.restore_skill(ctx, admin_user, skill_id)
    assert ctx.skills(db).get(skill_id).approved is True
    handlers.edit_skill(ctx, admin_user, skill_id, applicability="edited")
    assert ctx.skills(db).get(skill_id).applicability == "edited"
    handlers.set_library_approval(ctx, admin_user, skill_id, False, "too vague")
    detail = handlers.skill_detail(ctx, admin_user, skill_id)
    assert detail["approved"] is False
    assert [d["trigger"] for d in detail["decisions"]] == ["entry", "admin", "admin"]
    assert {event["action"] for event in _skill_events(db, skill_id)} == {"skill.set_approval", "skill.edit"}


def test_proposed_skill_is_pending_then_decided(make_ctx, db, admin_user, plain_user):
    pending_ctx = make_ctx()
    skill_id = handlers.propose_skill(pending_ctx, plain_user, "first idea", "do a t-test", "", "social science")
    assert pending_ctx.skills(db).get(skill_id).approved is None
    assert [row["id"] for row in handlers.list_skills(pending_ctx, plain_user, "pending")] == [skill_id]
    proposer = db.execute("select proposed_by::text as p from skills.skills where id = %s", (skill_id,)).fetchone()
    assert proposer["p"] == plain_user.id

    curator = FakeCurator()
    ctx = _curator_ctx(make_ctx, db, admin_user, curator)
    decided_id = handlers.propose_skill(ctx, plain_user, "second idea", "do a chi-square", "", "social science")
    assert ctx.skills(db).get(decided_id).approved is True
    assert [c.id for c in curator.candidate_calls] == [decided_id]
    detail = handlers.skill_detail(ctx, plain_user, decided_id)
    assert [(d["trigger"], d["decider"], d["verdict"]) for d in detail["decisions"]] == [
        ("entry", "curator", "approve")
    ]
    assert decided_id not in {row["id"] for row in handlers.list_skills(ctx, plain_user, "pending")}


def test_curator_failure_leaves_proposal_pending(make_ctx, db, admin_user, plain_user):
    ctx = _curator_ctx(make_ctx, db, admin_user, FakeCurator(candidate_error=RuntimeError("down")))
    skill_id = handlers.propose_skill(ctx, plain_user, "an idea", "do a t-test", "", "social science")
    assert ctx.skills(db).get(skill_id).approved is None


def test_status_filters(ctx, db, plain_user):
    approved = _add(ctx, db, "approved one")
    rejected = _add(ctx, db, "rejected one", approved=False, domain="biology")
    pending = _add(ctx, db, "pending one", approved=None)
    handlers.flag_skill(ctx, plain_user, approved, "check this")

    def ids(**kwargs):
        return {row["id"] for row in handlers.list_skills(ctx, plain_user, **kwargs)}

    assert ids() == {approved, rejected, pending}
    assert ids(status="approved") == {approved}
    assert ids(status="rejected") == {rejected}
    assert ids(status="pending") == {pending}
    assert ids(status="flagged") == {approved}
    assert ids(domain="biology") == {rejected}
    assert ids(text="pending") == {pending}
    row = next(r for r in handlers.list_skills(ctx, plain_user) if r["id"] == approved)
    assert row["status"] == "approved"
    assert row["flagged"] is True
    assert row["win_rate"] is None
    with pytest.raises(ValueError):
        handlers.list_skills(ctx, plain_user, status="bogus")


def test_preview_matches_proposal_agent_retrieval(ctx, db, plain_user):
    hypothesis = "women are promoted less often than men"
    match = _add(ctx, db, hypothesis)
    _add(ctx, db, "an unrelated strategy about plants")
    _add(ctx, db, hypothesis + " rejected", approved=False)
    agent = SkillAugmentedProposalAgent(FakeProposalAgent(), ctx.skills(db))
    agent.go(hypothesis)
    preview = handlers.retrieval_preview(ctx, plain_user, hypothesis)
    assert [row["id"] for row in preview] == agent.last_retrieved_ids == [match]
    assert preview[0]["similarity"] == pytest.approx(agent.last_retrieved[0][1])


def test_skill_detail_reports_runs_used_in(ctx, db, plain_user):
    skill_id = _add(ctx, db, "used skill")
    run_id = ctx.runs(db).create_run(run_config())
    ctx.runs(db).add_run_skills(run_id, [{"skill_id": skill_id, "role": "injected", "round": 1, "similarity": 0.9}])
    detail = handlers.skill_detail(ctx, plain_user, skill_id)
    assert detail["runs_used_in"] == [run_id]
