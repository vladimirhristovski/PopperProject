import uuid
from dataclasses import asdict

from popper_upgrade.app.auth import require_admin
from popper_upgrade.db.audit import log_event
from popper_upgrade.models import check_environment, serving_for
from popper_upgrade.skills.curation import PRECEDENT_COUNT
from popper_upgrade.skills.schema import Skill

STATUSES = ("all", "approved", "rejected", "pending", "flagged")
RETRIEVAL_MIN_SIMILARITY = 0.35
STATUS_FILTERS = {
    "all": {},
    "approved": {"approved": True},
    "rejected": {"approved": False},
    "pending": {"approved": None},
    "flagged": {"flagged": True},
}


def _status(skill):
    if skill.approved is None:
        return "pending"
    return "approved" if skill.approved else "rejected"


def _win_rate(skill):
    return skill.times_passed / skill.times_used if skill.times_used else None


def _flagged_ids(conn):
    return {row["id"] for row in conn.execute("select id::text as id from skills.skills where flagged").fetchall()}


def list_skills(ctx, user, status="all", domain=None, text=None):
    if status not in STATUSES:
        raise ValueError(f"unknown status {status}")
    with ctx.conn() as conn:
        skills = ctx.skills(conn).search(text=text or None, domain=domain or None, **STATUS_FILTERS[status])
        flagged = _flagged_ids(conn)
    return [
        {
            "id": skill.id,
            "applicability": skill.applicability,
            "domain": skill.domain,
            "status": _status(skill),
            "flagged": skill.id in flagged,
            "times_used": skill.times_used,
            "win_rate": _win_rate(skill),
            "created_at": skill.created_at,
        }
        for skill in skills
    ]


def skill_detail(ctx, user, skill_id):
    with ctx.conn() as conn:
        skill = ctx.skills(conn).get(skill_id)
        if skill is None:
            raise KeyError(skill_id)
        decisions = conn.execute(
            "select id::text as id, run_id::text as run_id, trigger, decider, user_id::text as user_id, verdict, "
            "reason, model, created_at from skills.skill_decisions where skill_id = %s order by created_at, id",
            (skill_id,),
        ).fetchall()
        flags = conn.execute(
            "select f.id::text as id, u.username, f.reason, f.created_at, f.resolved from skills.skill_flags f "
            "join auth.users u on u.id = f.user_id where f.skill_id = %s order by f.created_at, f.id",
            (skill_id,),
        ).fetchall()
        runs_used = conn.execute(
            "select distinct run_id::text as run_id from runs.run_skills where skill_id = %s and role = 'injected' "
            "order by run_id",
            (skill_id,),
        ).fetchall()
        flagged = skill_id in _flagged_ids(conn)
    return {
        **asdict(skill),
        "status": _status(skill),
        "flagged": flagged,
        "win_rate": _win_rate(skill),
        "decisions": decisions,
        "flags": flags,
        "runs_used_in": [row["run_id"] for row in runs_used],
    }


def _review_entry(ctx, conn, skill_id, skill):
    if ctx.curator_factory is None or ctx.default_curator_model is None:
        return
    repo = ctx.skills(conn)
    registry = ctx.registry(conn)
    try:
        model = registry.model(ctx.default_curator_model)
        check_environment(model, "laptop")
        with serving_for(model):
            curator = ctx.curator_factory(ctx.default_curator_model)
            precedents = [
                example
                for example in repo.similar_decisions(skill.applicability, PRECEDENT_COUNT)
                if example.skill.id != skill_id
            ]
            decision = curator.review_candidate(skill, repo.list_skills(True), precedents)
    except Exception:
        return
    repo.set_approval(
        skill_id,
        decision.approved,
        "curator",
        decision.reason,
        "entry",
        model=curator.model_name,
        prompt_version_id=registry.prompt_version("curator", curator.prompt_text),
    )


def propose_skill(ctx, user, applicability, design_template, preconditions, domain):
    if not applicability.strip() or not design_template.strip():
        raise ValueError("applicability and design_template are required")
    skill = Skill(
        id=str(uuid.uuid4()),
        applicability=applicability.strip(),
        design_template=design_template.strip(),
        preconditions=(preconditions or "").strip(),
        domain=(domain or "").strip(),
        source_hypothesis="",
    )
    with ctx.conn() as conn:
        skill_id = ctx.skills(conn).add_candidate_skill(skill)
        if skill_id is None:
            raise ValueError("a near-duplicate skill already exists")
        conn.execute("update skills.skills set proposed_by = %s where id = %s", (user.id, skill_id))
        log_event(
            conn,
            user.id,
            "skill.propose",
            "skill",
            skill_id,
            after={"applicability": skill.applicability, "domain": skill.domain},
        )
    with ctx.conn() as conn:
        _review_entry(ctx, conn, skill_id, skill)
    return skill_id


def flag_skill(ctx, user, skill_id, reason):
    if not reason.strip():
        raise ValueError("a reason is required")
    with ctx.conn() as conn:
        repo = ctx.skills(conn)
        if repo.get(skill_id) is None:
            raise KeyError(skill_id)
        flag_id = repo.flag(skill_id, user.id, reason.strip())
        log_event(conn, user.id, "skill.flag", "skill", skill_id, after={"reason": reason.strip()})
    return flag_id


def set_library_approval(ctx, user, skill_id, approved, reason):
    require_admin(user)
    with ctx.conn() as conn:
        repo = ctx.skills(conn)
        before = repo.get(skill_id)
        if before is None:
            raise KeyError(skill_id)
        repo.set_approval(skill_id, approved, "human", reason, "admin", user_id=user.id)
        log_event(
            conn,
            user.id,
            "skill.set_approval",
            "skill",
            skill_id,
            before={"approved": before.approved},
            after={"approved": approved, "reason": reason},
        )


def edit_skill(ctx, user, skill_id, **fields):
    require_admin(user)
    with ctx.conn() as conn:
        repo = ctx.skills(conn)
        before = repo.get(skill_id)
        if before is None:
            raise KeyError(skill_id)
        repo.update_content(skill_id, **fields)
        log_event(
            conn,
            user.id,
            "skill.edit",
            "skill",
            skill_id,
            before={name: getattr(before, name) for name in fields},
            after=fields,
        )


def restore_skill(ctx, user, skill_id):
    set_library_approval(ctx, user, skill_id, True, "restored by admin")


def delete_skill(ctx, user, skill_id):
    require_admin(user)
    with ctx.conn() as conn:
        row = conn.execute(
            "select to_jsonb(s) - 'embedding' as row from skills.skills s where id = %s", (skill_id,)
        ).fetchone()
        if row is None:
            raise KeyError(skill_id)
        conn.execute("update skills.skills set parent_skill_id = null where parent_skill_id = %s", (skill_id,))
        conn.execute("delete from skills.skill_flags where skill_id = %s", (skill_id,))
        conn.execute("delete from skills.skill_decisions where skill_id = %s", (skill_id,))
        conn.execute("delete from runs.run_skills where skill_id = %s", (skill_id,))
        conn.execute("delete from skills.skills where id = %s", (skill_id,))
        log_event(conn, user.id, "skill.delete", "skill", skill_id, before=row["row"])


def retrieval_preview(ctx, user, hypothesis, k=2):
    with ctx.conn() as conn:
        scored = ctx.skills(conn).retrieve_scored(hypothesis, k, min_similarity=RETRIEVAL_MIN_SIMILARITY)
    return [
        {"id": skill.id, "applicability": skill.applicability, "similarity": similarity} for skill, similarity in scored
    ]
