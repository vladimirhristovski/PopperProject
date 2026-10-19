import uuid
from dataclasses import replace

import psycopg
import pytest

from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, run_config

pytestmark = pytest.mark.db

TABLES = ("skills.skills", "runs.runs", "runs.run_skills", "runs.design_reviews", "skills.skill_decisions")


def _skill(text):
    return Skill(id="", applicability=text, design_template="t", preconditions="", domain="d", source_hypothesis=text)


@pytest.fixture
def library(db, tmp_path):
    embedder = FakeEmbedder()
    repo = SkillRepository(db, embedder)
    skill_id = repo.add_candidate_skill(replace(_skill("A"), id=str(uuid.uuid4())))
    repo.set_approval(skill_id, True, "curator", "fine", "entry")
    path = tmp_path / "snapshot.json"
    export_snapshot(db, path)
    return SnapshotSkillSource(path, embedder), skill_id


def _write_bundle(tmp_path, snapshot, skill_id):
    sink = BundleSink(tmp_path / "bundle", snapshot)
    run_id = sink.create_run(run_config(environment="cluster"))
    sink.add_run_skills(run_id, [{"skill_id": skill_id, "role": "injected", "round": 1, "similarity": 0.4}])
    sink.add_design_reviews(run_id, [{"round": 1, "attempt": 1, "proposal": "p", "approved": True, "reason": "r"}])
    sink.add_skill_decision(
        {
            "skill_id": skill_id,
            "run_id": run_id,
            "trigger": "pre_run",
            "decider": "curator",
            "verdict": "approve",
            "reason": "ok",
        }
    )
    candidate_id = sink.add_candidate_skill(_skill("B"))
    sink.finish_run(run_id, {"res": True, "res_stat": 7.0})
    return run_id, candidate_id


def _count(db, table):
    return db.execute(f"select count(*) as n from {table}").fetchone()["n"]


def test_import_bundle_twice_is_idempotent(db, tmp_path, library):
    snapshot, skill_id = library
    run_id, candidate_id = _write_bundle(tmp_path, snapshot, skill_id)
    before = {t: _count(db, t) for t in TABLES}

    first = import_bundle(db, tmp_path / "bundle")
    after_first = {t: _count(db, t) for t in before}
    second = import_bundle(db, tmp_path / "bundle")

    assert first == {
        "experiments": 0,
        "candidate_skills": 1,
        "runs": 1,
        "run_skills": 1,
        "design_reviews": 1,
        "skill_decisions": 1,
    }
    assert second == dict.fromkeys(first, 0)
    assert {t: _count(db, t) for t in before} == after_first
    assert after_first["skills.skills"] == before["skills.skills"] + 1
    run = db.execute(
        "select status, res, library_snapshot_id::text as snap from runs.runs where id = %s", (run_id,)
    ).fetchone()
    assert run == {"status": "completed", "res": True, "snap": snapshot.snapshot_id}
    candidate = db.execute("select approved from skills.skills where id = %s", (candidate_id,)).fetchone()
    assert candidate["approved"] is None
    events = db.execute("select count(*) as n from audit.events where action = 'bundle_import'").fetchone()["n"]
    assert events == 2


def test_import_rolls_back_on_bad_row(db, tmp_path, library):
    snapshot, skill_id = library
    run_id, candidate_id = _write_bundle(tmp_path, snapshot, skill_id)
    sink = BundleSink(tmp_path / "bundle", snapshot)
    sink.add_run_skills(run_id, [{"skill_id": str(uuid.uuid4()), "role": "injected", "round": 2, "similarity": 0.1}])

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        import_bundle(db, tmp_path / "bundle")

    assert db.execute("select count(*) as n from skills.skills where id = %s", (candidate_id,)).fetchone()["n"] == 0
    assert db.execute("select count(*) as n from runs.runs where id = %s", (run_id,)).fetchone()["n"] == 0


def test_import_applies_entry_decisions_once(db, tmp_path, library):
    snapshot, skill_id = library
    sink = BundleSink(tmp_path / "bundle", snapshot)
    approved_id = sink.add_candidate_skill(_skill("C"))
    rejected_id = sink.add_candidate_skill(_skill("D"))
    pending_id = sink.add_candidate_skill(_skill("E"))
    for candidate_id, verdict in ((approved_id, "approve"), (rejected_id, "reject")):
        sink.add_skill_decision(
            {"skill_id": candidate_id, "trigger": "entry", "decider": "curator", "verdict": verdict, "reason": verdict}
        )
    sink.add_skill_decision(
        {"skill_id": skill_id, "trigger": "pre_run", "decider": "curator", "verdict": "reject", "reason": "run only"}
    )

    import_bundle(db, tmp_path / "bundle")

    def state(sid):
        return db.execute("select approved, decided_by from skills.skills where id = %s", (sid,)).fetchone()

    assert state(approved_id) == {"approved": True, "decided_by": "curator"}
    assert state(rejected_id) == {"approved": False, "decided_by": "curator"}
    assert state(pending_id)["approved"] is None
    assert state(skill_id)["approved"] is True

    db.execute("update skills.skills set approved = false, decided_by = 'human' where id = %s", (approved_id,))
    import_bundle(db, tmp_path / "bundle")
    assert state(approved_id) == {"approved": False, "decided_by": "human"}
