import threading
import uuid

import pytest

from popper_upgrade.db.connection import connect
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, run_config

pytestmark = pytest.mark.db


def make_repo(conn):
    return RunRepository(conn, SkillRepository(conn, FakeEmbedder()))


def test_run_lifecycle_round_trip(db):
    repo = make_repo(db)
    skill_id = repo.add_candidate_skill(
        Skill(
            id=str(uuid.uuid4()),
            applicability="A",
            design_template="t",
            preconditions="",
            domain="",
            source_hypothesis="",
        )
    )

    run_id = repo.create_run(run_config())
    assert repo.get_run(run_id)["status"] == "running"
    repo.add_run_skills(run_id, [{"skill_id": skill_id, "role": "injected", "round": 1, "similarity": 0.8}])
    repo.add_run_skills(run_id, [{"skill_id": skill_id, "role": "curator_approved", "round": None, "similarity": None}])
    repo.add_design_reviews(
        run_id,
        [{"round": 1, "attempt": 1, "proposal": "p", "approved": False, "reason": "r", "overridden": False}],
    )
    decision_id = repo.add_skill_decision(
        {
            "skill_id": skill_id,
            "run_id": run_id,
            "trigger": "pre_run",
            "decider": "curator",
            "verdict": "approve",
            "reason": "ok",
        }
    )
    repo.finish_run(run_id, {"res": True, "res_stat": 12.5, "n_tests": 2, "parsed_result": {"conclusion": True}})

    run = repo.get_run(run_id)
    assert run["status"] == "completed"
    assert run["res"] is True
    assert run["res_stat"] == 12.5
    assert run["parsed_result"] == {"conclusion": True}
    assert run["datasets"] == ["winobias"]
    assert run["finished_at"] is not None
    assert {(r["role"], r["round"]) for r in repo.run_skills(run_id)} == {("injected", 1), ("curator_approved", None)}
    assert repo.design_reviews(run_id)[0]["reason"] == "r"
    assert decision_id
    assert [r["id"] for r in repo.list_runs({"arm": "baseline"})] == [run_id]
    assert repo.list_runs({"arm": "other"}) == []


def test_finish_run_rejects_unknown_columns(db):
    repo = make_repo(db)
    run_id = repo.create_run(run_config())

    with pytest.raises(ValueError):
        repo.finish_run(run_id, {"bogus": 1})


def test_concurrent_writers_both_persist(dsn):
    run_ids = []
    errors = []

    def worker(label):
        conn = connect(dsn)
        try:
            repo = make_repo(conn)
            run_id = repo.create_run(run_config(hypothesis=label))
            repo.finish_run(run_id, {"res": True})
            conn.commit()
            run_ids.append(run_id)
        except Exception as e:
            errors.append(e)
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(f"concurrent-{i}",)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check = connect(dsn)
    try:
        stored = check.execute(
            "select id::text as id from runs.runs where id::text = any(%s::text[])", (run_ids,)
        ).fetchall()
    finally:
        check.execute("delete from runs.runs where hypothesis like 'concurrent-%'")
        check.commit()
        check.close()

    assert errors == []
    assert len(run_ids) == 2
    assert {r["id"] for r in stored} == set(run_ids)


def _decision(skill_id, trigger, verdict, run_id=None):
    return {
        "skill_id": skill_id,
        "run_id": run_id,
        "trigger": trigger,
        "decider": "curator",
        "verdict": verdict,
        "reason": f"{trigger} {verdict}",
    }


def test_entry_decision_sets_library_flag_and_pre_run_does_not(db):
    skills = SkillRepository(db, FakeEmbedder())
    repo = RunRepository(db, skills)
    skill_id = repo.add_candidate_skill(
        Skill(
            id=str(uuid.uuid4()),
            applicability="E",
            design_template="t",
            preconditions="",
            domain="",
            source_hypothesis="",
        )
    )
    run_id = repo.create_run(run_config())

    repo.add_skill_decision(_decision(skill_id, "pre_run", "reject", run_id))
    assert skills.get(skill_id).approved is None

    repo.add_skill_decision(_decision(skill_id, "entry", "approve"))
    stored = skills.get(skill_id)
    assert (stored.approved, stored.decided_by, stored.decision_reason) == (True, "curator", "entry approve")

    repo.add_skill_decision(_decision(skill_id, "pre_run", "reject", run_id))
    assert skills.get(skill_id).approved is True


def test_run_cost_is_stored_and_listed_per_model(db):
    repo = make_repo(db)
    first = repo.create_run({**run_config(), "model_name": "paid-model"})
    second = repo.create_run({**run_config(), "model_name": "paid-model"})
    other = repo.create_run({**run_config(), "model_name": "other-model"})
    repo.create_run({**run_config(), "model_name": "paid-model"})
    repo.finish_run(first, {"tokens_in": 10, "cost_usd": 0.25})
    repo.finish_run(second, {"status": "error", "cost_usd": 0.5})
    repo.finish_run(other, {"cost_usd": 9.0})

    assert repo.get_run(first)["cost_usd"] == 0.25
    assert sorted(repo.model_costs("paid-model"), key=lambda row: row[0]) == [
        ("completed", 0.25, "baseline"),
        ("error", 0.5, "baseline"),
        ("running", None, "baseline"),
    ]
    assert repo.model_costs("never-ran") == []


def test_run_cost_must_not_be_negative(db):
    import psycopg

    repo = make_repo(db)
    run_id = repo.create_run(run_config())

    with pytest.raises(psycopg.errors.CheckViolation):
        repo.finish_run(run_id, {"cost_usd": -1.0})
