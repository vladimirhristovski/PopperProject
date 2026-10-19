import json
import uuid

import pytest

from popper_upgrade.app.handlers import runs as handlers
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import run_config

pytestmark = pytest.mark.db


def _finished_run(ctx, db, tracked_stat, **config):
    repo = ctx.runs(db)
    run_id = repo.create_run(run_config(**config))
    path = ctx.trajectory_dir / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory = {
        "log": {"designer": ["design text"], "executor": []},
        "tracked_tests": ["t"] * len(tracked_stat),
        "tracked_stat": tracked_stat,
        "res": False,
        "res_stat": 1.0,
        "parsed_result": {},
        "last_message": "done",
    }
    path.write_text(json.dumps(trajectory), encoding="utf-8")
    repo.finish_run(run_id, {"res": False, "res_stat": 1.0, "n_tests": len(tracked_stat), "trajectory_path": str(path)})
    return run_id


def test_list_runs_filters(ctx, db, plain_user):
    first = _finished_run(ctx, db, [0.5], arm="baseline", model_name="m1")
    second = _finished_run(ctx, db, [0.5], arm="skills", model_name="m2")
    ids = {run["id"] for run in handlers.list_runs(ctx, plain_user, {})}
    assert {first, second} <= ids
    assert [r["id"] for r in handlers.list_runs(ctx, plain_user, {"arm": "skills", "model": "m2"})] == [second]
    assert handlers.list_runs(ctx, plain_user, {"model": "m1", "arm": "skills"}) == []
    assert [r["id"] for r in handlers.list_runs(ctx, plain_user, {"arm": "baseline", "status": ""})] == [first]


def test_run_detail_groups_skills_and_loads_trajectory(ctx, db, plain_user):
    repo = ctx.skills(db)
    skill_id = repo.add_candidate_skill(
        Skill(
            id=str(uuid.uuid4()),
            applicability="grouped skill",
            design_template="t",
            preconditions="",
            domain="",
            source_hypothesis="",
        )
    )
    run_id = _finished_run(ctx, db, [0.04, 0.2])
    runs = ctx.runs(db)
    runs.add_run_skills(
        run_id,
        [
            {"skill_id": skill_id, "role": "curator_approved", "round": None, "similarity": None},
            {"skill_id": skill_id, "role": "injected", "round": 1, "similarity": 0.8},
        ],
    )
    runs.add_design_reviews(
        run_id, [{"round": 1, "attempt": 1, "proposal": "p", "approved": True, "reason": "ok", "overridden": False}]
    )
    detail = handlers.run_detail(ctx, plain_user, run_id)
    assert [s["applicability"] for s in detail["skills"]["curator_approved"]] == ["grouped skill"]
    assert [s["round"] for s in detail["skills"]["injected"]] == [1]
    assert detail["skills"]["curator_rejected"] == []
    assert [r["reason"] for r in detail["design_reviews"]] == ["ok"]
    assert detail["trajectory"]["tracked_stat"] == [0.04, 0.2]
    assert len(handlers.evalue_series(detail["trajectory"]["tracked_stat"], detail["run"]["alpha"])) == 2


def test_run_detail_with_zero_completed_tests(ctx, db, plain_user):
    run_id = _finished_run(ctx, db, [])
    detail = handlers.run_detail(ctx, plain_user, run_id)
    assert detail["trajectory"]["tracked_stat"] == []
    assert handlers.evalue_series(detail["trajectory"]["tracked_stat"], detail["run"]["alpha"]) == []


def test_run_detail_without_trajectory_or_outside_root(ctx, db, plain_user, tmp_path):
    run_id = ctx.runs(db).create_run(run_config())
    assert handlers.run_detail(ctx, plain_user, run_id)["trajectory"] is None
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    ctx.runs(db).finish_run(run_id, {"trajectory_path": str(outside)})
    assert handlers.run_detail(ctx, plain_user, run_id)["trajectory"] is None
    with pytest.raises(KeyError):
        handlers.run_detail(ctx, plain_user, str(uuid.uuid4()))
