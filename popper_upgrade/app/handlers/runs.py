import json
import math
import sys
from pathlib import Path

from popper.agent import e_value_kappa_calibrator

FILTER_COLUMNS = {
    "model": "model_name",
    "arm": "arm",
    "status": "status",
    "environment": "environment",
    "experiment": "experiment_id",
}
SKILL_ROLES = (
    "curator_approved",
    "curator_rejected",
    "human_override_approve",
    "human_override_reject",
    "injected",
)


def list_runs(ctx, user, filters):
    columns = {
        FILTER_COLUMNS[name]: value for name, value in (filters or {}).items() if value and name in FILTER_COLUMNS
    }
    with ctx.conn() as conn:
        return ctx.runs(conn).list_runs(columns)


def evalue_series(tracked_stat, alpha):
    threshold = 1 / alpha
    series = []
    for i in range(1, len(tracked_stat) + 1):
        _, cum_e = e_value_kappa_calibrator(tracked_stat[:i], alpha)
        value = float(cum_e)
        if not math.isfinite(value):
            value = sys.float_info.max
        series.append({"test": i, "e_value": value, "threshold": threshold})
    return series


def _load_trajectory(ctx, path):
    if not path:
        return None
    root = Path(ctx.trajectory_dir).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def run_detail(ctx, user, run_id):
    with ctx.conn() as conn:
        repo = ctx.runs(conn)
        run = repo.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        applicability = {skill.id: skill.applicability for skill in ctx.skills(conn).search()}
        skills = {role: [] for role in SKILL_ROLES}
        for row in repo.run_skills(run_id):
            skills[row["role"]].append(
                {
                    "skill_id": row["skill_id"],
                    "applicability": applicability.get(row["skill_id"], ""),
                    "round": row["round"],
                    "similarity": row["similarity"],
                }
            )
        reviews = repo.design_reviews(run_id)
    return {
        "run": run,
        "skills": skills,
        "design_reviews": reviews,
        "trajectory": _load_trajectory(ctx, run.get("trajectory_path")),
    }
