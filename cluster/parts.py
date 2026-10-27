import json
import shutil
from pathlib import Path

from cluster import store

PARTS = "parts.json"
EXPERIMENTS = "experiments.jsonl"
CANDIDATES = "candidate_skills.jsonl"
DECISIONS = "skill_decisions.jsonl"
RUN_SKILLS = "run_skills.jsonl"
DESIGN_REVIEWS = "design_reviews.jsonl"
MANIFEST = "manifest.json"
FREEZE_INFO = "freeze.json"
FROZEN_SNAPSHOT = "frozen_snapshot.json"
REGISTRY = "registry.json"
TRAJECTORIES = "trajectories"


def read_parts(workdir):
    path = Path(workdir) / PARTS
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def write_parts(workdir, parts):
    (Path(workdir) / PARTS).write_text(json.dumps(parts, indent=2), encoding="utf-8")


def shipped_ids(parts, key):
    return {item for part in parts for item in part.get(key, [])}


def _copy_if_present(source, target):
    if source.is_file():
        shutil.copyfile(source, target)


def stage_part(bundle, stage, parts):
    bundle, stage = Path(bundle), Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    shipped_runs = shipped_ids(parts, "runs")
    runs = [run for run in store.final_runs(bundle) if run["id"] not in shipped_runs]
    kept = {run["id"] for run in runs}
    experiment_ids = {run.get("experiment_id") for run in runs}
    store.write_lines(stage / store.RUNS, runs)
    store.write_lines(
        stage / EXPERIMENTS, [row for row in store.read_lines(bundle / EXPERIMENTS) if row["id"] in experiment_ids]
    )
    run_skills = [row for row in store.read_lines(bundle / RUN_SKILLS) if row.get("run_id") in kept]
    reviews = [row for row in store.read_lines(bundle / DESIGN_REVIEWS) if row.get("run_id") in kept]
    shipped_candidates = shipped_ids(parts, "candidates")
    all_candidates = store.read_lines(bundle / CANDIDATES)
    new_ids = {row["id"] for row in all_candidates if row["id"] not in shipped_candidates}
    decisions = [
        row
        for row in store.read_lines(bundle / DECISIONS)
        if row.get("run_id") in kept or (row.get("run_id") is None and row.get("skill_id") in new_ids)
    ]
    referenced = {row["skill_id"] for row in run_skills} | {row["skill_id"] for row in decisions}
    candidates = [row for row in all_candidates if row["id"] in new_ids | referenced]
    store.write_lines(stage / RUN_SKILLS, run_skills)
    store.write_lines(stage / DESIGN_REVIEWS, reviews)
    store.write_lines(stage / DECISIONS, decisions)
    store.write_lines(stage / CANDIDATES, candidates)
    for name in (MANIFEST, FREEZE_INFO, FROZEN_SNAPSHOT, REGISTRY):
        _copy_if_present(bundle / name, stage / name)
    (stage / TRAJECTORIES).mkdir(exist_ok=True)
    for run_id in kept:
        _copy_if_present(bundle / TRAJECTORIES / f"{run_id}.json", stage / TRAJECTORIES / f"{run_id}.json")
    return runs, sorted(new_ids)


def stage_cumulative(bundle, view):
    bundle, view = Path(bundle), Path(view)
    view.mkdir(parents=True, exist_ok=True)
    runs = store.final_runs(bundle)
    kept = {run["id"] for run in runs}
    store.write_lines(view / store.RUNS, runs)
    for name in (RUN_SKILLS, DESIGN_REVIEWS, DECISIONS):
        rows = [row for row in store.read_lines(bundle / name) if row.get("run_id") in kept | {None}]
        store.write_lines(view / name, rows)
    for name in (EXPERIMENTS, MANIFEST, FREEZE_INFO):
        _copy_if_present(bundle / name, view / name)
