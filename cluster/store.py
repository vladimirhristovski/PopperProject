import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from popper_upgrade.db.rows import run_key

RUNS = "runs.jsonl"
EXPERIMENTS = "experiments.jsonl"
PLAN = "plan.json"
LEARN = "learn"
EVALUATE = "evaluate"


def now():
    return datetime.now(timezone.utc)


def read_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def merged_runs(directory):
    merged = {}
    for row in read_lines(Path(directory) / RUNS):
        merged[row["id"]] = {**merged.get(row["id"], {}), **row}
    return merged


def close_stale_runs(bundle):
    stale = [run_id for run_id, row in merged_runs(bundle).items() if row.get("status") == "running"]
    if stale:
        with (Path(bundle) / RUNS).open("a", encoding="utf-8") as handle:
            for run_id in stale:
                row = {
                    "id": run_id,
                    "status": "error",
                    "error": "the job stopped before this run finished",
                    "finished_at": now().isoformat(),
                }
                handle.write(json.dumps(row) + "\n")
    return len(stale)


def final_runs(bundle):
    best = {}
    for row in merged_runs(bundle).values():
        key = run_key(row)
        held = best.get(key)
        if held is None or row.get("status") == "completed" or held.get("status") != "completed":
            best[key] = row
    return list(best.values())


def experiment_phases(bundle):
    return {
        row["id"]: (row.get("spec") or {}).get("phase") or EVALUATE for row in read_lines(Path(bundle) / EXPERIMENTS)
    }


def run_phase(row, phases):
    return phases.get(row.get("experiment_id"), EVALUATE)


def _parse(stamp):
    try:
        return datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None


def run_seconds(row):
    wall = row.get("wall_time_s")
    if isinstance(wall, int | float) and wall >= 0:
        return float(wall)
    started, finished = _parse(row.get("created_at")), _parse(row.get("finished_at"))
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds())


def phase_median_seconds(bundle, phase):
    phases = experiment_phases(bundle)
    values = [
        run_seconds(row)
        for row in merged_runs(bundle).values()
        if row.get("status") == "completed" and run_phase(row, phases) == phase
    ]
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def load_plan(bundle):
    path = Path(bundle) / PLAN
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_plan(bundle, plan):
    Path(bundle).mkdir(parents=True, exist_ok=True)
    (Path(bundle) / PLAN).write_text(json.dumps(plan, indent=2), encoding="utf-8")
