import json
import math
import os
import time
from collections import Counter
from pathlib import Path

from cluster import store

MAX_HOURS_ENV = "POPPER_MAX_HOURS"
DEFAULT_EST_MINUTES = 10.0
SAFETY_SECONDS = 600.0
HINT_KEY = "est_run_minutes"


def resolve_max_hours(value=None):
    raw = value if value is not None else os.environ.get(MAX_HOURS_ENV) or None
    if raw is None:
        return None
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"the time limit must be a number of hours, not {raw!r}") from None
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError(f"the time limit must be above zero hours, not {raw!r}")
    return hours


def _positive(value):
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and value > 0 else None


def hint_seconds(spec_path, model_row):
    model_row = model_row or {}
    try:
        raw = json.loads(Path(spec_path).read_text(encoding="utf-8")).get(HINT_KEY)
    except (OSError, ValueError, AttributeError):
        raw = None
    if isinstance(raw, dict):
        raw = raw.get(model_row.get("name"))
    minutes = _positive(raw) or _positive(model_row.get(HINT_KEY)) or DEFAULT_EST_MINUTES
    return minutes * 60


def completed_counts(bundle):
    phases = store.experiment_phases(bundle)
    return Counter(
        (store.run_phase(row, phases), row["arm"])
        for row in store.final_runs(bundle)
        if row.get("status") == "completed"
    )


def progress_report(bundle, plan):
    done = completed_counts(bundle)
    phases = {}
    for phase, arms in plan.items():
        entries = {arm: {"done": min(done[(phase, arm)], total), "total": total} for arm, total in arms.items()}
        phases[phase] = {
            "done": sum(entry["done"] for entry in entries.values()),
            "total": sum(entry["total"] for entry in entries.values()),
            "arms": entries,
        }
    return {
        "done": sum(entry["done"] for entry in phases.values()),
        "total": sum(entry["total"] for entry in phases.values()),
        "phases": phases,
    }


class Budget:
    def __init__(self, bundle_dir, max_hours, clock=time.monotonic, started=None):
        self.bundle_dir = Path(bundle_dir)
        self.max_hours = max_hours
        self.limit = None if max_hours is None else max_hours * 3600
        self.clock = clock
        self.started = clock() if started is None else started
        self.hint = DEFAULT_EST_MINUTES * 60
        self.exhausted = False

    def begin(self, plan, spec_path, model_row):
        self.hint = hint_seconds(spec_path, model_row)
        store.save_plan(self.bundle_dir, plan)

    def elapsed(self):
        return self.clock() - self.started

    def estimate(self, phase):
        median = store.phase_median_seconds(self.bundle_dir, phase)
        return median if median is not None else self.hint

    def may_start(self, phase):
        if self.limit is None:
            return True
        if self.exhausted:
            return False
        if self.elapsed() + self.estimate(phase) + SAFETY_SECONDS > self.limit:
            self.exhausted = True
            return False
        return True

    def progress(self):
        plan = store.load_plan(self.bundle_dir)
        return progress_report(self.bundle_dir, plan) if plan else None
