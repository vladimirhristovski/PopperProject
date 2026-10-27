import json
import statistics
from pathlib import Path

from popper_upgrade.config import (
    DEFAULT_BUDGET_EUR,
    DEFAULT_RUN_ESTIMATE_EUR,
    EUR_USD_RATE,
    EUR_USD_RATE_DATE,
)
from popper_upgrade.serving.pricing import check_pricing, eur_to_usd, usd_to_eur

BUDGET_KEY = "budget_eur"
ESTIMATE_KEY = "est_run_cost_eur"


def _positive(value):
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and value > 0 else None


def spec_value(spec_path, key, model_name):
    try:
        raw = json.loads(Path(spec_path).read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(raw, dict):
        raw = raw.get(model_name)
    return _positive(raw)


class CostGuard:
    def __init__(self, history, rate=EUR_USD_RATE, rate_date=EUR_USD_RATE_DATE, out=print):
        self._history = history
        self.rate = rate
        self.rate_date = rate_date
        self._out = out
        self.exhausted = False
        self.counter = None
        self.model_name = None
        self.budget_eur = None
        self.estimate_eur = None
        self.baseline_usd = 0.0
        self.planned = 0

    def begin(self, plan, spec_path, model_row):
        name = model_row["name"]
        if not model_row.get("pricing"):
            raise ValueError(f"model {name} has no pricing, so its spend cannot be guarded")
        check_pricing(model_row["pricing"])
        self.model_name = name
        self.budget_eur = (
            spec_value(spec_path, BUDGET_KEY, name) or _positive(model_row.get(BUDGET_KEY)) or DEFAULT_BUDGET_EUR
        )
        self.estimate_eur = spec_value(spec_path, ESTIMATE_KEY, name) or DEFAULT_RUN_ESTIMATE_EUR
        self.baseline_usd = sum(cost for _, cost in self._history(name))
        self.planned = sum(sum(arms.values()) for arms in plan.values())
        self._out(
            f"[cost] {name}: budget EUR {self.budget_eur:.2f} at {self.rate} USD per EUR (ECB {self.rate_date}); "
            f"already spent EUR {usd_to_eur(self.baseline_usd, self.rate):.2f}"
        )

    def attach(self, counter, model_row=None):
        self.counter = counter

    def spend_usd(self):
        live = self.counter.total_cost_usd() if self.counter is not None else 0.0
        return self.baseline_usd + live

    def spend_eur(self):
        return usd_to_eur(self.spend_usd(), self.rate)

    def completed_costs(self):
        return [cost for status, cost in self._history(self.model_name) if status == "completed"]

    def estimate_usd(self):
        costs = self.completed_costs()
        if costs:
            return statistics.median(costs)
        return eur_to_usd(self.estimate_eur, self.rate)

    def may_start(self, phase):
        if self.exhausted:
            return False
        spend = self.spend_usd()
        estimate = self.estimate_usd()
        budget = eur_to_usd(self.budget_eur, self.rate)
        self._out(self._line(spend, estimate, phase))
        if spend + estimate > budget:
            self.exhausted = True
            self._out(
                f"[cost] {self.model_name}: stopping before the next run, spend plus the estimated run cost "
                f"would exceed EUR {self.budget_eur:.2f}"
            )
            return False
        return True

    def _line(self, spend_usd, estimate_usd, phase):
        done = len(self.completed_costs())
        return (
            f"[cost] {self.model_name} ({phase}): spent EUR {usd_to_eur(spend_usd, self.rate):.2f} of "
            f"EUR {self.budget_eur:.2f}; {done} completed runs with cost; next run estimated "
            f"EUR {usd_to_eur(estimate_usd, self.rate):.3f}"
        )

    def progress(self):
        return {
            "model": self.model_name,
            "spent_eur": self.spend_eur(),
            "budget_eur": self.budget_eur,
            "completed_runs": len(self.completed_costs()),
            "planned_runs": self.planned,
            "exhausted": self.exhausted,
        }

    def summary(self):
        return (
            f"[cost] {self.model_name}: total spent EUR {self.spend_eur():.2f} of EUR {self.budget_eur:.2f}"
            + (" (budget reached, run stopped before the limit)" if self.exhausted else "")
        )
