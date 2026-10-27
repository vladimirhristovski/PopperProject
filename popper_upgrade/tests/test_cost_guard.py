import contextlib
from types import SimpleNamespace

import pytest

import popper_upgrade.eval.run_experiment as module
from popper_upgrade.config import EUR_USD_RATE
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.cost_guard import CostGuard, spec_value
from popper_upgrade.eval.run_experiment import run_experiment
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.fakes import OLLAMA_MODEL, FakeEmbedder, FakeSink, write_snapshot
from popper_upgrade.tests.test_run_experiment import DATASETS, HYPOTHESES, _write_spec

PRICING = {"input": 2, "cache_read": 0.1, "cache_write": 2.5, "output": 10}
PAID = {**OLLAMA_MODEL, "pricing": PRICING, "budget_eur": 5}
USD_PER_TOKEN = 10 / 1e6


def _tokens(usd):
    return round(usd / USD_PER_TOKEN)


@pytest.fixture
def paid(monkeypatch):
    state = SimpleNamespace(history=[], costs=[], ran=0, lines=[], served=[])

    class Session:
        def __init__(self, spec, source, sink, curator, reviewer, extractor, trajectory_dir, **kwargs):
            self.usage = kwargs["usage"]

        def run(self):
            self.usage.reset()
            self.usage.add({"prompt_tokens": 0, "completion_tokens": _tokens(state.costs.pop(0))})
            state.history.append(("completed", self.usage.cost_usd()))
            state.ran += 1

    @contextlib.contextmanager
    def serving(model, log_dir=None):
        counter = UsageCounter(model.get("pricing"))
        counter.port = 4321
        state.served.append(counter)
        yield counter

    monkeypatch.setattr(module, "RunSession", Session)
    monkeypatch.setattr(module, "serving_for", serving)
    monkeypatch.setattr(
        module, "build_agents", lambda kwargs, skills, reviewer, extractor: (object() if skills else None, None, None)
    )
    state.guard = lambda: CostGuard(lambda name: list(state.history), out=state.lines.append)
    return state


def _snapshot(tmp_path, model=PAID):
    embedder = FakeEmbedder()
    path = write_snapshot(
        tmp_path / "snap.json",
        [],
        embedder,
        models=[model],
        hypothesis_sets={"gender_v1": HYPOTHESES},
        datasets=DATASETS,
    )
    return SnapshotSkillSource(path, embedder)


def _run(tmp_path, guard, spec=None, model=PAID):
    return run_experiment(
        spec or _write_spec(tmp_path),
        "llama3.1:8b",
        _snapshot(tmp_path, model),
        FakeSink(),
        "laptop",
        data_dir=tmp_path,
        trajectory_dir=tmp_path / "traj",
        budget=guard,
        on_serving=guard.attach,
    )


def test_runs_continue_until_spend_plus_median_would_exceed_the_budget(tmp_path, paid):
    paid.costs = [2.0] * 24
    guard = paid.guard()

    _run(tmp_path, guard)

    assert paid.ran == 2
    assert guard.exhausted is True
    assert guard.spend_usd() == pytest.approx(4.0)
    assert guard.spend_eur() == pytest.approx(4.0 / EUR_USD_RATE)
    assert "stopping before the next run" in paid.lines[-1]


def test_cumulative_spend_is_shown_in_the_progress_output(tmp_path, paid):
    paid.costs = [2.0] * 24
    guard = paid.guard()

    _run(tmp_path, guard)

    assert paid.lines[0].startswith("[cost] llama3.1:8b: budget EUR 5.00 at 1.1193 USD per EUR (ECB 2026-10-23)")
    assert paid.lines[1] == (
        "[cost] llama3.1:8b (evaluate): spent EUR 0.00 of EUR 5.00; 0 completed runs with cost; "
        "next run estimated EUR 0.500"
    )
    assert paid.lines[2] == (
        "[cost] llama3.1:8b (evaluate): spent EUR 1.79 of EUR 5.00; 1 completed runs with cost; "
        "next run estimated EUR 1.787"
    )
    assert guard.progress()["spent_eur"] == pytest.approx(4.0 / EUR_USD_RATE)
    assert guard.summary().endswith("(budget reached, run stopped before the limit)")


def test_a_started_run_is_never_interrupted_even_when_it_overshoots(tmp_path, paid):
    paid.costs = [1.0, 6.0, 1.0, 1.0]
    guard = paid.guard()

    _run(tmp_path, guard)

    assert paid.ran == 2
    assert [round(cost, 6) for _, cost in paid.history] == [1.0, 6.0]
    assert guard.spend_usd() == pytest.approx(7.0)
    assert guard.spend_usd() > 5 * EUR_USD_RATE
    assert guard.exhausted is True


def test_a_run_is_refused_up_front_when_the_budget_is_already_spent(tmp_path, paid):
    paid.history = [("completed", 30.0)]
    paid.costs = [1.0]
    guard = paid.guard()

    _run(tmp_path, guard)

    assert paid.ran == 0
    assert guard.exhausted is True


def test_the_configured_estimate_gates_the_first_run(tmp_path, paid):
    paid.costs = [0.1] * 24
    guard = paid.guard()

    _run(tmp_path, guard, model={**PAID, "budget_eur": 0.4})

    assert paid.ran == 0

    paid.costs = [0.1] * 24
    guard = paid.guard()
    spec = _write_spec(tmp_path, est_run_cost_eur=0.05, budget_eur=0.4)

    _run(tmp_path, guard, spec=spec, model={**PAID, "budget_eur": 0.4})

    assert paid.ran > 0


def test_spend_before_this_process_and_error_runs_count_but_only_completed_runs_set_the_median(tmp_path, paid):
    paid.history = [("completed", 2.0), ("error", 1.0)]
    paid.costs = [2.0] * 24
    guard = paid.guard()

    _run(tmp_path, guard)

    assert paid.ran == 1
    assert guard.spend_usd() == pytest.approx(5.0)


def test_spec_values_override_the_registry_and_accept_per_model_dicts(tmp_path, paid):
    paid.costs = [1.0] * 24
    spec = _write_spec(tmp_path, budget_eur={"llama3.1:8b": 1.0}, est_run_cost_eur=0.1)
    guard = paid.guard()

    _run(tmp_path, guard, spec=spec)

    assert guard.budget_eur == 1.0
    assert guard.estimate_eur == 0.1
    assert paid.ran == 1
    assert spec_value(spec, "budget_eur", "other") is None
    assert spec_value(tmp_path / "missing.json", "budget_eur", "x") is None


def test_a_model_without_pricing_cannot_be_guarded(tmp_path, paid):
    guard = paid.guard()

    with pytest.raises(ValueError, match="no pricing"):
        _run(tmp_path, guard, model=OLLAMA_MODEL)

    assert paid.ran == 0
    assert paid.served == []


def test_default_budget_is_twenty_euro_and_converts_at_the_configured_rate(tmp_path, paid):
    paid.costs = [1.0] * 24
    guard = paid.guard()

    _run(tmp_path, guard, model={k: v for k, v in PAID.items() if k != "budget_eur"})

    assert guard.budget_eur == 20.0
    assert paid.ran == 22
    assert guard.spend_usd() == pytest.approx(22.0)
    assert guard.exhausted is True


def _begun(tmp_path, history, **model):
    lines = []
    guard = CostGuard(lambda name: list(history), out=lines.append)
    row = {"name": "m", "pricing": PRICING, "budget_eur": 5, **model}
    guard.begin({"evaluate": {"baseline": 1, "skills": 1}}, tmp_path / "none.json", row)
    return guard


def test_the_pre_run_estimate_is_the_median_of_the_arm_then_overall_then_configured(tmp_path):
    history = [("completed", 0.1, "baseline")] * 3 + [("completed", 3.0, "skills")] * 3
    guard = _begun(tmp_path, history)

    assert guard.estimate_usd("baseline") == pytest.approx(0.1)
    assert guard.estimate_usd("skills") == pytest.approx(3.0)
    assert guard.estimate_usd("random_skills") == pytest.approx(1.55)
    assert guard.estimate_usd() == pytest.approx(1.55)
    assert _begun(tmp_path, []).estimate_usd("skills") == pytest.approx(0.5 * EUR_USD_RATE)
    legacy = _begun(tmp_path, [("completed", 2.0), ("completed", 4.0)])
    assert legacy.estimate_usd("skills") == pytest.approx(3.0)


def test_an_expensive_arm_is_stopped_while_a_cheap_arm_may_still_start(tmp_path):
    history = [("completed", 0.1, "baseline")] * 3 + [("completed", 3.0, "skills")] * 3
    guard = _begun(tmp_path, history, budget_eur=5)
    guard.baseline_usd = 3.0

    assert guard.may_start("evaluate", "baseline") is True
    assert guard.may_start("evaluate", "skills") is False
    assert guard.exhausted is True


def test_runs_that_crashed_without_a_cost_are_charged_at_the_costliest_completed_run(tmp_path):
    history = [("running", None, "skills"), ("error", None, "baseline"), ("completed", 2.0, "baseline")]
    history += [("completed", 4.0, "skills"), ("error", 1.0, "skills")]

    guard = _begun(tmp_path, history)

    assert guard.baseline_usd == pytest.approx(2.0 + 4.0 + 1.0 + 2 * 4.0)
    assert guard.estimate_usd("skills") == pytest.approx(4.0)


def test_crashed_runs_without_any_completed_run_are_charged_at_the_configured_estimate(tmp_path):
    guard = _begun(tmp_path, [("running", None, "skills")] * 2)

    assert guard.baseline_usd == pytest.approx(2 * 0.5 * EUR_USD_RATE)


def test_attach_caps_the_live_spend_at_what_is_left_of_the_budget(tmp_path):
    guard = _begun(tmp_path, [("completed", 1.0, "baseline")], budget_eur=5)
    counter = UsageCounter(PRICING)
    guard.attach(counter)

    assert counter.over_limit() is False
    counter.add({"prompt_tokens": 0, "completion_tokens": _tokens(5 * EUR_USD_RATE - 1.0) - 1})
    assert counter.over_limit() is False
    counter.add({"prompt_tokens": 0, "completion_tokens": 2})
    assert counter.over_limit() is True
    assert "spend limit" in counter.limit_message()
