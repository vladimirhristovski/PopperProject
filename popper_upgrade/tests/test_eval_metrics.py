import pytest

from popper_upgrade.eval import metrics


def _run(res=True, status="completed", permuted=False, answer=None, **extra):
    return {"id": extra.pop("id", "r"), "status": status, "res": res, "permuted": permuted, "answer": answer, **extra}


def test_wilson_known_value():
    low, high = metrics.wilson(1, 20)

    assert low == pytest.approx(0.0089, abs=1e-3)
    assert high == pytest.approx(0.2361, abs=1e-3)


def test_wilson_with_no_trials_is_none():
    assert metrics.wilson(0, 0) is None


def test_type1_zero_runs():
    assert metrics.type1([]) == {"n": 0, "k": 0, "rate": None, "ci": None}
    unpermuted = [_run(permuted=False, answer=True)]
    assert metrics.type1(unpermuted)["n"] == 0


def test_type1_counts_only_permuted_runs():
    runs = [_run(res=True, permuted=True, answer=False), _run(res=False, permuted=True, answer=False)]
    runs.append(_run(res=True, permuted=False, answer=True))

    result = metrics.type1(runs)

    assert (result["n"], result["k"], result["rate"]) == (2, 1, 0.5)
    assert result["ci"] == metrics.wilson(1, 2)


def test_errors_excluded_from_denominators():
    runs = [
        _run(res=True, permuted=True, answer=False),
        _run(res=None, status="error", permuted=True, answer=False),
        _run(res=True, answer=True),
        _run(res=None, status="error", answer=True),
    ]

    assert metrics.type1(runs)["n"] == 1
    assert metrics.power(runs)["n"] == 1
    assert metrics.accuracy(runs)["n"] == 2
    errors = metrics.error_rate(runs)
    assert (errors["n"], errors["k"], errors["rate"]) == (4, 2, 0.5)


def test_error_rate_without_runs_is_none():
    assert metrics.error_rate([])["rate"] is None


def test_power_and_accuracy():
    runs = [
        _run(res=True, answer=True),
        _run(res=False, answer=True),
        _run(res=False, permuted=True, answer=False),
        _run(res=True, permuted=True, answer=False),
        _run(res=True),
    ]

    assert metrics.power(runs)["rate"] == 0.5
    accuracy = metrics.accuracy(runs)
    assert (accuracy["n"], accuracy["k"]) == (4, 2)


def test_cost_per_correct():
    runs = [
        _run(res=True, answer=True, tokens_in=100, tokens_out=50, n_tests=2, wall_time_s=10.0),
        _run(res=False, answer=True, tokens_in=300, tokens_out=50, n_tests=4, wall_time_s=20.0),
        _run(res=None, status="error", answer=True, tokens_in=999, tokens_out=999),
    ]

    cost = metrics.cost(runs)

    assert cost["cost_per_correct"] == 500
    assert cost["tokens_in"] == {"mean": 200, "median": 200}
    assert cost["n_tests"]["mean"] == 3
    assert cost["wall_time_s"]["median"] == 15


def test_cost_without_tokens_or_correct_answers():
    assert metrics.cost([_run(res=True, answer=True)])["cost_per_correct"] is None
    wrong = [_run(res=False, answer=True, tokens_in=1, tokens_out=1)]
    assert metrics.cost(wrong)["cost_per_correct"] is None
    assert metrics.cost([])["tokens_in"] == {"mean": None, "median": None}


def _verdicts(verdicts):
    return [{"skill_id": f"s{i}", "context_hash": "c", "verdict": v} for i, v in enumerate(verdicts)]


def test_kappa_perfect_and_chance():
    references = _verdicts(["approve", "approve", "reject", "reject"])

    perfect = metrics.curator_agreement(_verdicts(["approve", "approve", "reject", "reject"]), references)
    chance = metrics.curator_agreement(_verdicts(["approve", "reject", "approve", "reject"]), references)
    opposite = metrics.curator_agreement(_verdicts(["reject", "reject", "approve", "approve"]), references)
    none = metrics.curator_agreement([], references)

    assert perfect == {"n": 4, "agreement": 1.0, "kappa": 1.0}
    assert chance["agreement"] == 0.5
    assert chance["kappa"] == pytest.approx(0.0)
    assert opposite["kappa"] == pytest.approx(-1.0)
    assert none == {"n": 0, "agreement": None, "kappa": None}


def test_curator_agreement_matches_on_skill_and_context():
    references = [{"skill_id": "s1", "context_hash": "c1", "verdict": "approve"}]
    decisions = [
        {"skill_id": "s1", "context_hash": "c2", "verdict": "reject"},
        {"skill_id": "s1", "context_hash": "c1", "verdict": "approve"},
    ]

    assert metrics.curator_agreement(decisions, references)["n"] == 1


def test_reviewer_stats():
    runs = [{"id": "a"}, {"id": "b"}]
    reviews = [
        {"run_id": "a", "round": 1, "attempt": 1, "approved": False, "overridden": False},
        {"run_id": "a", "round": 1, "attempt": 2, "approved": True, "overridden": False},
        {"run_id": "b", "round": 1, "attempt": 1, "approved": False, "overridden": False},
        {"run_id": "b", "round": 1, "attempt": 2, "approved": False, "overridden": True},
        {"run_id": "other", "round": 1, "attempt": 1, "approved": False, "overridden": False},
    ]

    stats = metrics.reviewer_stats(runs, reviews)

    assert stats["n_reviews"] == 4
    assert stats["rejection_rate"] == 0.75
    assert stats["override_rate"] == 0.5
    assert stats["mean_attempts"] == 2
    assert metrics.reviewer_stats(runs, [])["rejection_rate"] is None


def test_skill_usage():
    rows = [
        {"run_id": "a", "role": "injected"},
        {"run_id": "a", "role": "injected"},
        {"run_id": "b", "role": "curator_approved"},
        {"run_id": "c", "role": "injected"},
    ]

    usage = metrics.skill_usage(rows, ["a", "b", "c", "d"])

    assert usage == {"n_runs": 4, "injections_per_run": 0.75, "share_with_injection": 0.5}
    assert metrics.skill_usage([], [])["injections_per_run"] is None
    assert metrics.skill_usage(rows)["n_runs"] == 2
