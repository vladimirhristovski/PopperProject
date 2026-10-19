import hashlib
import math
import statistics
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _rate(k, n):
    return {"n": n, "k": k, "rate": k / n if n else None, "ci": wilson(k, n)}


def completed(runs):
    return [run for run in runs if run.get("status") == "completed"]


def errored(runs):
    return [run for run in runs if run.get("status") == "error"]


def type1(runs):
    permuted = [run for run in completed(runs) if run.get("permuted")]
    return _rate(sum(1 for run in permuted if run.get("res")), len(permuted))


def power(runs):
    positives = [run for run in completed(runs) if run.get("answer") is True]
    return _rate(sum(1 for run in positives if run.get("res")), len(positives))


def _labelled(runs):
    return [run for run in completed(runs) if run.get("answer") is not None]


def accuracy(runs):
    labelled = _labelled(runs)
    return _rate(sum(1 for run in labelled if bool(run.get("res")) == run["answer"]), len(labelled))


def error_rate(runs):
    n_errors = len(errored(runs))
    return _rate(n_errors, n_errors + len(completed(runs)))


def _summary(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "median": None}
    return {"mean": statistics.fmean(values), "median": statistics.median(values)}


def cost(runs):
    done = completed(runs)
    keys = ("tokens_in", "tokens_out", "n_tests", "wall_time_s")
    result = {key: _summary(run.get(key) for run in done) for key in keys}
    labelled = _labelled(done)
    correct = sum(1 for run in labelled if bool(run.get("res")) == run["answer"])
    tokens = [(run.get("tokens_in"), run.get("tokens_out")) for run in labelled]
    spent = sum((tin or 0) + (tout or 0) for tin, tout in tokens)
    has_tokens = any(tin is not None or tout is not None for tin, tout in tokens)
    result["cost_per_correct"] = spent / correct if correct and has_tokens else None
    return result


def reviewer_stats(runs, reviews):
    run_ids = {run["id"] for run in runs if "id" in run}
    own = [review for review in reviews if review["run_id"] in run_ids]
    attempts = defaultdict(int)
    overridden = set()
    for review in own:
        key = (review["run_id"], review["round"])
        attempts[key] = max(attempts[key], review["attempt"])
        if review.get("overridden"):
            overridden.add(key)
    if not own:
        return {"n_reviews": 0, "rejection_rate": None, "override_rate": None, "mean_attempts": None}
    return {
        "n_reviews": len(own),
        "rejection_rate": sum(1 for review in own if not review["approved"]) / len(own),
        "override_rate": len(overridden) / len(attempts),
        "mean_attempts": statistics.fmean(attempts.values()),
    }


def context_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def curator_agreement(decisions, references):
    reference = {(row["skill_id"], row["context_hash"]): row["verdict"] for row in references}
    pairs = [
        (row["verdict"], reference[(row["skill_id"], row["context_hash"])])
        for row in decisions
        if (row["skill_id"], row["context_hash"]) in reference
    ]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    agreement = sum(1 for a, b in pairs if a == b) / n
    labels = {label for pair in pairs for label in pair}
    expected = sum(
        (sum(1 for a, _ in pairs if a == label) / n) * (sum(1 for _, b in pairs if b == label) / n)
        for label in labels
    )
    kappa = 1.0 if expected == 1 else (agreement - expected) / (1 - expected)
    return {"n": n, "agreement": agreement, "kappa": kappa}


def skill_usage(run_skills, run_ids=None):
    injected = [row for row in run_skills if row["role"] == "injected"]
    counts = defaultdict(int)
    for row in injected:
        counts[row["run_id"]] += 1
    ids = set(run_ids) if run_ids is not None else set(counts)
    if not ids:
        return {"n_runs": 0, "injections_per_run": None, "share_with_injection": None}
    return {
        "n_runs": len(ids),
        "injections_per_run": sum(counts[run_id] for run_id in ids) / len(ids),
        "share_with_injection": sum(1 for run_id in ids if counts[run_id] > 0) / len(ids),
    }
