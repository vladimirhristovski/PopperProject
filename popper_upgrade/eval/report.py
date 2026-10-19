import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from popper_upgrade.eval import metrics

RATE_METRICS = ("type1", "power", "accuracy", "error_rate")


def _fetch_runs(conn, experiment_id):
    rows = conn.execute("select * from runs.runs where experiment_id = %s order by created_at, id", (experiment_id,))
    return [{**row, "id": str(row["id"])} for row in rows.fetchall()]


def _fetch_reviews(conn, run_ids):
    rows = conn.execute(
        "select run_id::text as run_id, round, attempt, approved, overridden from runs.design_reviews "
        "where run_id::text = any(%s::text[])",
        (run_ids,),
    )
    return rows.fetchall()


def _fetch_run_skills(conn, run_ids):
    rows = conn.execute(
        "select run_id::text as run_id, skill_id::text as skill_id, role from runs.run_skills "
        "where run_id::text = any(%s::text[])",
        (run_ids,),
    )
    return rows.fetchall()


def _fetch_curator_decisions(conn, run_ids):
    rows = conn.execute(
        "select d.run_id::text as run_id, d.skill_id::text as skill_id, d.verdict, r.hypothesis "
        "from skills.skill_decisions d join runs.runs r on r.id = d.run_id "
        "where d.trigger = 'pre_run' and d.decider = 'curator' and d.run_id::text = any(%s::text[])",
        (run_ids,),
    )
    return [
        {**row, "context_hash": metrics.context_hash(row["hypothesis"])} for row in rows.fetchall()
    ]


def _fetch_references(conn):
    rows = conn.execute(
        "select skill_id::text as skill_id, context_hash, verdict from registry.curation_references"
    )
    return rows.fetchall()


def _assemble(experiment, runs, reviews, run_skills, decisions, references):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["model_name"], run["arm"])].append(run)
    rows = []
    for (model, arm), group in sorted(groups.items()):
        ids = {run["id"] for run in group}
        rows.append(
            {
                "model": model,
                "arm": arm,
                "n_runs": len(group),
                "n_completed": len(metrics.completed(group)),
                "type1": metrics.type1(group),
                "power": metrics.power(group),
                "accuracy": metrics.accuracy(group),
                "error_rate": metrics.error_rate(group),
                "cost": metrics.cost(group),
                "reviewer": metrics.reviewer_stats(group, reviews),
                "skill_usage": metrics.skill_usage([row for row in run_skills if row["run_id"] in ids], ids),
                "curator": metrics.curator_agreement(
                    [row for row in decisions if row["run_id"] in ids], references
                ),
                "control_degenerate": any(run.get("control_degenerate") for run in group),
            }
        )
    return {
        "experiment": experiment,
        "snapshot_ids": sorted({str(run["library_snapshot_id"]) for run in runs if run.get("library_snapshot_id")}),
        "git_shas": sorted({run["git_sha"] for run in runs if run.get("git_sha")}),
        "prompt_hashes": experiment["spec"].get("prompt_versions", {}),
        "rows": rows,
    }


def build_report(conn, experiment_id):
    experiment = conn.execute(
        "select id::text as id, name, spec, status from runs.experiments where id = %s", (experiment_id,)
    ).fetchone()
    if experiment is None:
        raise KeyError(experiment_id)
    runs = _fetch_runs(conn, experiment_id)
    run_ids = [run["id"] for run in runs]
    return _assemble(
        experiment,
        runs,
        _fetch_reviews(conn, run_ids),
        _fetch_run_skills(conn, run_ids),
        _fetch_curator_decisions(conn, run_ids),
        _fetch_references(conn),
    )


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def report_from_bundle(directory, experiment_id=None, references=()):
    directory = Path(directory)
    experiments = _read_jsonl(directory / "experiments.jsonl")
    if experiment_id is not None:
        experiments = [row for row in experiments if row["id"] == experiment_id]
    if not experiments:
        raise KeyError(experiment_id)
    row = experiments[0]
    experiment = {key: row[key] for key in ("id", "name", "spec", "status")}
    merged = {}
    for entry in _read_jsonl(directory / "runs.jsonl"):
        merged[entry["id"]] = {**merged.get(entry["id"], {}), **entry}
    runs = sorted(
        (run for run in merged.values() if run.get("experiment_id") == experiment["id"]),
        key=lambda run: (run["created_at"], run["id"]),
    )
    by_id = {run["id"]: run for run in runs}
    reviews = [row for row in _read_jsonl(directory / "design_reviews.jsonl") if row["run_id"] in by_id]
    run_skills = [row for row in _read_jsonl(directory / "run_skills.jsonl") if row["run_id"] in by_id]
    decisions = [
        {**row, "context_hash": metrics.context_hash(by_id[row["run_id"]]["hypothesis"])}
        for row in _read_jsonl(directory / "skill_decisions.jsonl")
        if row.get("run_id") in by_id and row.get("trigger") == "pre_run" and row.get("decider") == "curator"
    ]
    return _assemble(experiment, runs, reviews, run_skills, decisions, list(references))


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, inner, out)
    elif isinstance(value, tuple | list) and prefix.endswith("_ci"):
        out[f"{prefix}_low"], out[f"{prefix}_high"] = value
    elif prefix.endswith("_ci"):
        out[f"{prefix}_low"] = out[f"{prefix}_high"] = None
    else:
        out[prefix] = value


def flatten_row(row):
    out = {}
    _flatten("", row, out)
    return out


def _fmt_rate(metric):
    if metric["n"] == 0:
        return "n=0"
    low, high = metric["ci"]
    return f"{metric['k']}/{metric['n']} = {metric['rate']:.3f} [{low:.3f}, {high:.3f}]"


def _fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _label(row):
    return f"{row['arm']} (control_degenerate)" if row["control_degenerate"] else row["arm"]


def render_markdown(report):
    experiment = report["experiment"]
    rows = report["rows"]
    outcomes = _table(
        ["model", "arm", "runs", "completed", "error rate", "Type-I", "power", "accuracy"],
        [
            [
                row["model"],
                _label(row),
                str(row["n_runs"]),
                str(row["n_completed"]),
                _fmt_rate(row["error_rate"]),
                _fmt_rate(row["type1"]),
                _fmt_rate(row["power"]),
                _fmt_rate(row["accuracy"]),
            ]
            for row in rows
        ],
    )
    costs = _table(
        ["model", "arm", "tokens in", "tokens out", "tests", "wall s", "tokens per correct"],
        [
            [
                row["model"],
                _label(row),
                _fmt(row["cost"]["tokens_in"]["mean"], 0),
                _fmt(row["cost"]["tokens_out"]["mean"], 0),
                _fmt(row["cost"]["n_tests"]["mean"]),
                _fmt(row["cost"]["wall_time_s"]["mean"]),
                _fmt(row["cost"]["cost_per_correct"], 0),
            ]
            for row in rows
        ],
    )
    skills = _table(
        ["model", "arm", "injections/run", "runs with injection", "reviews", "rejection rate", "override rate",
         "mean attempts", "curator n", "curator agreement", "curator kappa"],
        [
            [
                row["model"],
                _label(row),
                _fmt(row["skill_usage"]["injections_per_run"], 2),
                _fmt(row["skill_usage"]["share_with_injection"], 2),
                str(row["reviewer"]["n_reviews"]),
                _fmt(row["reviewer"]["rejection_rate"], 2),
                _fmt(row["reviewer"]["override_rate"], 2),
                _fmt(row["reviewer"]["mean_attempts"], 2),
                str(row["curator"]["n"]),
                _fmt(row["curator"]["agreement"], 2),
                _fmt(row["curator"]["kappa"], 2),
            ]
            for row in rows
        ],
    )
    spec = json.dumps(experiment["spec"], indent=2, sort_keys=True)
    prompts = "\n".join(f"- {role}: `{digest}`" for role, digest in sorted(report["prompt_hashes"].items()))
    return (
        f"# {experiment['name']}\n\n"
        f"- experiment id: `{experiment['id']}`\n"
        f"- library snapshot id: {', '.join(f'`{i}`' for i in report['snapshot_ids']) or '-'}\n"
        f"- git SHA: {', '.join(f'`{i}`' for i in report['git_shas']) or '-'}\n\n"
        f"## Outcomes\n\n{outcomes}\n\n## Cost\n\n{costs}\n\n## Skills, reviewer and curator\n\n{skills}\n\n"
        f"## Prompt hashes\n\n{prompts}\n\n## Experiment spec\n\n```json\n{spec}\n```\n"
    )


def write_report(report, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = [flatten_row(row) for row in report["rows"]]
    columns = list(flat[0]) if flat else ["model", "arm"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    (out_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.eval.report")
    parser.add_argument("experiment_id")
    parser.add_argument("out_dir")
    args = parser.parse_args(argv)
    from popper_upgrade.db.connection import connect

    with connect() as conn:
        write_report(build_report(conn, args.experiment_id), args.out_dir)
    print(Path(args.out_dir) / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
