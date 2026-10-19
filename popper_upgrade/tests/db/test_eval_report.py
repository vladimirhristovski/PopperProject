import csv
import uuid

import pytest

from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.eval.report import build_report, write_report
from popper_upgrade.skills.embedding import DEFAULT_EMBEDDING_MODEL
from popper_upgrade.tests.fakes import FakeEmbedder, run_config

pytestmark = pytest.mark.db

PROMPTS = {"curator": "c" * 64, "reviewer": "r" * 64, "extractor": "e" * 64}


def _experiment(repo):
    experiment_id = str(uuid.uuid4())
    repo.create_experiment(
        {"id": experiment_id, "name": "report_test", "spec": {"prompt_versions": PROMPTS}, "status": "open"}
    )
    return experiment_id


def _add(repo, experiment_id, model, arm, res, answer, permuted=False, status="completed", **result):
    run_id = repo.create_run(
        run_config(
            experiment_id=experiment_id,
            model_name=model,
            arm=arm,
            permuted=permuted,
            answer=answer,
            task="task",
            replicate=0,
        )
    )
    fields = {"status": status, "tokens_in": 100, "tokens_out": 20, "n_tests": 2, "wall_time_s": 5.0, **result}
    if status == "completed":
        fields["res"] = res
    repo.finish_run(run_id, fields)
    return run_id


def _seed(db):
    repo = RunRepository(db, None)
    experiment_id = _experiment(repo)
    _add(repo, experiment_id, "small", "baseline", True, True)
    _add(repo, experiment_id, "small", "baseline", False, False, permuted=True)
    _add(repo, experiment_id, "small", "baseline", True, False, permuted=True)
    _add(repo, experiment_id, "small", "baseline", None, True, status="error")
    _add(repo, experiment_id, "small", "wrong_domain_skills", False, True, control_degenerate=True)
    _add(repo, experiment_id, "large", "baseline", True, True)
    return repo, experiment_id


def test_report_rows_per_model_arm(db):
    _, experiment_id = _seed(db)

    report = build_report(db, experiment_id)

    rows = {(row["model"], row["arm"]): row for row in report["rows"]}
    assert set(rows) == {("small", "baseline"), ("small", "wrong_domain_skills"), ("large", "baseline")}
    baseline = rows[("small", "baseline")]
    assert baseline["n_runs"] == 4
    assert baseline["n_completed"] == 3
    assert (baseline["type1"]["n"], baseline["type1"]["k"]) == (2, 1)
    assert baseline["power"]["n"] == 1
    assert baseline["error_rate"]["k"] == 1
    assert baseline["accuracy"]["n"] == 3
    assert rows[("small", "wrong_domain_skills")]["control_degenerate"] is True
    assert baseline["control_degenerate"] is False
    assert report["prompt_hashes"] == PROMPTS
    assert report["experiment"]["name"] == "report_test"


def test_write_report_files(db, tmp_path):
    _, experiment_id = _seed(db)

    write_report(build_report(db, experiment_id), tmp_path / "out")

    with (tmp_path / "out" / "summary.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(table) == 3
    assert {"model", "arm", "type1_rate", "type1_ci_low", "type1_ci_high", "cost_cost_per_correct"} <= set(table[0])
    markdown = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "report_test" in markdown
    assert "wrong_domain_skills (control_degenerate)" in markdown
    assert "n=0" in markdown
    assert PROMPTS["curator"] in markdown


def test_unknown_experiment_raises(db):
    with pytest.raises(KeyError):
        build_report(db, str(uuid.uuid4()))


def test_completed_runs_keys_only_completed(db):
    repo, experiment_id = _seed(db)

    keys = repo.completed_runs(experiment_id)

    assert ("small", "baseline", "task", "H", 0, True) in keys
    assert ("small", "baseline", "task", "H", 0, False) in keys
    assert len(keys) == 4
    assert repo.completed_runs(str(uuid.uuid4())) == set()


def test_bundle_carries_experiments_into_the_database(db, tmp_path):
    embedder = FakeEmbedder(model_name=DEFAULT_EMBEDDING_MODEL)
    export_snapshot(db, tmp_path / "snapshot.json")
    snapshot = SnapshotSkillSource(tmp_path / "snapshot.json", embedder)
    sink = BundleSink(tmp_path / "bundle", snapshot)
    experiment_id = str(uuid.uuid4())
    experiment = {"id": experiment_id, "name": "bundled", "spec": {"a": 1}, "status": "open"}

    assert sink.create_experiment(experiment) == experiment_id
    sink.create_experiment(experiment)
    run_id = sink.create_run(run_config(environment="cluster", experiment_id=experiment_id, task="t", replicate=1))
    sink.finish_run(run_id, {"res": True, "res_stat": 2.0})
    assert sink.completed_runs(experiment_id) == {("claude-sonnet-4-5", "baseline", "t", "H", 1, False)}

    counts = import_bundle(db, tmp_path / "bundle")
    again = import_bundle(db, tmp_path / "bundle")

    assert counts["experiments"] == 1
    assert again["experiments"] == 0
    row = db.execute("select name, spec from runs.experiments where id = %s", (experiment_id,)).fetchone()
    assert row == {"name": "bundled", "spec": {"a": 1}}
    assert db.execute("select task, replicate from runs.runs where id = %s", (run_id,)).fetchone() == {
        "task": "t",
        "replicate": 1,
    }
