import json
import uuid

import pytest

from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.eval.report import build_report, report_from_bundle
from popper_upgrade.skills.embedding import DEFAULT_EMBEDDING_MODEL
from popper_upgrade.tests.fakes import FakeEmbedder, run_config

pytestmark = pytest.mark.db


def test_summary_from_bundle_matches_db_report(db, tmp_path):
    export_snapshot(db, tmp_path / "snapshot.json")
    snapshot = SnapshotSkillSource(tmp_path / "snapshot.json", FakeEmbedder(model_name=DEFAULT_EMBEDDING_MODEL))
    sink = BundleSink(tmp_path / "bundle", snapshot)
    experiment_id = str(uuid.uuid4())
    prompts = {"curator": "c" * 64, "reviewer": "r" * 64, "extractor": "e" * 64}
    spec = {"prompt_versions": prompts}
    sink.create_experiment({"id": experiment_id, "name": "bundle_report", "spec": spec, "status": "open"})
    outcomes = (
        ("small", "baseline", True, True, False, "completed"),
        ("small", "baseline", False, False, True, "completed"),
        ("small", "baseline", True, False, True, "completed"),
        ("small", "baseline", None, True, False, "error"),
        ("large", "skills", True, True, False, "completed"),
    )
    for index, (model, arm, res, answer, permuted, status) in enumerate(outcomes):
        run_id = sink.create_run(
            run_config(
                experiment_id=experiment_id,
                model_name=model,
                arm=arm,
                answer=answer,
                permuted=permuted,
                task="task",
                replicate=index,
                hypothesis=f"H{index}",
            )
        )
        fields = {"status": status, "tokens_in": 100 + index, "tokens_out": 20, "n_tests": 2, "wall_time_s": 5.0}
        if status == "completed":
            fields["res"] = res
        sink.finish_run(run_id, fields)
        review = {"round": 1, "attempt": 1 + index % 2, "proposal": "p", "approved": bool(index % 2), "reason": "r"}
        sink.add_design_reviews(run_id, [review])
    import_bundle(db, tmp_path / "bundle")

    from_db = build_report(db, experiment_id)
    from_bundle = report_from_bundle(tmp_path / "bundle")

    assert json.loads(json.dumps(from_bundle, default=str)) == json.loads(json.dumps(from_db, default=str))
