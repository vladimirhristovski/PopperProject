import json
import uuid
from dataclasses import replace

from cluster import parts as parts_module
from popper_upgrade.db.bundle import BundleSink
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import run_config


def stage_part(bundle, destination, previous):
    runs, candidates = parts_module.stage_part(bundle, destination, previous)
    record = {"part": len(previous) + 1, "runs": [run["id"] for run in runs], "candidates": candidates}
    return destination, [*previous, record]


def build_two_parts(tmp_path, snapshot, library_skill_id):
    bundle = tmp_path / "bundle"
    sink = BundleSink(bundle, snapshot)
    experiment_id = str(uuid.uuid4())
    sink.create_experiment({"id": experiment_id, "name": "parts", "spec": {"phase": "evaluate"}, "status": "open"})
    learn_id = sink.create_run(
        run_config(environment="cluster", experiment_id=experiment_id, task="t", replicate=0, hypothesis="learn h")
    )
    learned = Skill(
        id=str(uuid.uuid4()),
        applicability="P",
        design_template="t",
        preconditions="",
        domain="d",
        source_hypothesis="P",
        source_run_id=learn_id,
    )
    candidate_id = sink.add_candidate_skill(replace(learned))
    sink.add_skill_decision(
        {"skill_id": candidate_id, "trigger": "entry", "decider": "curator", "verdict": "approve", "reason": "ok"}
    )
    sink.finish_run(learn_id, {"res": True, "res_stat": 1.0})
    first, previous = stage_part(bundle, tmp_path / "part1", [])
    frozen_id = str(uuid.uuid4())
    info = {
        "snapshot_id": frozen_id,
        "created_at": "2026-10-19T00:00:00+00:00",
        "embedding_model": snapshot.embedding_model,
        "content_hash": "1" * 64,
        "skill_ids": [library_skill_id, candidate_id],
    }
    (bundle / "freeze.json").write_text(json.dumps(info), encoding="utf-8")
    eval_id = sink.create_run(
        run_config(
            environment="cluster",
            experiment_id=experiment_id,
            task="t",
            replicate=0,
            hypothesis="eval h",
            library_snapshot_id=frozen_id,
        )
    )
    sink.add_run_skills(eval_id, [{"skill_id": candidate_id, "role": "injected", "round": 1, "similarity": 0.5}])
    sink.add_design_reviews(eval_id, [{"round": 1, "attempt": 1, "proposal": "p", "approved": True, "reason": "r"}])
    sink.add_skill_decision(
        {
            "skill_id": candidate_id,
            "run_id": eval_id,
            "trigger": "pre_run",
            "decider": "curator",
            "verdict": "approve",
            "reason": "ok",
        }
    )
    sink.finish_run(eval_id, {"res": False, "res_stat": 2.0})
    second, _ = stage_part(bundle, tmp_path / "part2", previous)
    return {
        "first": first,
        "second": second,
        "learn_id": learn_id,
        "eval_id": eval_id,
        "candidate_id": candidate_id,
        "frozen_id": frozen_id,
        "experiment_id": experiment_id,
    }
