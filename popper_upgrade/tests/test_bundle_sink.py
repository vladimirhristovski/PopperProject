import json

from popper_upgrade.db.bundle import BundleSink
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, write_snapshot


def _skill(skill_id, applicability, approved=True):
    return Skill(
        id=skill_id,
        applicability=applicability,
        design_template="t",
        preconditions="",
        domain="d",
        source_hypothesis=applicability,
        approved=approved,
    )


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sink(tmp_path, skills=()):
    embedder = FakeEmbedder()
    snapshot = SnapshotSkillSource(write_snapshot(tmp_path / "snap.json", list(skills), embedder), embedder)
    return BundleSink(tmp_path / "bundle", snapshot, git_sha="abc123"), snapshot


def test_bundle_sink_writes_jsonl_lines(tmp_path):
    sink, snapshot = _sink(tmp_path, [_skill("s1", "A")])

    run_id = sink.create_run({"hypothesis": "H", "arm": "baseline"})
    sink.add_run_skills(run_id, [{"skill_id": "s1", "role": "injected", "round": 1, "similarity": 0.5}])
    sink.add_design_reviews(run_id, [{"round": 1, "attempt": 1, "proposal": "p", "approved": True, "reason": "r"}])
    decision_id = sink.add_skill_decision(
        {"skill_id": "s1", "trigger": "pre_run", "decider": "curator", "verdict": "approve", "reason": "ok"}
    )
    sink.finish_run(run_id, {"res": True, "res_stat": 3.0})
    candidate_id = sink.add_candidate_skill(_skill("", "B", approved=None))

    bundle = tmp_path / "bundle"
    runs = _lines(bundle / "runs.jsonl")
    assert [r["status"] for r in runs] == ["running", "completed"]
    assert runs[-1]["id"] == run_id
    assert runs[-1]["res"] is True
    assert runs[-1]["library_snapshot_id"] == snapshot.snapshot_id
    assert runs[-1]["environment"] == "cluster"
    assert _lines(bundle / "run_skills.jsonl")[0]["run_id"] == run_id
    assert _lines(bundle / "design_reviews.jsonl")[0]["proposal"] == "p"
    assert _lines(bundle / "skill_decisions.jsonl")[0]["id"] == decision_id
    candidate = _lines(bundle / "candidate_skills.jsonl")[0]
    assert candidate["id"] == candidate_id
    assert len(candidate["embedding"]) == 384
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["snapshot_id"] == snapshot.snapshot_id
    assert manifest["git_sha"] == "abc123"
    assert set(manifest) == {"snapshot_id", "created_at", "git_sha", "host"}


def test_candidate_dedups_against_snapshot_and_bundle(tmp_path):
    sink, _ = _sink(tmp_path, [_skill("s1", "A", approved=False)])

    assert sink.add_candidate_skill(_skill("", "A", approved=None)) is None
    assert sink.add_candidate_skill(_skill("", "B", approved=None)) is not None
    assert sink.add_candidate_skill(_skill("", "B", approved=None)) is None
    assert len(_lines(tmp_path / "bundle" / "candidate_skills.jsonl")) == 1


def test_candidate_dedup_survives_reopening_bundle(tmp_path):
    sink, snapshot = _sink(tmp_path)
    sink.add_candidate_skill(_skill("", "B", approved=None))

    reopened = BundleSink(tmp_path / "bundle", snapshot)

    assert reopened.add_candidate_skill(_skill("", "B", approved=None)) is None
