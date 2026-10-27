import json
import tarfile
import uuid
from dataclasses import replace

import psycopg
import pytest

from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, run_config
from popper_upgrade.tests.part_fixtures import build_two_parts

pytestmark = pytest.mark.db

TABLES = ("skills.skills", "runs.runs", "runs.run_skills", "runs.design_reviews", "skills.skill_decisions")


def _skill(text):
    return Skill(id="", applicability=text, design_template="t", preconditions="", domain="d", source_hypothesis=text)


@pytest.fixture
def library(db, tmp_path):
    embedder = FakeEmbedder()
    repo = SkillRepository(db, embedder)
    skill_id = repo.add_candidate_skill(replace(_skill("A"), id=str(uuid.uuid4())))
    repo.set_approval(skill_id, True, "curator", "fine", "entry")
    path = tmp_path / "snapshot.json"
    export_snapshot(db, path)
    return SnapshotSkillSource(path, embedder), skill_id


def _write_bundle(tmp_path, snapshot, skill_id):
    sink = BundleSink(tmp_path / "bundle", snapshot)
    run_id = sink.create_run(run_config(environment="cluster"))
    sink.add_run_skills(run_id, [{"skill_id": skill_id, "role": "injected", "round": 1, "similarity": 0.4}])
    sink.add_design_reviews(run_id, [{"round": 1, "attempt": 1, "proposal": "p", "approved": True, "reason": "r"}])
    sink.add_skill_decision(
        {
            "skill_id": skill_id,
            "run_id": run_id,
            "trigger": "pre_run",
            "decider": "curator",
            "verdict": "approve",
            "reason": "ok",
        }
    )
    candidate_id = sink.add_candidate_skill(_skill("B"))
    sink.finish_run(run_id, {"res": True, "res_stat": 7.0})
    return run_id, candidate_id


def _count(db, table):
    return db.execute(f"select count(*) as n from {table}").fetchone()["n"]


def test_import_bundle_twice_is_idempotent(db, tmp_path, library):
    snapshot, skill_id = library
    run_id, candidate_id = _write_bundle(tmp_path, snapshot, skill_id)
    before = {t: _count(db, t) for t in TABLES}

    first = import_bundle(db, tmp_path / "bundle")
    after_first = {t: _count(db, t) for t in before}
    second = import_bundle(db, tmp_path / "bundle")

    assert first == {
        "experiments": 0,
        "candidate_skills": 1,
        "runs": 1,
        "run_skills": 1,
        "design_reviews": 1,
        "skill_decisions": 1,
    }
    assert second == dict.fromkeys(first, 0)
    assert {t: _count(db, t) for t in before} == after_first
    assert after_first["skills.skills"] == before["skills.skills"] + 1
    run = db.execute(
        "select status, res, library_snapshot_id::text as snap from runs.runs where id = %s", (run_id,)
    ).fetchone()
    assert run == {"status": "completed", "res": True, "snap": snapshot.snapshot_id}
    candidate = db.execute("select approved from skills.skills where id = %s", (candidate_id,)).fetchone()
    assert candidate["approved"] is None
    events = db.execute("select count(*) as n from audit.events where action = 'bundle_import'").fetchone()["n"]
    assert events == 2


def test_import_rolls_back_on_bad_row(db, tmp_path, library):
    snapshot, skill_id = library
    run_id, candidate_id = _write_bundle(tmp_path, snapshot, skill_id)
    sink = BundleSink(tmp_path / "bundle", snapshot)
    sink.add_run_skills(run_id, [{"skill_id": str(uuid.uuid4()), "role": "injected", "round": 2, "similarity": 0.1}])

    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        import_bundle(db, tmp_path / "bundle")

    assert db.execute("select count(*) as n from skills.skills where id = %s", (candidate_id,)).fetchone()["n"] == 0
    assert db.execute("select count(*) as n from runs.runs where id = %s", (run_id,)).fetchone()["n"] == 0


def test_import_applies_entry_decisions_once(db, tmp_path, library):
    snapshot, skill_id = library
    sink = BundleSink(tmp_path / "bundle", snapshot)
    approved_id = sink.add_candidate_skill(_skill("C"))
    rejected_id = sink.add_candidate_skill(_skill("D"))
    pending_id = sink.add_candidate_skill(_skill("E"))
    for candidate_id, verdict in ((approved_id, "approve"), (rejected_id, "reject")):
        sink.add_skill_decision(
            {"skill_id": candidate_id, "trigger": "entry", "decider": "curator", "verdict": verdict, "reason": verdict}
        )
    sink.add_skill_decision(
        {"skill_id": skill_id, "trigger": "pre_run", "decider": "curator", "verdict": "reject", "reason": "run only"}
    )

    import_bundle(db, tmp_path / "bundle")

    def state(sid):
        return db.execute("select approved, decided_by from skills.skills where id = %s", (sid,)).fetchone()

    assert state(approved_id) == {"approved": True, "decided_by": "curator"}
    assert state(rejected_id) == {"approved": False, "decided_by": "curator"}
    assert state(pending_id)["approved"] is None
    assert state(skill_id)["approved"] is True

    db.execute("update skills.skills set approved = false, decided_by = 'human' where id = %s", (approved_id,))
    import_bundle(db, tmp_path / "bundle")
    assert state(approved_id) == {"approved": False, "decided_by": "human"}


def test_import_records_the_frozen_snapshot_once(db, tmp_path, library):
    snapshot, skill_id = library
    _write_bundle(tmp_path, snapshot, skill_id)
    frozen_id = str(uuid.uuid4())
    info = {
        "snapshot_id": frozen_id,
        "created_at": "2026-10-19T00:00:00+00:00",
        "embedding_model": snapshot.embedding_model,
        "content_hash": "0" * 64,
        "skill_ids": [skill_id],
    }
    (tmp_path / "bundle" / "freeze.json").write_text(json.dumps(info), encoding="utf-8")

    first = import_bundle(db, tmp_path / "bundle")
    second = import_bundle(db, tmp_path / "bundle")

    assert first["frozen_snapshots"] == 1
    assert second["frozen_snapshots"] == 0
    row = db.execute(
        "select content_hash, skill_ids from registry.library_snapshots where id = %s", (frozen_id,)
    ).fetchone()
    assert row == {"content_hash": "0" * 64, "skill_ids": [skill_id]}


def _pack(directory, archive):
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(directory, arcname="popper_results_parts")
    return archive


@pytest.fixture
def two_parts(tmp_path, library):
    snapshot, skill_id = library
    return build_two_parts(tmp_path, snapshot, skill_id)


def test_parts_import_in_any_order_and_repeated(db, tmp_path, two_parts):
    first, second = two_parts["first"], _pack(two_parts["second"], tmp_path / "part2.tar.gz")
    candidate_id = two_parts["candidate_id"]

    late = import_bundle(db, second)
    assert (late["runs"], late["candidate_skills"], late["frozen_snapshots"]) == (1, 1, 1)
    orphan = db.execute(
        "select source_run_id, approved from skills.skills where id = %s", (candidate_id,)
    ).fetchone()
    assert orphan == {"source_run_id": None, "approved": None}

    early = import_bundle(db, first)
    assert (early["runs"], early["candidate_skills"], early["skill_decisions"]) == (1, 1, 1)
    repaired = db.execute(
        "select source_run_id::text as source_run_id, approved, decided_by from skills.skills where id = %s",
        (candidate_id,),
    ).fetchone()
    assert repaired == {"source_run_id": two_parts["learn_id"], "approved": True, "decided_by": "curator"}

    again = [import_bundle(db, second), import_bundle(db, first), import_bundle(db, second)]
    assert all(set(counts.values()) == {0} for counts in again)
    runs = db.execute(
        "select count(*) as n from runs.runs where experiment_id = %s", (two_parts["experiment_id"],)
    ).fetchone()["n"]
    assert runs == 2
    assert _count(db, "runs.run_skills") >= 1
    snapshot = db.execute(
        "select count(*) as n from registry.library_snapshots where id = %s", (two_parts["frozen_id"],)
    ).fetchone()["n"]
    assert snapshot == 1


def test_parts_in_shipping_order_give_the_same_result(db, tmp_path, two_parts):
    candidate_id = two_parts["candidate_id"]

    import_bundle(db, two_parts["first"])
    import_bundle(db, two_parts["second"])

    row = db.execute(
        "select source_run_id::text as source_run_id, approved from skills.skills where id = %s", (candidate_id,)
    ).fetchone()
    assert row == {"source_run_id": two_parts["learn_id"], "approved": True}
