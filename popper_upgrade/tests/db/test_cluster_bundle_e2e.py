import json
import uuid
from pathlib import Path

import pytest

import popper_upgrade
import popper_upgrade.run as run_module
import popper_upgrade.runner as runner_module
from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.registry_repo import RegistryRepository
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeCurator, FakeEmbedder, FakeExtractor, FakePopper, patch_agent_classes

pytestmark = pytest.mark.db

REGISTRY_DIR = Path(popper_upgrade.__file__).parent / "registry"
CLUSTER_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"


def _skill(applicability, source_hypothesis="unrelated source"):
    return Skill(
        id=str(uuid.uuid4()),
        applicability=applicability,
        design_template="compare group means",
        preconditions="two groups",
        domain="social science",
        source_hypothesis=source_hypothesis,
    )


@pytest.fixture
def world(db, monkeypatch, tmp_path):
    monkeypatch.setattr(runner_module, "Popper", FakePopper)
    monkeypatch.setattr(FakePopper, "failure", None)
    patch_agent_classes(monkeypatch, run_module)
    candidate = _skill("a brand new general strategy")
    monkeypatch.setattr(run_module, "SkillExtractor", lambda llm, **kw: FakeExtractor([candidate]))
    registry = RegistryRepository(db)
    registry.seed_from(REGISTRY_DIR)
    skills = SkillRepository(db, FakeEmbedder())
    return skills, registry, tmp_path


def _library_skill(skills, applicability):
    skill_id = skills.add_candidate_skill(_skill(applicability))
    skills.set_approval(skill_id, True, "curator", "fine", "entry")
    return skill_id


def test_cluster_bundle_round_trip(db, world):
    skills, registry, tmp_path = world
    hypotheses = registry.hypothesis_set("gender_v1")[:2]
    skill_id = _library_skill(skills, hypotheses[0]["text"])
    prompt_version_id = registry.prompt_version("curator", FakeCurator.prompt_text)
    snapshot_path = tmp_path / "snapshot.json"
    export_snapshot(db, snapshot_path)
    snapshot = SnapshotSkillSource(snapshot_path, FakeEmbedder())
    model = next(m for m in snapshot.registry["models"] if m["name"] == CLUSTER_MODEL)
    bundle = tmp_path / "bundle"

    run_ids = run_module.run_hypotheses(
        model,
        [h for h in snapshot.registry["hypothesis_sets"]["gender_v1"] if h["id"] in {x["id"] for x in hypotheses}],
        "skills_reviewer",
        snapshot,
        BundleSink(bundle, snapshot),
        "cluster",
        learn=True,
        data_dir=tmp_path / "data",
        trajectory_dir=tmp_path / "trajectories",
    )

    assert len(run_ids) == 2
    before = db.execute("select count(*) as n from runs.runs").fetchone()["n"]
    counts = import_bundle(db, bundle)
    assert counts["runs"] == 2
    assert counts["candidate_skills"] == 1
    assert db.execute("select count(*) as n from runs.runs").fetchone()["n"] == before + 2

    run = db.execute(
        "select status, environment, arm, skills_enabled, reviewer_enabled, human_review, model_id::text as model_id, "
        "hypothesis_id::text as hypothesis_id, library_snapshot_id::text as snap, res, n_tests, reviewer_rejections "
        "from runs.runs where id = %s",
        (run_ids[0],),
    ).fetchone()
    assert run == {
        "status": "completed",
        "environment": "cluster",
        "arm": "skills_reviewer",
        "skills_enabled": True,
        "reviewer_enabled": True,
        "human_review": False,
        "model_id": model["id"],
        "hypothesis_id": hypotheses[0]["id"],
        "snap": snapshot.snapshot_id,
        "res": True,
        "n_tests": 1,
        "reviewer_rejections": 0,
    }
    roles = {
        row["role"]
        for row in db.execute(
            "select role from runs.run_skills where run_id = %s and skill_id = %s", (run_ids[0], skill_id)
        ).fetchall()
    }
    assert roles == {"curator_approved", "injected"}
    reviews = db.execute("select approved from runs.design_reviews where run_id = %s", (run_ids[0],)).fetchall()
    assert [r["approved"] for r in reviews] == [True]
    decisions = db.execute(
        "select trigger, decider, verdict, prompt_version_id::text as pv from skills.skill_decisions "
        "where run_id = %s and skill_id = %s",
        (run_ids[0], skill_id),
    ).fetchall()
    assert decisions == [{"trigger": "pre_run", "decider": "curator", "verdict": "approve", "pv": prompt_version_id}]
    entry = db.execute(
        "select s.approved, s.decided_by, d.prompt_version_id::text as pv from skills.skills s "
        "join skills.skill_decisions d on d.skill_id = s.id and d.trigger = 'entry' where s.source_run_id = %s",
        (run_ids[0],),
    ).fetchall()
    assert entry == [{"approved": True, "decided_by": "curator", "pv": prompt_version_id}]
    trajectory = json.loads((tmp_path / "trajectories" / f"{run_ids[0]}.json").read_text(encoding="utf-8"))
    assert trajectory["tracked_tests"] == ["design one"]


def test_postgres_session_wiring(db, world, monkeypatch):
    skills, registry, tmp_path = world
    hypothesis = {**registry.hypothesis_set("gender_v1")[0], "split": "train"}
    skill_id = _library_skill(skills, hypothesis["text"])
    prompt_version_id = registry.prompt_version("curator", FakeCurator.prompt_text)
    counter = UsageCounter()

    class UsagePopper(FakePopper):
        def validate(self, text):
            counter.add({"prompt_tokens": 12, "completion_tokens": 3})
            return super().validate(text)

    monkeypatch.setattr(runner_module, "Popper", UsagePopper)
    runs = RunRepository(db, skills)

    [run_id] = run_module.run_hypotheses(
        registry.model("llama3.1:8b"),
        [hypothesis],
        "skills",
        skills,
        runs,
        "laptop",
        usage=counter,
        data_dir=tmp_path / "data",
        trajectory_dir=tmp_path / "trajectories",
    )

    run = runs.get_run(run_id)
    assert (run["status"], run["tokens_in"], run["tokens_out"]) == ("completed", 12, 3)
    assert skills.get(skill_id).times_used == 1
    pv = db.execute(
        "select prompt_version_id::text as pv from skills.skill_decisions where run_id = %s", (run_id,)
    ).fetchone()
    assert pv["pv"] == prompt_version_id
