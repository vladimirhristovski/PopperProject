import uuid

import pytest

import popper_upgrade.runner as runner_module
from popper_upgrade.db.registry_repo import RegistryRepository
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.session import RunSession, RunSpec
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeCurator, FakeEmbedder, FakeExtractor, FakePopper

pytestmark = pytest.mark.db

HYPOTHESIS = "women are promoted less often than men"


def _skill(applicability, source_hypothesis="unrelated source"):
    return Skill(
        id=str(uuid.uuid4()),
        applicability=applicability,
        design_template="compare promotion rates by group",
        preconditions="two groups",
        domain="social science",
        source_hypothesis=source_hypothesis,
    )


def _spec(**overrides):
    fields = {
        "hypothesis": HYPOTHESIS,
        "split": "train",
        "arm": "skills",
        "model": {"name": "fake-model"},
        "popper_kwargs": {"llm": "fake-model", "domain": "social science"},
        "register_kwargs": {"data_path": "data/promotions", "loader_type": "custom"},
        "configure_kwargs": {"alpha": 0.1, "max_num_of_tests": 3, "max_retry": 1, "time_limit": 1},
        "skills_enabled": True,
    }
    fields.update(overrides)
    return RunSpec(**fields)


@pytest.fixture
def repos(db, monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", FakePopper)
    monkeypatch.setattr(FakePopper, "failure", None)
    skills = SkillRepository(db, FakeEmbedder())
    return skills, RunRepository(db, skills), RegistryRepository(db)


def _library_skill(skills, applicability, approved=True):
    skill_id = skills.add_candidate_skill(_skill(applicability))
    skills.set_approval(skill_id, approved, "curator", "fine", "entry")
    return skill_id


def test_human_override_is_run_scoped(db, repos, tmp_path):
    skills, runs, registry = repos
    skill_id = _library_skill(skills, HYPOTHESIS)
    session = RunSession(
        _spec(human_review=True),
        skills,
        runs,
        FakeCurator(rejected={skill_id}),
        None,
        None,
        tmp_path,
        overrides_provider=lambda result: {skill_id: True},
        prompt_version_resolver=registry.prompt_version,
    )

    run_id = session.run()

    assert skills.get(skill_id).approved is True
    roles = {row["role"] for row in runs.run_skills(run_id)}
    assert {"curator_rejected", "human_override_approve", "injected"} <= roles
    decisions = db.execute(
        "select decider, verdict, trigger, prompt_version_id::text as pv from skills.skill_decisions "
        "where run_id = %s order by decider",
        (run_id,),
    ).fetchall()
    assert [(d["decider"], d["verdict"], d["trigger"]) for d in decisions] == [
        ("curator", "reject", "pre_run"),
        ("human", "approve", "pre_run"),
    ]
    assert decisions[0]["pv"] is not None
    assert runs.get_run(run_id)["status"] == "completed"


def test_curator_rejection_does_not_change_library_flag(db, repos, tmp_path):
    skills, runs, registry = repos
    skill_id = _library_skill(skills, HYPOTHESIS)
    session = RunSession(
        _spec(), skills, runs, FakeCurator(rejected={skill_id}), None, None, tmp_path,
        prompt_version_resolver=registry.prompt_version,
    )

    run_id = session.run()

    assert skills.get(skill_id).approved is True
    assert {row["skill_id"]: row["role"] for row in runs.run_skills(run_id)}[skill_id] == "curator_rejected"


def test_learned_candidate_is_approved_at_entry_and_usage_recorded(db, repos, tmp_path):
    skills, runs, registry = repos
    skill_id = _library_skill(skills, HYPOTHESIS)
    extractor = FakeExtractor([_skill("a brand new general strategy")])
    session = RunSession(
        _spec(), skills, runs, FakeCurator(), None, extractor, tmp_path,
        prompt_version_resolver=registry.prompt_version, usage_recorder=skills.record_usage,
    )

    run_id = session.run()

    [candidate] = [s for s in skills.list_skills("any") if s.source_run_id == run_id]
    assert candidate.approved is True
    assert candidate.decided_by == "curator"
    assert skills.get(skill_id).times_used == 1
    run = runs.get_run(run_id)
    assert run["res"] is True
    assert run["n_tests"] == 1
    assert run["trajectory_path"].endswith(f"{run_id}.json")


def test_rejected_candidate_stays_in_library_as_rejected(db, repos, tmp_path):
    skills, runs, registry = repos
    extractor = FakeExtractor([_skill("another general strategy")])
    session = RunSession(
        _spec(), skills, runs, FakeCurator(candidate_approved=False), None, extractor, tmp_path,
        prompt_version_resolver=registry.prompt_version,
    )

    run_id = session.run()

    [candidate] = [s for s in skills.list_skills("any") if s.source_run_id == run_id]
    assert candidate.approved is False


def test_validate_error_is_stored(db, repos, tmp_path, monkeypatch):
    skills, runs, registry = repos
    skill_id = _library_skill(skills, HYPOTHESIS)
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    session = RunSession(
        _spec(), skills, runs, FakeCurator(), None, None, tmp_path, prompt_version_resolver=registry.prompt_version
    )

    run_id = session.run()

    run = runs.get_run(run_id)
    assert run["status"] == "error"
    assert "model down" in run["error"]
    assert skill_id in {row["skill_id"] for row in runs.run_skills(run_id)}
