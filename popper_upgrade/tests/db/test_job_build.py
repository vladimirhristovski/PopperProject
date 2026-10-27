import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import popper_upgrade
from popper_upgrade.db.registry_repo import RegistryRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.eval.run_experiment import current_prompt_hashes
from popper_upgrade.jobs.build import build_job
from popper_upgrade.skills.embedding import DEFAULT_EMBEDDING_MODEL
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder

pytestmark = pytest.mark.db

REGISTRY_DIR = Path(popper_upgrade.__file__).parent / "registry"
MODEL = "casperhansen/llama-3.3-70b-instruct-awq"


def _spec(tmp_path, models=(MODEL,)):
    spec = {
        "name": "built_v1",
        "models": list(models),
        "arms": ["baseline"],
        "source": {"kind": "gender_custom", "hypothesis_set": "gender_v1", "task": "gender_bias"},
        "train_tasks": [],
        "test_tasks": ["gender_bias"],
        "replicates": 1,
        "permutation_seeds": [1],
        "prompt_versions": current_prompt_hashes(),
    }
    path = tmp_path / "spec_source.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


@pytest.fixture
def registry(db):
    repo = RegistryRepository(db)
    repo.seed_from(REGISTRY_DIR)
    return repo


def _embedder(model_name=DEFAULT_EMBEDDING_MODEL):
    return FakeEmbedder(model_name=model_name)


def test_build_writes_snapshot_spec_and_job_json(db, registry, tmp_path):
    out = tmp_path / "jobs" / "demo"

    job = build_job(db, "demo", MODEL, _spec(tmp_path), out, _embedder(), data_dir=tmp_path)

    assert json.loads((out / "job.json").read_text(encoding="utf-8")) == job
    assert {key: job[key] for key in ("name", "model", "spec", "snapshot")} == {
        "name": "demo",
        "model": MODEL,
        "spec": "spec.json",
        "snapshot": "snapshot.json",
    }
    assert {"created_at", "git_sha"} <= set(job)
    snapshot = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))
    model = next(m for m in snapshot["models"] if m["name"] == MODEL)
    assert model["min_disk_gb"] == 45
    assert json.loads((out / "spec.json").read_text(encoding="utf-8"))["name"] == "built_v1"


def test_build_refuses_a_snapshot_derived_from_test_hypotheses(db, registry, tmp_path):
    test_hypothesis = registry.hypothesis_set("gender_v1")[0]["text"]
    skills = SkillRepository(db, FakeEmbedder())
    leaked = Skill(
        id=str(uuid.uuid4()),
        applicability="a general strategy",
        design_template="compare",
        preconditions="",
        domain="social science",
        source_hypothesis=test_hypothesis,
    )
    skill_id = skills.add_candidate_skill(replace(leaked))
    skills.set_approval(skill_id, True, "curator", "fine", "entry")
    out = tmp_path / "jobs" / "leaky"

    with pytest.raises(ValueError, match="derived from test hypothesis"):
        build_job(db, "leaky", MODEL, _spec(tmp_path), out, _embedder("fake-embedder"), data_dir=tmp_path)

    assert not (out / "job.json").exists()
    assert not (out / "snapshot.json").exists()


def test_build_refuses_a_model_outside_the_spec(db, registry, tmp_path):
    spec = _spec(tmp_path, models=["other"])

    with pytest.raises(ValueError, match="not part of built_v1"):
        build_job(db, "demo", MODEL, spec, tmp_path / "out", _embedder(), data_dir=tmp_path)


def test_build_refuses_a_laptop_model(db, registry, tmp_path):
    spec = _spec(tmp_path, models=["llama3.1:8b"])

    with pytest.raises(ValueError, match="not available on"):
        build_job(db, "demo", "llama3.1:8b", spec, tmp_path / "out", _embedder(), data_dir=tmp_path)

    assert not (tmp_path / "out" / "job.json").exists()


def test_qwen_large_job_builds_from_the_qwen_round_spec(db, registry, tmp_path):
    out = tmp_path / "jobs" / "qwen-large"
    spec = REGISTRY_DIR.parent / "eval" / "specs" / "qwen_round.json"

    job = build_job(db, "qwen-large", "qwen-large", spec, out, _embedder(), data_dir=tmp_path)

    assert {key: job[key] for key in ("name", "model", "spec", "snapshot")} == {
        "name": "qwen-large",
        "model": "qwen-large",
        "spec": "spec.json",
        "snapshot": "snapshot.json",
    }
    assert "library_from" not in job
    snapshot = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))
    model = next(m for m in snapshot["models"] if m["name"] == "qwen-large")
    expected = next(
        m for m in json.loads((REGISTRY_DIR / "models.json").read_text(encoding="utf-8")) if m["name"] == "qwen-large"
    )
    for field in ("served_model_id", "quantization", "dtype", "max_model_len", "serving", "request_overrides"):
        assert model[field] == expected[field]
    assert json.loads((out / "spec.json").read_text(encoding="utf-8"))["name"] == "qwen_round"
