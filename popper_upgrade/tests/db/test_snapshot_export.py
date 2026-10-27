import json
import uuid
from pathlib import Path

import pytest

from popper_upgrade.db.registry_repo import RegistryRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder

pytestmark = pytest.mark.db

SEED_DIR = Path(__file__).resolve().parents[2] / "registry"


def _add(repo, text, approved):
    sid = repo.add_candidate_skill(
        Skill(
            id=str(uuid.uuid4()),
            applicability=text,
            design_template="t",
            preconditions="",
            domain="d",
            source_hypothesis=text,
        )
    )
    if approved is not None:
        repo.set_approval(sid, approved, "curator", f"reason {text}", "entry")
    return sid


def test_snapshot_source_matches_repo_ranking(db, tmp_path):
    embedder = FakeEmbedder()
    repo = SkillRepository(db, embedder)
    for text in "ABCDEF":
        _add(repo, text, True)
    _add(repo, "G", False)
    _add(repo, "H", None)
    path = tmp_path / "snapshot.json"

    snapshot_id = export_snapshot(db, path)
    source = SnapshotSkillSource(path, embedder)

    for query in ("A", "C", "unrelated"):
        expected = [(s.id, pytest.approx(sim, abs=1e-5)) for s, sim in repo.retrieve_scored(query, 4)]
        actual = [(s.id, sim) for s, sim in source.retrieve_scored(query, 4)]
        assert actual == expected
    assert source.snapshot_id == snapshot_id
    assert len(source.list_skills(approved=False)) == 1
    assert source.list_skills(approved=None) == []


def test_export_records_snapshot_row_and_registry(db, tmp_path):
    embedder = FakeEmbedder()
    repo = SkillRepository(db, embedder)
    kept = _add(repo, "A", True)
    rejected = _add(repo, "B", False)
    RegistryRepository(db).seed_from(SEED_DIR)
    path = tmp_path / "snapshot.json"

    snapshot_id = export_snapshot(db, path, include_rejected=False)

    payload = json.loads(path.read_text(encoding="utf-8"))
    row = db.execute("select * from registry.library_snapshots where id = %s", (snapshot_id,)).fetchone()
    assert [s["id"] for s in payload["skills"]] == [kept]
    assert rejected not in {s["id"] for s in payload["skills"]}
    assert payload["embedding_model"] == "fake-embedder"
    assert row["skill_ids"] == [kept]
    assert row["embedding_model"] == "fake-embedder"
    assert [d["skill_id"] for d in payload["decisions"]] == [kept]
    assert len(payload["models"]) == len(json.loads((SEED_DIR / "models.json").read_text(encoding="utf-8")))
    assert len(payload["hypothesis_sets"]["gender_v1"]) == 5
    assert {d["name"] for d in payload["datasets"]} == {"winobias", "bbq", "stereoset"}
    assert SnapshotSkillSource(path, embedder).registry["datasets"][0]["permutable_columns"]


def test_export_carries_launch_settings_of_models(db, tmp_path):
    RegistryRepository(db).seed_from(SEED_DIR)
    path = tmp_path / "snapshot.json"

    export_snapshot(db, path)
    models = {row["name"]: row for row in json.loads(path.read_text(encoding="utf-8"))["models"]}

    assert models["Qwen/Qwen3.5-9B"]["request_overrides"]["set"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert models["Qwen/Qwen3.5-9B"]["dtype"] == "bfloat16"
    assert models["casperhansen/llama-3.3-70b-instruct-awq"]["dtype"] is None
    assert models["gpt-6-luna"]["request_overrides"]["set"] == {"model": "gpt-6-luna"}
