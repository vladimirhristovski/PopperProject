import json
from pathlib import Path

import pytest

from popper_upgrade.db.registry_repo import RegistryRepository

pytestmark = pytest.mark.db

SEED_DIR = Path(__file__).resolve().parents[2] / "registry"


def _count(db, table):
    return db.execute(f"select count(*) as n from {table}").fetchone()["n"]


def test_seed_is_idempotent(db):
    registry = RegistryRepository(db)

    registry.seed_from(SEED_DIR)
    first = {t: _count(db, t) for t in ("registry.models", "registry.hypotheses", "registry.datasets")}
    registry.seed_from(SEED_DIR)
    second = {t: _count(db, t) for t in first}

    assert first == second
    assert first == {"registry.models": 6, "registry.hypotheses": 5, "registry.datasets": 3}


def test_seeded_lookups(db):
    registry = RegistryRepository(db)
    registry.seed_from(SEED_DIR)

    assert {m["name"] for m in registry.models("cluster")} == {
        "casperhansen/llama-3.3-70b-instruct-awq",
        "Qwen/Qwen2.5-72B-Instruct-AWQ",
    }
    assert registry.model("gpt-4o")["backend"] == "openai"
    assert len(registry.hypothesis_set("gender_v1")) == 5
    assert registry.dataset("winobias")["permutable_columns"] == ["subset"]
    assert registry.dataset("bbq")["permutable_columns"] == ["question"]
    assert registry.dataset("stereoset")["permutable_columns"] == ["label"]
    with pytest.raises(KeyError):
        registry.model("missing")


def test_seed_files_match_expected_shape():
    models = json.loads((SEED_DIR / "models.json").read_text(encoding="utf-8"))

    assert {m["port"] for m in models if m["backend"] == "vllm"} == {8001}


def test_prompt_version_dedups_by_hash(db):
    registry = RegistryRepository(db)

    first = registry.prompt_version("curator", "You are a curator.")
    again = registry.prompt_version("curator", "You are a curator.")
    other = registry.prompt_version("reviewer", "You are a curator.")

    assert first == again
    assert first != other
    assert _count(db, "registry.prompt_versions") == 2
