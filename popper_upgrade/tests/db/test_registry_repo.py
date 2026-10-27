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
    total = len(json.loads((SEED_DIR / "models.json").read_text(encoding="utf-8")))
    assert first == {"registry.models": total, "registry.hypotheses": 5, "registry.datasets": 3}


def test_seeded_lookups(db):
    registry = RegistryRepository(db)
    registry.seed_from(SEED_DIR)

    cluster = {m["name"] for m in registry.models("cluster")}
    assert {"casperhansen/llama-3.3-70b-instruct-awq", "Qwen/Qwen2.5-72B-Instruct-AWQ"} <= cluster
    assert {m["backend"] for m in registry.models("cluster")} == {"vllm"}
    assert registry.model("Qwen/Qwen3.5-9B")["request_overrides"] == {
        "set": {"chat_template_kwargs": {"enable_thinking": False}}
    }
    assert registry.model("gpt-6-luna")["request_overrides"]["set"] == {"model": "gpt-6-luna"}
    assert registry.model("casperhansen/llama-3.3-70b-instruct-awq")["quantization"] is None
    assert registry.model("gpt-4o")["backend"] == "openai"
    assert len(registry.hypothesis_set("gender_v1")) == 5
    assert registry.dataset("winobias")["permutable_columns"] == ["subset"]
    assert registry.dataset("bbq")["permutable_columns"] == ["question"]
    assert registry.dataset("stereoset")["permutable_columns"] == ["label"]
    with pytest.raises(KeyError):
        registry.model("missing")


def test_paid_models_round_trip_pricing_budget_and_the_native_backend(db):
    registry = RegistryRepository(db)
    registry.seed_from(SEED_DIR)

    sonnet = registry.model("claude-sonnet-5-5")
    sol = registry.model("gpt-6-sol")

    assert sonnet["backend"] == "anthropic_messages"
    assert sonnet["pricing"] == {"input": 2, "cache_read": 0.1, "cache_write": 2.5, "output": 10}
    assert sonnet["request_overrides"]["set"]["thinking"] == {"type": "between_tools"}
    assert (sonnet["budget_eur"], sol["budget_eur"]) == (20, 20)
    assert sol["pricing"] == {"input": 2, "cache_read": 0.2, "output": 10}
    assert registry.model("qwen-large")["pricing"] is None


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
