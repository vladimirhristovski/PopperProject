import json
import tarfile
import uuid
from pathlib import Path

import pytest

from popper_upgrade.app.context import REGISTRY_DIR
from popper_upgrade.db.bundle import BundleSink, import_bundle
from popper_upgrade.db.registry_repo import RegistryRepository, prompt_hash, upsert_model_row
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, run_config, write_snapshot
from popper_upgrade.tests.part_fixtures import stage_part

pytestmark = pytest.mark.db

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PARTS = {1: FIXTURES / "proof_part01.tar.gz", 2: FIXTURES / "proof_part02.tar.gz"}
REGISTRY_TABLES = (
    "registry.models",
    "registry.hypothesis_sets",
    "registry.hypotheses",
    "registry.datasets",
    "registry.prompt_versions",
    "registry.library_snapshots",
)
RUN_TABLES = ("runs.experiments", "runs.runs", "runs.design_reviews")


def _count(db, table):
    return db.execute(f"select count(*) as n from {table}").fetchone()["n"]


def _counts(db, tables):
    return {table: _count(db, table) for table in tables}


def _archive_runs(archive):
    runs = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if member.name.endswith("/runs.jsonl"):
                for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                    row = json.loads(line)
                    runs[row["id"]] = {**runs.get(row["id"], {}), **row}
    return runs


def _dangling(db, column, table):
    return _count(
        db, f"runs.runs r left join {table} t on t.id = r.{column} where r.{column} is not null and t.id is null"
    )


@pytest.fixture
def seeded(db):
    RegistryRepository(db).seed_from(REGISTRY_DIR)


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_proof_parts_import_into_a_freshly_seeded_registry(db, seeded, order):
    expected = {}
    for number in order:
        expected.update(_archive_runs(PARTS[number]))
    before = _counts(db, REGISTRY_TABLES)

    for number in order:
        import_bundle(db, PARTS[number])

    assert {row["id"] for row in db.execute("select id::text as id from runs.runs").fetchall()} >= set(expected)
    assert _dangling(db, "model_id", "registry.models") == 0
    assert _dangling(db, "hypothesis_id", "registry.hypotheses") == 0
    assert _dangling(db, "library_snapshot_id", "registry.library_snapshots") == 0
    assert _count(db, "registry.models where name = 'qwen3b-proof'") == 1
    after_first = _counts(db, REGISTRY_TABLES + RUN_TABLES)

    for number in order:
        again = import_bundle(db, PARTS[number])
        assert again["runs"] == 0
        assert "registry" not in again

    assert _counts(db, REGISTRY_TABLES + RUN_TABLES) == after_first
    assert after_first["registry.models"] == before["registry.models"] + 1


def test_proof_part_reuses_a_model_that_exists_under_another_id(db, seeded):
    existing = upsert_model_row(
        db,
        {
            "name": "qwen3b-proof",
            "backend": "vllm",
            "served_model_id": "Qwen/Qwen2.5-3B-Instruct-AWQ",
            "size_class": "small",
            "environment": "cluster",
            "port": 8000,
        },
    )
    before = _count(db, "registry.models")

    import_bundle(db, PARTS[1])

    assert _count(db, "registry.models") == before
    rows = db.execute("select model_id::text as model_id from runs.runs where model_name = 'qwen3b-proof'").fetchall()
    assert {row["model_id"] for row in rows} == {existing["id"]}


def _decision_version(db, run_id):
    row = db.execute(
        "select prompt_version_id::text as version from skills.skill_decisions where run_id = %s", (run_id,)
    ).fetchone()
    return row["version"]


def _registry_snapshot(tmp_path, embedder):
    model = {
        "id": str(uuid.uuid4()),
        "name": "shipped-model",
        "backend": "vllm",
        "served_model_id": "org/shipped",
        "size_class": "small",
        "environment": "cluster",
        "port": 8000,
        "default_temperature": None,
        "base_url": None,
        "api_key_env": None,
        "strip_tools": True,
        "min_disk_gb": None,
        "quantization": None,
        "dtype": None,
        "max_model_len": None,
        "system_prefix": None,
        "request_overrides": None,
        "serving": {"weights_gb": 1.0},
    }
    hypothesis = {"id": str(uuid.uuid4()), "text": "shipped hypothesis", "split": "test", "ground_truth": True}
    dataset = {
        "id": str(uuid.uuid4()),
        "name": "shipped_dataset",
        "path": "data/shipped.csv",
        "loader_type": "custom",
        "schema": {"n_rows": 3},
        "permutable_columns": ["a"],
    }
    version = {"id": str(uuid.uuid4()), "role": "curator", "text": "curate", "hash": prompt_hash("curator", "curate")}
    snapshot_id = str(uuid.uuid4())
    path = write_snapshot(
        tmp_path / "snapshot.json",
        [],
        embedder,
        snapshot_id=snapshot_id,
        models=[model],
        hypothesis_sets={"shipped_set": [hypothesis]},
        datasets=[dataset],
        prompt_versions=[version],
    )
    return path, model, hypothesis, dataset, version, snapshot_id


def _new_format_archive(tmp_path):
    embedder = FakeEmbedder()
    path, model, hypothesis, dataset, version, snapshot_id = _registry_snapshot(tmp_path, embedder)
    snapshot = SnapshotSkillSource(path, embedder)
    bundle = tmp_path / "bundle"
    sink = BundleSink(bundle, snapshot)
    experiment_id = str(uuid.uuid4())
    sink.create_experiment({"id": experiment_id, "name": "shipped", "spec": {}, "status": "open"})
    run_id = sink.create_run(
        run_config(
            environment="cluster",
            experiment_id=experiment_id,
            task="t",
            replicate=0,
            model_name=model["name"],
            model_id=model["id"],
            hypothesis=hypothesis["text"],
            hypothesis_id=hypothesis["id"],
            datasets=[dataset["name"]],
        )
    )
    skill = Skill(
        id="", applicability="shipped skill", design_template="t", preconditions="", domain="d", source_hypothesis="x"
    )
    decision_skill = sink.add_candidate_skill(skill)
    sink.add_skill_decision(
        {
            "skill_id": decision_skill,
            "run_id": run_id,
            "trigger": "pre_run",
            "decider": "curator",
            "verdict": "approve",
            "reason": "ok",
            "prompt_version_id": version["id"],
        }
    )
    sink.finish_run(run_id, {"res": True, "res_stat": 4.0})
    staged, _ = stage_part(bundle, tmp_path / "part", [])
    archive = tmp_path / "part.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staged, arcname="part")
    return archive, run_id, model, hypothesis, dataset, version, snapshot_id, staged


def test_every_part_ships_the_registry_it_references(tmp_path):
    archive, _, model, hypothesis, dataset, version, snapshot_id, _ = _new_format_archive(tmp_path)

    with tarfile.open(archive) as tar:
        shipped = json.loads(tar.extractfile("part/registry.json").read().decode("utf-8"))

    assert shipped["library_snapshot"]["id"] == snapshot_id
    assert [m["name"] for m in shipped["models"]] == [model["name"]]
    assert shipped["hypothesis_sets"]["shipped_set"][0]["id"] == hypothesis["id"]
    assert [d["name"] for d in shipped["datasets"]] == [dataset["name"]]
    assert [v["hash"] for v in shipped["prompt_versions"]] == [version["hash"]]


def test_new_format_part_imports_into_an_empty_registry(db, tmp_path):
    archive, run_id, model, hypothesis, dataset, version, snapshot_id, _ = _new_format_archive(tmp_path)

    first = import_bundle(db, archive)
    second = import_bundle(db, archive)

    assert first["registry"] == {
        "models": 1,
        "hypotheses": 1,
        "datasets": 1,
        "prompt_versions": 1,
        "library_snapshots": 1,
    }
    assert "registry" not in second
    run = db.execute(
        "select model_id::text as model_id, hypothesis_id::text as hypothesis_id, library_snapshot_id::text as snap "
        "from runs.runs where id = %s",
        (run_id,),
    ).fetchone()
    assert run == {"model_id": model["id"], "hypothesis_id": hypothesis["id"], "snap": snapshot_id}
    assert _count(db, f"registry.datasets where name = '{dataset['name']}'") == 1
    assert _decision_version(db, run_id) == version["id"]


def test_new_format_part_matches_existing_rows_by_natural_key(db, tmp_path):
    archive, run_id, model, hypothesis, dataset, version, snapshot_id, _ = _new_format_archive(tmp_path)
    other_model = upsert_model_row(db, {**model, "id": None})
    set_id = db.execute(
        "insert into registry.hypothesis_sets (id, name) values (%s, 'shipped_set') returning id::text as id",
        (str(uuid.uuid4()),),
    ).fetchone()["id"]
    other_hypothesis = db.execute(
        "insert into registry.hypotheses (id, set_id, text, split) "
        "values (%s, %s, %s, 'test') returning id::text as id",
        (str(uuid.uuid4()), set_id, hypothesis["text"]),
    ).fetchone()["id"]
    other_version = db.execute(
        "insert into registry.prompt_versions (id, role, text, hash) values (%s, %s, %s, %s) returning id::text as id",
        (str(uuid.uuid4()), version["role"], version["text"], version["hash"]),
    ).fetchone()["id"]
    db.execute(
        "insert into registry.datasets (id, name, path, loader_type, schema, permutable_columns) "
        "values (%s, %s, 'p', 'custom', '{}', '[]')",
        (str(uuid.uuid4()), dataset["name"]),
    )
    before = _counts(db, REGISTRY_TABLES)

    result = import_bundle(db, archive)

    assert result["registry"] == {"library_snapshots": 1}
    after = _counts(db, REGISTRY_TABLES)
    assert after == {**before, "registry.library_snapshots": before["registry.library_snapshots"] + 1}
    run = db.execute(
        "select model_id::text as model_id, hypothesis_id::text as hypothesis_id from runs.runs where id = %s",
        (run_id,),
    ).fetchone()
    assert run == {"model_id": str(other_model["id"]), "hypothesis_id": other_hypothesis}
    assert _decision_version(db, run_id) == other_version


def test_part_without_its_registry_is_rejected_clearly(db, tmp_path):
    _, _, model, _, _, _, snapshot_id, staged = _new_format_archive(tmp_path)
    (staged / "registry.json").unlink()

    with pytest.raises(ValueError, match="does not carry the library snapshot"):
        import_bundle(db, staged)

    db.execute(
        "insert into registry.library_snapshots (id, embedding_model, skill_ids, content_hash) "
        "values (%s, 'e', '[]', 'h')",
        (snapshot_id,),
    )
    with pytest.raises(ValueError, match=f"does not carry the model {model['name']}"):
        import_bundle(db, staged)
