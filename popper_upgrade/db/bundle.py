import argparse
import contextlib
import json
import os
import shutil
import socket
import sys
import tarfile
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from popper_upgrade.db.audit import log_event
from popper_upgrade.db.registry_repo import JSON_FIELDS, MODEL_FIELDS, model_values
from popper_upgrade.db.rows import apply_entry_decision, insert_row, run_key

EXPERIMENTS = "experiments.jsonl"
RUNS = "runs.jsonl"
RUN_SKILLS = "run_skills.jsonl"
DESIGN_REVIEWS = "design_reviews.jsonl"
SKILL_DECISIONS = "skill_decisions.jsonl"
CANDIDATE_SKILLS = "candidate_skills.jsonl"
MANIFEST = "manifest.json"
FREEZE_INFO = "freeze.json"
FROZEN_SNAPSHOT = "frozen_snapshot.json"
REGISTRY = "registry.json"
UNSHIPPED_HASH = "unshipped"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class BundleSink:
    def __init__(self, directory, snapshot, git_sha=None, dedup_threshold=0.95):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._snapshot = snapshot
        self._dedup_threshold = dedup_threshold
        self._runs = {}
        self._candidate_vectors = [
            self._unit(entry["embedding"]) for entry in _read_lines(self._dir / CANDIDATE_SKILLS)
        ]
        manifest = self._dir / MANIFEST
        if not manifest.exists():
            manifest.write_text(
                json.dumps(
                    {
                        "snapshot_id": snapshot.snapshot_id,
                        "created_at": _now(),
                        "git_sha": git_sha or os.environ.get("POPPER_GIT_SHA", "unknown"),
                        "host": socket.gethostname(),
                    }
                ),
                encoding="utf-8",
            )
        registry = self._dir / REGISTRY
        if not registry.exists():
            registry.write_text(
                json.dumps({"library_snapshot": snapshot.library_info, **snapshot.registry}), encoding="utf-8"
            )

    @staticmethod
    def _unit(vector):
        array = np.array(vector, dtype=np.float32)
        return array / np.linalg.norm(array)

    def _append(self, name, row):
        with (self._dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def create_experiment(self, experiment):
        known = {row["id"] for row in _read_lines(self._dir / EXPERIMENTS)}
        if experiment["id"] not in known:
            self._append(EXPERIMENTS, {"created_at": _now(), **experiment})
        return experiment["id"]

    def completed_runs(self, experiment_id):
        runs = {}
        for row in _read_lines(self._dir / RUNS):
            runs[row["id"]] = {**runs.get(row["id"], {}), **row}
        return {
            run_key(row)
            for row in runs.values()
            if row.get("experiment_id") == experiment_id and row.get("status") == "completed"
        }

    def create_run(self, config):
        row = {"status": "running", "environment": "cluster", "created_at": _now(), **config}
        row.setdefault("id", str(uuid.uuid4()))
        row.setdefault("library_snapshot_id", self._snapshot.snapshot_id)
        self._runs[row["id"]] = row
        self._append(RUNS, row)
        return row["id"]

    def finish_run(self, run_id, result):
        row = {**self._runs[run_id], "status": "completed", **result, "finished_at": _now()}
        self._runs[run_id] = row
        self._append(RUNS, row)

    def add_run_skills(self, run_id, rows):
        for row in rows:
            self._append(RUN_SKILLS, {"run_id": run_id, **row})

    def add_design_reviews(self, run_id, reviews):
        for review in reviews:
            self._append(DESIGN_REVIEWS, {"id": str(uuid.uuid4()), "run_id": run_id, "created_at": _now(), **review})

    def add_skill_decision(self, decision):
        row = {"id": str(uuid.uuid4()), "created_at": _now(), **decision}
        self._append(SKILL_DECISIONS, row)
        return row["id"]

    def add_candidate_skill(self, skill):
        vector = self._unit(self._snapshot.embedder.encode(skill.applicability))
        nearest = self._snapshot.max_similarity(skill.applicability)
        if nearest is not None and nearest >= self._dedup_threshold:
            return None
        if any(float(existing @ vector) >= self._dedup_threshold for existing in self._candidate_vectors):
            return None
        skill_id = skill.id or str(uuid.uuid4())
        self._append(
            CANDIDATE_SKILLS,
            {
                **asdict(skill),
                "id": skill_id,
                "created_at": skill.created_at or _now(),
                "content_hash": skill.content_hash,
                "embedding_model": self._snapshot.embedding_model,
                "embedding": [float(x) for x in vector],
            },
        )
        self._candidate_vectors.append(vector)
        return skill_id


def _insert_candidate(conn, entry):
    cur = conn.execute(
        "insert into skills.skills (id, applicability, design_template, preconditions, domain, source_hypothesis, "
        "source_run_id, parent_skill_id, approved, decided_by, decision_reason, times_used, times_passed, "
        "content_hash, embedding_model, embedding, created_at) "
        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "on conflict (id) do update set source_run_id = excluded.source_run_id "
        "where skills.skills.source_run_id is null and excluded.source_run_id is not null",
        (
            entry["id"],
            entry["applicability"],
            entry["design_template"],
            entry["preconditions"],
            entry["domain"],
            entry["source_hypothesis"],
            entry.get("source_run_id"),
            entry.get("parent_skill_id"),
            entry.get("approved"),
            entry.get("decided_by"),
            entry.get("decision_reason"),
            entry.get("times_used", 0),
            entry.get("times_passed", 0),
            entry["content_hash"],
            entry["embedding_model"],
            np.array(entry["embedding"], dtype=np.float32),
            entry["created_at"],
        ),
    )
    return cur.rowcount


def _with_known_source_run(conn, entry, bundle_runs):
    run_id = entry.get("source_run_id")
    if run_id is None or run_id in bundle_runs:
        return entry
    if conn.execute("select 1 from runs.runs where id = %s", (run_id,)).fetchone() is not None:
        return entry
    return {**entry, "source_run_id": None}


def _insert_frozen_snapshot(conn, directory):
    path = directory / FREEZE_INFO
    if not path.exists():
        return None
    info = json.loads(path.read_text(encoding="utf-8"))
    cur = conn.execute(
        "insert into registry.library_snapshots (id, created_at, embedding_model, skill_ids, content_hash, path) "
        "values (%s, %s, %s, %s::jsonb, %s, %s) on conflict do nothing",
        (
            info["snapshot_id"],
            info["created_at"],
            info["embedding_model"],
            json.dumps(info["skill_ids"]),
            info["content_hash"],
            None,
        ),
    )
    return cur.rowcount


def _insert_decision(conn, row):
    inserted = insert_row(conn, "skills.skill_decisions", row)
    if inserted and row.get("trigger") == "entry":
        apply_entry_decision(conn, row)
    return inserted


def _safe_extract(archive, destination):
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isreg() or member.isdir()):
                raise ValueError(f"unsafe entry in archive: {member.name}")
        for member in members:
            tar.extract(member, root)


def _bundle_root(directory):
    if (directory / MANIFEST).exists():
        return directory
    candidates = [path for path in directory.iterdir() if path.is_dir() and (path / MANIFEST).exists()]
    if len(candidates) != 1:
        raise ValueError("the archive does not contain a bundle")
    return candidates[0]


def import_bundle(conn, path, trajectory_dir=None):
    path = Path(path)
    if path.is_file():
        with tempfile.TemporaryDirectory() as scratch:
            _safe_extract(path, Path(scratch))
            return _import_directory(conn, _bundle_root(Path(scratch)), trajectory_dir)
    return _import_directory(conn, path, trajectory_dir)


@contextlib.contextmanager
def _opened(path):
    path = Path(path)
    if not path.is_file():
        yield _bundle_root(path)
        return
    with tempfile.TemporaryDirectory() as scratch:
        _safe_extract(path, Path(scratch))
        yield _bundle_root(Path(scratch))


def _row_id(row):
    return row["id"]


def _run_skill_key(row):
    return (row["run_id"], row["skill_id"], row["role"], row.get("round"))


MERGE_KEYS = {
    EXPERIMENTS: _row_id,
    CANDIDATE_SKILLS: _row_id,
    DESIGN_REVIEWS: _row_id,
    SKILL_DECISIONS: _row_id,
    RUN_SKILLS: _run_skill_key,
}


def merge_parts(paths, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    runs = {}
    merged = {name: {} for name in MERGE_KEYS}
    freeze = None
    manifest = None
    for path in paths:
        with _opened(path) as directory:
            for row in _read_lines(directory / RUNS):
                runs[row["id"]] = {**runs.get(row["id"], {}), **row}
            for name, key in MERGE_KEYS.items():
                for row in _read_lines(directory / name):
                    merged[name].setdefault(key(row), row)
            if (directory / FREEZE_INFO).exists():
                info = json.loads((directory / FREEZE_INFO).read_text(encoding="utf-8"))
                if freeze is None or info["created_at"] > freeze["created_at"]:
                    freeze = info
            if manifest is None:
                manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if manifest is None:
        raise ValueError("no parts were given")
    (target / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    if freeze is not None:
        (target / FREEZE_INFO).write_text(json.dumps(freeze), encoding="utf-8")
    for name, rows in {RUNS: runs, **merged}.items():
        (target / name).write_text("".join(json.dumps(row) + "\n" for row in rows.values()), encoding="utf-8")
    return target


def _shipped_registry(directory):
    shipped = {"models": [], "hypothesis_sets": {}, "datasets": [], "prompt_versions": [], "library_snapshots": []}
    for name in (REGISTRY, FROZEN_SNAPSHOT):
        path = directory / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("models", "datasets", "prompt_versions"):
            shipped[key].extend(data.get(key, []))
        for set_name, items in data.get("hypothesis_sets", {}).items():
            shipped["hypothesis_sets"].setdefault(set_name, []).extend(items)
        if data.get("library_snapshot"):
            shipped["library_snapshots"].append(data["library_snapshot"])
    return shipped


def _present(conn, table, value):
    return conn.execute(f"select 1 from {table} where id = %s", (value,)).fetchone() is not None


def _model_id(conn, model_id, model_name, by_id):
    if _present(conn, "registry.models", model_id):
        return model_id, 0
    entry = by_id.get(model_id)
    found = conn.execute(
        "select id::text as id from registry.models where name = %s", (entry["name"] if entry else model_name,)
    ).fetchone()
    if found is not None:
        return found["id"], 0
    if entry is None:
        raise ValueError(f"the archive does not carry the model {model_name} ({model_id})")
    placeholders = ", ".join("%s::jsonb" if f in JSON_FIELDS else "%s" for f in MODEL_FIELDS)
    conn.execute(
        f"insert into registry.models (id, {', '.join(MODEL_FIELDS)}) values (%s, {placeholders}) "
        "on conflict do nothing",
        [model_id, *model_values(entry)],
    )
    return model_id, 1


def _hypothesis_id(conn, hypothesis_id, by_id):
    if _present(conn, "registry.hypotheses", hypothesis_id):
        return hypothesis_id, 0
    if hypothesis_id not in by_id:
        raise ValueError(f"the archive does not carry the hypothesis {hypothesis_id}")
    set_name, entry = by_id[hypothesis_id]
    set_id = conn.execute(
        "insert into registry.hypothesis_sets (id, name) values (%s, %s) "
        "on conflict (name) do update set name = excluded.name returning id::text as id",
        (str(uuid.uuid4()), set_name),
    ).fetchone()["id"]
    found = conn.execute(
        "select id::text as id from registry.hypotheses where set_id = %s and text = %s", (set_id, entry["text"])
    ).fetchone()
    if found is not None:
        return found["id"], 0
    conn.execute(
        "insert into registry.hypotheses (id, set_id, text, split, ground_truth) values (%s, %s, %s, %s, %s)",
        (hypothesis_id, set_id, entry["text"], entry.get("split", "none"), entry.get("ground_truth")),
    )
    return hypothesis_id, 1


def _prompt_version_id(conn, version_id, by_id):
    if _present(conn, "registry.prompt_versions", version_id):
        return version_id, 0
    entry = by_id.get(version_id)
    if entry is None:
        raise ValueError(f"the archive does not carry the prompt version {version_id}")
    found = conn.execute(
        "select id::text as id from registry.prompt_versions where hash = %s", (entry["hash"],)
    ).fetchone()
    if found is not None:
        return found["id"], 0
    conn.execute(
        "insert into registry.prompt_versions (id, role, text, hash) values (%s, %s, %s, %s)",
        (version_id, entry["role"], entry["text"], entry["hash"]),
    )
    return version_id, 1


def _insert_dataset(conn, entry):
    if _present(conn, "registry.datasets", entry["id"]):
        return 0
    return conn.execute(
        "insert into registry.datasets (id, name, path, loader_type, schema, permutable_columns) "
        "values (%s, %s, %s, %s, %s::jsonb, %s::jsonb) on conflict do nothing",
        (
            entry["id"],
            entry["name"],
            entry["path"],
            entry["loader_type"],
            json.dumps(entry["schema"]),
            json.dumps(entry["permutable_columns"]),
        ),
    ).rowcount


def _insert_library_snapshot(conn, info):
    return conn.execute(
        "insert into registry.library_snapshots (id, created_at, embedding_model, skill_ids, content_hash, path) "
        "values (%s, %s, %s, %s::jsonb, %s, %s) on conflict do nothing",
        (
            info["id"],
            info["created_at"],
            info["embedding_model"],
            json.dumps(info["skill_ids"]),
            info["content_hash"],
            None,
        ),
    ).rowcount


def _legacy_snapshot(directory):
    manifest_path, freeze_path = directory / MANIFEST, directory / FREEZE_INFO
    if not (manifest_path.exists() and freeze_path.exists()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    return {
        "id": manifest["snapshot_id"],
        "created_at": manifest["created_at"],
        "embedding_model": freeze["embedding_model"],
        "skill_ids": [],
        "content_hash": UNSHIPPED_HASH,
    }


def _ensure_snapshots(conn, snapshot_ids, directory, shipped):
    by_id = {info["id"]: info for info in shipped["library_snapshots"]}
    legacy = _legacy_snapshot(directory)
    if legacy is not None:
        by_id.setdefault(legacy["id"], legacy)
    inserted = 0
    for snapshot_id in snapshot_ids:
        if _present(conn, "registry.library_snapshots", snapshot_id):
            continue
        if snapshot_id not in by_id:
            raise ValueError(f"the archive does not carry the library snapshot {snapshot_id}")
        inserted += _insert_library_snapshot(conn, by_id[snapshot_id])
    return inserted


def _remap(row, column, mapping):
    value = row.get(column)
    return row if value is None else {**row, column: mapping.get(value, value)}


def _ensure_registry(conn, directory, runs, decisions):
    shipped = _shipped_registry(directory)
    counts = dict.fromkeys(("models", "hypotheses", "datasets", "prompt_versions", "library_snapshots"), 0)
    snapshot_ids = sorted({row["library_snapshot_id"] for row in runs.values() if row.get("library_snapshot_id")})
    counts["library_snapshots"] = _ensure_snapshots(conn, snapshot_ids, directory, shipped)
    models_by_id = {entry["id"]: entry for entry in shipped["models"]}
    hypotheses_by_id = {
        entry["id"]: (set_name, entry) for set_name, items in shipped["hypothesis_sets"].items() for entry in items
    }
    versions_by_id = {entry["id"]: entry for entry in shipped["prompt_versions"]}
    model_map, hypothesis_map, version_map = {}, {}, {}
    for row in runs.values():
        if row.get("model_id") and row["model_id"] not in model_map:
            model_map[row["model_id"]], added = _model_id(conn, row["model_id"], row["model_name"], models_by_id)
            counts["models"] += added
        if row.get("hypothesis_id") and row["hypothesis_id"] not in hypothesis_map:
            hypothesis_map[row["hypothesis_id"]], added = _hypothesis_id(conn, row["hypothesis_id"], hypotheses_by_id)
            counts["hypotheses"] += added
    for row in decisions:
        if row.get("prompt_version_id") and row["prompt_version_id"] not in version_map:
            version_map[row["prompt_version_id"]], added = _prompt_version_id(
                conn, row["prompt_version_id"], versions_by_id
            )
            counts["prompt_versions"] += added
    used = {name for row in runs.values() for name in row.get("datasets") or []}
    counts["datasets"] = sum(_insert_dataset(conn, entry) for entry in shipped["datasets"] if entry["name"] in used)
    remapped_runs = {
        run_id: _remap(_remap(row, "model_id", model_map), "hypothesis_id", hypothesis_map)
        for run_id, row in runs.items()
    }
    remapped_decisions = [_remap(row, "prompt_version_id", version_map) for row in decisions]
    return remapped_runs, remapped_decisions, {key: value for key, value in counts.items() if value}


def _relocate_trajectory(row, directory, trajectory_dir):
    source = directory / "trajectories" / f"{row['id']}.json"
    if trajectory_dir is None or not row.get("trajectory_path") or not source.is_file():
        return row
    target = Path(trajectory_dir) / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {**row, "trajectory_path": str(target)}


def _import_directory(conn, directory, trajectory_dir=None):
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    runs = {}
    for row in _read_lines(directory / RUNS):
        runs[row["id"]] = {**runs.get(row["id"], {}), **row}
    runs = {run_id: _relocate_trajectory(row, directory, trajectory_dir) for run_id, row in runs.items()}
    counts = {}
    with conn.transaction():
        counts["experiments"] = sum(
            insert_row(conn, "runs.experiments", row) for row in _read_lines(directory / EXPERIMENTS)
        )
        counts["candidate_skills"] = sum(
            _insert_candidate(conn, _with_known_source_run(conn, entry, runs))
            for entry in _read_lines(directory / CANDIDATE_SKILLS)
        )
        frozen = _insert_frozen_snapshot(conn, directory)
        if frozen is not None:
            counts["frozen_snapshots"] = frozen
        decisions = _read_lines(directory / SKILL_DECISIONS)
        runs, decisions, registry_counts = _ensure_registry(conn, directory, runs, decisions)
        if registry_counts:
            counts["registry"] = registry_counts
        counts["runs"] = sum(insert_row(conn, "runs.runs", row) for row in runs.values())
        for name, table, key in (
            (RUN_SKILLS, "runs.run_skills", "run_skills"),
            (DESIGN_REVIEWS, "runs.design_reviews", "design_reviews"),
        ):
            counts[key] = sum(insert_row(conn, table, row) for row in _read_lines(directory / name))
        counts["skill_decisions"] = sum(_insert_decision(conn, row) for row in decisions)
        log_event(conn, None, "bundle_import", "bundle", manifest["snapshot_id"], after=counts)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.db.bundle")
    parser.add_argument("command", choices=["import"])
    parser.add_argument("path")
    args = parser.parse_args(argv)
    from popper_upgrade.db.connection import connect
    from popper_upgrade.run import default_trajectory_dir

    with connect() as conn:
        print(json.dumps(import_bundle(conn, args.path, default_trajectory_dir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
