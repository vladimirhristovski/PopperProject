import argparse
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
from popper_upgrade.db.rows import apply_entry_decision, insert_row, run_key

EXPERIMENTS = "experiments.jsonl"
RUNS = "runs.jsonl"
RUN_SKILLS = "run_skills.jsonl"
DESIGN_REVIEWS = "design_reviews.jsonl"
SKILL_DECISIONS = "skill_decisions.jsonl"
CANDIDATE_SKILLS = "candidate_skills.jsonl"
MANIFEST = "manifest.json"


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
        "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing",
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
            _insert_candidate(conn, entry) for entry in _read_lines(directory / CANDIDATE_SKILLS)
        )
        counts["runs"] = sum(insert_row(conn, "runs.runs", row) for row in runs.values())
        for name, table, key in (
            (RUN_SKILLS, "runs.run_skills", "run_skills"),
            (DESIGN_REVIEWS, "runs.design_reviews", "design_reviews"),
        ):
            counts[key] = sum(insert_row(conn, table, row) for row in _read_lines(directory / name))
        counts["skill_decisions"] = sum(
            _insert_decision(conn, row) for row in _read_lines(directory / SKILL_DECISIONS)
        )
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
