import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from popper_upgrade.db.skills_repo import SKILL_COLUMNS, row_to_skill
from popper_upgrade.skills.embedding import DEFAULT_EMBEDDING_MODEL
from popper_upgrade.skills.schema import DecisionExample, Skill


def export_snapshot(conn, path, include_rejected=True):
    verdicts = "approved is not null" if include_rejected else "approved is true"
    rows = conn.execute(
        f"select {SKILL_COLUMNS}, embedding_model, embedding from skills.skills where {verdicts} order by id"
    ).fetchall()
    models = {row["embedding_model"] for row in rows}
    if len(models) > 1:
        raise ValueError(f"library mixes embedding models: {sorted(models)}")
    embedding_model = models.pop() if models else DEFAULT_EMBEDDING_MODEL
    skills = [{**asdict(row_to_skill(row)), "embedding": [float(x) for x in row["embedding"]]} for row in rows]
    skill_ids = [skill["id"] for skill in skills]
    decisions = conn.execute(
        "select distinct on (skill_id) skill_id::text as skill_id, verdict, reason, decider "
        "from skills.skill_decisions where skill_id::text = any(%s::text[]) order by skill_id, created_at desc, id",
        (skill_ids,),
    ).fetchall()
    prompt_versions = conn.execute(
        "select id::text as id, role, text, hash from registry.prompt_versions order by hash"
    ).fetchall()
    models_table = conn.execute(
        "select id::text as id, name, backend, served_model_id, size_class, environment, port, default_temperature, "
        "base_url, api_key_env, strip_tools, min_disk_gb from registry.models order by name"
    ).fetchall()
    hypotheses = conn.execute(
        "select s.name as set_name, h.id::text as id, h.text, h.split, h.ground_truth "
        "from registry.hypotheses h join registry.hypothesis_sets s on s.id = h.set_id order by s.name, h.text"
    ).fetchall()
    hypothesis_sets = {}
    for row in hypotheses:
        hypothesis_sets.setdefault(row["set_name"], []).append({k: v for k, v in row.items() if k != "set_name"})
    datasets = conn.execute(
        "select id::text as id, name, path, loader_type, schema, permutable_columns "
        "from registry.datasets order by name"
    ).fetchall()

    snapshot_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    fingerprints = sorted(f"{s['id']}:{Skill(**_skill_fields(s)).content_hash}:{s['approved']}" for s in skills)
    digest = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    payload = {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "embedding_model": embedding_model,
        "skills": skills,
        "decisions": decisions,
        "prompt_versions": prompt_versions,
        "models": models_table,
        "hypothesis_sets": hypothesis_sets,
        "datasets": datasets,
    }
    path = Path(path)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    conn.execute(
        "insert into registry.library_snapshots (id, created_at, embedding_model, skill_ids, content_hash, path) "
        "values (%s, %s, %s, %s::jsonb, %s, %s)",
        (snapshot_id, created_at, embedding_model, json.dumps(skill_ids), digest, str(path)),
    )
    return snapshot_id


def _skill_fields(entry):
    return {k: v for k, v in entry.items() if k != "embedding"}


class SnapshotSkillSource:
    def __init__(self, path, embedder):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["embedding_model"] != embedder.model_name:
            raise ValueError(f"snapshot embedding model {payload['embedding_model']} != embedder {embedder.model_name}")
        self.snapshot_id = payload["snapshot_id"]
        self.created_at = payload["created_at"]
        self.embedding_model = payload["embedding_model"]
        self.embedder = embedder
        self._skills = [Skill(**_skill_fields(entry)) for entry in payload["skills"]]
        matrix = np.array([entry["embedding"] for entry in payload["skills"]], dtype=np.float32)
        if self._skills:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1, norms)
        self._matrix = matrix
        self._decisions = payload["decisions"]
        self.registry = {
            "models": payload["models"],
            "hypothesis_sets": payload["hypothesis_sets"],
            "datasets": payload["datasets"],
            "prompt_versions": payload["prompt_versions"],
        }

    def _similarities(self, text):
        if not self._skills:
            return np.zeros(0, dtype=np.float32)
        vector = np.array(self.embedder.encode(text), dtype=np.float32)
        vector = vector / np.linalg.norm(vector)
        return self._matrix @ vector

    def max_similarity(self, text):
        sims = self._similarities(text)
        return float(sims.max()) if sims.size else None

    def retrieve_scored(self, query, k, allowed_ids=None, min_similarity=0.0):
        if allowed_ids is not None and not allowed_ids:
            return []
        sims = self._similarities(query)
        candidates = []
        for index, skill in enumerate(self._skills):
            if allowed_ids is None:
                if skill.approved is not True:
                    continue
            elif skill.id not in allowed_ids:
                continue
            if sims[index] >= min_similarity:
                candidates.append((skill, float(sims[index])))
        candidates.sort(key=lambda pair: (-pair[1], pair[0].id))
        return candidates[:k]

    def list_skills(self, approved=True):
        return [skill for skill in self._skills if skill.approved is approved]

    def similar_decisions(self, text, k=5):
        sims = self._similarities(text)
        by_id = {skill.id: index for index, skill in enumerate(self._skills)}
        ranked = sorted(
            (d for d in self._decisions if d["skill_id"] in by_id),
            key=lambda d: (-sims[by_id[d["skill_id"]]], d["skill_id"]),
        )
        return [
            DecisionExample(self._skills[by_id[d["skill_id"]]], d["verdict"], d["reason"], d["decider"])
            for d in ranked[:k]
        ]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.db.snapshot")
    parser.add_argument("command", choices=["export"])
    parser.add_argument("path")
    args = parser.parse_args(argv)
    from popper_upgrade.db.connection import connect

    with connect() as conn:
        print(export_snapshot(conn, args.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
