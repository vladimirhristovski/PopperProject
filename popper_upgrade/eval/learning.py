import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from popper_upgrade.db.snapshot import SnapshotSkillSource, skills_content_hash
from popper_upgrade.skills.schema import Skill

FREEZE_NAMESPACE = uuid.UUID("0c7f0b1e-41a6-4f6e-8a0f-5b7c6c9d2e11")
CANDIDATES = "candidate_skills.jsonl"
DECISIONS = "skill_decisions.jsonl"
FROZEN_SNAPSHOT = "frozen_snapshot.json"
FREEZE_INFO = "freeze.json"


@dataclass
class Learning:
    source: object
    sink: object
    freeze: object


def _read_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _skill_from(candidate, decision):
    fields = {key: candidate[key] for key in Skill.__dataclass_fields__ if key in candidate}
    fields.update(approved=True, decided_by=decision["decider"], decision_reason=decision["reason"])
    return Skill(**fields)


def decided_candidates(bundle_dir):
    candidates = {row["id"]: row for row in _read_lines(Path(bundle_dir) / CANDIDATES)}
    latest = {}
    for row in _read_lines(Path(bundle_dir) / DECISIONS):
        if row.get("trigger") == "entry" and row.get("skill_id") in candidates:
            latest[row["skill_id"]] = row
    return candidates, latest


def approved_candidates(bundle_dir):
    candidates, latest = decided_candidates(bundle_dir)
    return [(candidates[skill_id], row) for skill_id, row in latest.items() if row["verdict"] == "approve"]


def _decision_entry(row):
    return {"skill_id": row["skill_id"], "verdict": row["verdict"], "reason": row["reason"], "decider": row["decider"]}


class LearningSink:
    def __init__(self, sink, library):
        self._sink = sink
        self._library = library
        self._pending = {}

    def __getattr__(self, name):
        return getattr(self._sink, name)

    def add_candidate_skill(self, skill):
        skill_id = self._sink.add_candidate_skill(skill)
        if skill_id is not None:
            self._pending[skill_id] = skill
        return skill_id

    def add_skill_decision(self, decision):
        result = self._sink.add_skill_decision(decision)
        skill = self._pending.pop(decision.get("skill_id"), None)
        if skill is not None and decision.get("trigger") == "entry" and decision.get("verdict") == "approve":
            approved = Skill(
                **{
                    **asdict(skill),
                    "id": decision["skill_id"],
                    "approved": True,
                    "decided_by": decision.get("decider"),
                    "decision_reason": decision.get("reason"),
                }
            )
            self._library.add_skill(approved, self._library.embedder.encode(approved.applicability), decision)
        return result


def open_library(snapshot_path, bundle_dir, embedder):
    library = SnapshotSkillSource(snapshot_path, embedder)
    for candidate, decision in approved_candidates(bundle_dir):
        library.add_skill(_skill_from(candidate, decision), candidate["embedding"], _decision_entry(decision))
    return library


def freeze_bundle(snapshot_path, bundle_dir, embedder):
    bundle_dir = Path(bundle_dir)
    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    candidates, _ = decided_candidates(bundle_dir)
    approved = approved_candidates(bundle_dir)
    for candidate, decision in approved:
        skill = _skill_from(candidate, decision)
        payload["skills"].append({**asdict(skill), "embedding": candidate["embedding"]})
        payload["decisions"].append(_decision_entry(decision))
    base_id = payload["snapshot_id"]
    new_ids = sorted(candidate["id"] for candidate, _ in approved)
    payload["snapshot_id"] = str(uuid.uuid5(FREEZE_NAMESPACE, ":".join([base_id, *new_ids])))
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    path = bundle_dir / FROZEN_SNAPSHOT
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    approved_total = sum(1 for entry in payload["skills"] if entry["approved"] is True)
    info = {
        "snapshot_id": payload["snapshot_id"],
        "base_snapshot_id": base_id,
        "created_at": payload["created_at"],
        "embedding_model": payload["embedding_model"],
        "content_hash": skills_content_hash(payload["skills"]),
        "skill_ids": sorted(entry["id"] for entry in payload["skills"]),
        "candidates": len(candidates),
        "learned_skills": len(approved),
        "approved_skills": approved_total,
        "empty_library": approved_total == 0,
    }
    (bundle_dir / FREEZE_INFO).write_text(json.dumps(info, indent=2), encoding="utf-8")
    return SnapshotSkillSource(path, embedder)


def bundle_learning(snapshot_path, bundle_dir, embedder, library, sink):
    return Learning(
        source=library,
        sink=LearningSink(sink, library),
        freeze=lambda: freeze_bundle(snapshot_path, bundle_dir, embedder),
    )


def db_learning(conn, embedder, sink, snapshot_dir):
    from popper_upgrade.db.skills_repo import SkillRepository

    def freeze():
        from popper_upgrade.db.snapshot import export_snapshot

        path = Path(snapshot_dir) / f"{uuid.uuid4()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        export_snapshot(conn, path)
        return SnapshotSkillSource(path, embedder)

    return Learning(source=SkillRepository(conn, embedder), sink=sink, freeze=freeze)
