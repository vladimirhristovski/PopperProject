import uuid

import numpy as np

from popper_upgrade.skills.schema import DecisionExample, Skill

SKILL_COLUMNS = (
    "id::text as id, applicability, design_template, preconditions, domain, source_hypothesis, "
    "source_run_id::text as source_run_id, parent_skill_id::text as parent_skill_id, approved, decided_by, "
    "decision_reason, times_used, times_passed, created_at"
)
EDITABLE_FIELDS = {"applicability", "design_template", "preconditions", "domain", "source_hypothesis"}


def row_to_skill(row):
    fields = {name: row[name] for name in Skill.__dataclass_fields__}
    fields["created_at"] = fields["created_at"].isoformat()
    return Skill(**fields)


class SkillRepository:
    def __init__(self, conn, embedder, dedup_threshold=0.95):
        self._conn = conn
        self._embedder = embedder
        self._dedup_threshold = dedup_threshold

    @property
    def conn(self):
        return self._conn

    @property
    def embedding_model(self):
        return self._embedder.model_name

    def _vector(self, text):
        return np.array(self._embedder.encode(text), dtype=np.float32)

    def retrieve_scored(self, query, k, allowed_ids=None, min_similarity=0.0):
        if allowed_ids is not None and not allowed_ids:
            return []
        scope = "approved is true" if allowed_ids is None else "id = any(%(ids)s::uuid[])"
        params = {"q": self._vector(query), "min": min_similarity, "k": k}
        if allowed_ids is not None:
            params["ids"] = sorted(allowed_ids)
        rows = self._conn.execute(
            f"select {SKILL_COLUMNS}, 1 - (embedding <=> %(q)s) as similarity from skills.skills "
            f"where {scope} and 1 - (embedding <=> %(q)s) >= %(min)s "
            "order by embedding <=> %(q)s, id limit %(k)s",
            params,
        ).fetchall()
        return [(row_to_skill(row), float(row["similarity"])) for row in rows]

    def list_skills(self, approved=True):
        return self.search(approved=approved)

    def similar_decisions(self, text, k=5):
        rows = self._conn.execute(
            "with latest as ("
            "select distinct on (skill_id) skill_id, verdict, reason, decider from skills.skill_decisions "
            "order by skill_id, created_at desc, id) "
            f"select {SKILL_COLUMNS}, l.verdict, l.reason as decision_text, l.decider as decision_by "
            "from latest l join skills.skills s on s.id = l.skill_id "
            "order by s.embedding <=> %(q)s, s.id limit %(k)s",
            {"q": self._vector(text), "k": k},
        ).fetchall()
        return [
            DecisionExample(row_to_skill(row), row["verdict"], row["decision_text"], row["decision_by"])
            for row in rows
        ]

    def add_candidate_skill(self, skill):
        vector = self._vector(skill.applicability)
        nearest = self._conn.execute(
            "select 1 - (embedding <=> %s) as similarity from skills.skills order by embedding <=> %s limit 1",
            (vector, vector),
        ).fetchone()
        if nearest is not None and nearest["similarity"] >= self._dedup_threshold:
            return None
        skill_id = skill.id or str(uuid.uuid4())
        self._conn.execute(
            "insert into skills.skills (id, applicability, design_template, preconditions, domain, "
            "source_hypothesis, source_run_id, parent_skill_id, approved, decided_by, decision_reason, "
            "times_used, times_passed, content_hash, embedding_model, embedding, created_at) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "coalesce(nullif(%s, '')::timestamptz, now()))",
            (
                skill_id,
                skill.applicability,
                skill.design_template,
                skill.preconditions,
                skill.domain,
                skill.source_hypothesis,
                skill.source_run_id,
                skill.parent_skill_id,
                skill.approved,
                skill.decided_by,
                skill.decision_reason,
                skill.times_used,
                skill.times_passed,
                skill.content_hash,
                self.embedding_model,
                vector,
                skill.created_at,
            ),
        )
        return skill_id

    def get(self, skill_id):
        row = self._conn.execute(f"select {SKILL_COLUMNS} from skills.skills where id = %s", (skill_id,)).fetchone()
        return row_to_skill(row) if row else None

    def search(self, text=None, approved="any", domain=None, flagged=None):
        clauses = []
        params = []
        if text:
            clauses.append("(applicability ilike %s or design_template ilike %s)")
            params += [f"%{text}%", f"%{text}%"]
        if approved is None:
            clauses.append("approved is null")
        elif approved != "any":
            clauses.append("approved = %s")
            params.append(approved)
        if domain is not None:
            clauses.append("domain = %s")
            params.append(domain)
        if flagged is not None:
            clauses.append("flagged = %s")
            params.append(flagged)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"select {SKILL_COLUMNS} from skills.skills{where} order by created_at, id", params
        ).fetchall()
        return [row_to_skill(row) for row in rows]

    def set_approval(
        self,
        skill_id,
        approved,
        decider,
        reason,
        trigger,
        user_id=None,
        run_id=None,
        model=None,
        prompt_version_id=None,
    ):
        decision_id = str(uuid.uuid4())
        self._conn.execute(
            "insert into skills.skill_decisions (id, skill_id, run_id, trigger, decider, user_id, verdict, reason, "
            "model, prompt_version_id) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                decision_id,
                skill_id,
                run_id,
                trigger,
                decider,
                user_id,
                "approve" if approved else "reject",
                reason,
                model,
                prompt_version_id,
            ),
        )
        self._conn.execute(
            "update skills.skills set approved = %s, decided_by = %s, decision_reason = %s, updated_at = now() "
            "where id = %s",
            (approved, decider, reason, skill_id),
        )
        return decision_id

    def update_content(self, skill_id, **fields):
        unknown = set(fields) - EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"cannot update fields: {sorted(unknown)}")
        if not fields:
            return
        current = self.get(skill_id)
        if current is None:
            raise KeyError(skill_id)
        for name, value in fields.items():
            setattr(current, name, value)
        assignments = [f"{name} = %s" for name in fields]
        params = list(fields.values())
        if "applicability" in fields:
            assignments += ["embedding = %s", "embedding_model = %s"]
            params += [self._vector(current.applicability), self.embedding_model]
        assignments += ["content_hash = %s", "updated_at = now()"]
        params += [current.content_hash, skill_id]
        self._conn.execute(f"update skills.skills set {', '.join(assignments)} where id = %s", params)

    def flag(self, skill_id, user_id, reason):
        flag_id = str(uuid.uuid4())
        self._conn.execute(
            "insert into skills.skill_flags (id, skill_id, user_id, reason) values (%s, %s, %s, %s)",
            (flag_id, skill_id, user_id, reason),
        )
        self._conn.execute("update skills.skills set flagged = true, updated_at = now() where id = %s", (skill_id,))
        return flag_id

    def record_usage(self, skill_id, passed):
        self._conn.execute(
            "update skills.skills set times_used = times_used + 1, times_passed = times_passed + %s where id = %s",
            (1 if passed else 0, skill_id),
        )
