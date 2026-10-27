import json
import uuid

from popper_upgrade.db.rows import RUN_COLUMNS, TABLES, apply_entry_decision, check_columns, insert_row, run_key


def _clean(row):
    return {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in row.items()}


class RunRepository:
    def __init__(self, conn, skills):
        self._conn = conn
        self._skills = skills

    def create_experiment(self, experiment):
        insert_row(self._conn, "runs.experiments", experiment)
        return experiment["id"]

    def completed_runs(self, experiment_id):
        rows = self._conn.execute(
            "select model_name, arm, task, hypothesis, replicate, permuted from runs.runs "
            "where experiment_id = %s and status = 'completed'",
            (experiment_id,),
        ).fetchall()
        return {run_key(row) for row in rows}

    def model_costs(self, model_name):
        rows = self._conn.execute(
            "select status, cost_usd, arm from runs.runs where model_name = %s", (model_name,)
        ).fetchall()
        return [(row["status"], row["cost_usd"], row["arm"]) for row in rows]

    def create_run(self, config):
        row = {"status": "running", **config}
        row.setdefault("id", str(uuid.uuid4()))
        insert_row(self._conn, "runs.runs", row)
        return row["id"]

    def finish_run(self, run_id, result):
        fields = {"status": "completed", **result}
        check_columns(fields, RUN_COLUMNS - {"id"})
        json_columns = TABLES["runs.runs"][1]
        assignments = [f"{c} = %s::jsonb" if c in json_columns else f"{c} = %s" for c in fields]
        params = [json.dumps(v) if c in json_columns and v is not None else v for c, v in fields.items()]
        self._conn.execute(
            f"update runs.runs set {', '.join(assignments)}, finished_at = now() where id = %s", [*params, run_id]
        )

    def add_run_skills(self, run_id, rows):
        for row in rows:
            insert_row(self._conn, "runs.run_skills", {"run_id": run_id, **row})

    def add_design_reviews(self, run_id, reviews):
        for review in reviews:
            insert_row(self._conn, "runs.design_reviews", {"id": str(uuid.uuid4()), "run_id": run_id, **review})

    def add_skill_decision(self, decision):
        row = {"id": str(uuid.uuid4()), **decision}
        if insert_row(self._conn, "skills.skill_decisions", row) and row.get("trigger") == "entry":
            apply_entry_decision(self._conn, row)
        return row["id"]

    def add_candidate_skill(self, skill):
        return self._skills.add_candidate_skill(skill)

    def get_run(self, run_id):
        row = self._conn.execute("select * from runs.runs where id = %s", (run_id,)).fetchone()
        return _clean(row) if row else None

    def list_runs(self, filters=None):
        filters = filters or {}
        check_columns(filters, RUN_COLUMNS)
        where = f" where {' and '.join(f'{c} = %s' for c in filters)}" if filters else ""
        rows = self._conn.execute(
            f"select * from runs.runs{where} order by created_at desc, id", list(filters.values())
        ).fetchall()
        return [_clean(row) for row in rows]

    def run_skills(self, run_id):
        rows = self._conn.execute(
            "select run_id, skill_id, role, round, similarity from runs.run_skills where run_id = %s "
            "order by round_key, role, skill_id",
            (run_id,),
        ).fetchall()
        return [_clean(row) for row in rows]

    def design_reviews(self, run_id):
        rows = self._conn.execute(
            "select * from runs.design_reviews where run_id = %s order by round, attempt", (run_id,)
        ).fetchall()
        return [_clean(row) for row in rows]
