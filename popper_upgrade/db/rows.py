import json

RUN_COLUMNS = set(
    """id experiment_id created_at finished_at status environment launched_by hypothesis_id hypothesis arm
    skills_enabled reviewer_enabled human_review model_id model_name temperature seed alpha aggregate_test
    max_num_of_tests max_retry time_limit domain datasets permuted permute_seed library_snapshot_id popper_version
    git_sha pins_hash res res_stat n_tests parsed_result reviewer_rejections reviewer_overrides tokens_in tokens_out
    wall_time_s trajectory_path error answer task replicate control_degenerate""".split()
)
EXPERIMENT_COLUMNS = set("id name spec status created_at".split())
RUN_SKILL_COLUMNS = set("run_id skill_id role round similarity".split())
DESIGN_REVIEW_COLUMNS = set("id run_id round attempt proposal approved reason overridden created_at".split())
DECISION_COLUMNS = set(
    "id skill_id run_id trigger decider user_id verdict reason model prompt_version_id created_at".split()
)

TABLES = {
    "runs.experiments": (EXPERIMENT_COLUMNS, {"spec"}),
    "runs.runs": (RUN_COLUMNS, {"datasets", "parsed_result"}),
    "runs.run_skills": (RUN_SKILL_COLUMNS, set()),
    "runs.design_reviews": (DESIGN_REVIEW_COLUMNS, set()),
    "skills.skill_decisions": (DECISION_COLUMNS, set()),
}


def run_key(row):
    return (
        row["model_name"],
        row["arm"],
        row.get("task"),
        row["hypothesis"],
        row.get("replicate"),
        bool(row.get("permuted")),
    )


def check_columns(row, allowed):
    unknown = set(row) - allowed
    if unknown:
        raise ValueError(f"unknown columns: {sorted(unknown)}")


def insert_row(conn, table, row):
    allowed, json_columns = TABLES[table]
    check_columns(row, allowed)
    columns = list(row)
    placeholders = ", ".join("%s::jsonb" if c in json_columns else "%s" for c in columns)
    values = [json.dumps(row[c]) if c in json_columns and row[c] is not None else row[c] for c in columns]
    cur = conn.execute(
        f"insert into {table} ({', '.join(columns)}) values ({placeholders}) on conflict do nothing", values
    )
    return cur.rowcount


def apply_entry_decision(conn, row):
    conn.execute(
        "update skills.skills set approved = %s, decided_by = %s, decision_reason = %s, updated_at = now() "
        "where id = %s",
        (row["verdict"] == "approve", row["decider"], row["reason"], row["skill_id"]),
    )
