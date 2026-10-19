import json


def log_event(conn, user_id, action, target_type, target_id, before=None, after=None):
    conn.execute(
        "insert into audit.events (user_id, action, target_type, target_id, before, after) "
        "values (%s, %s, %s, %s, %s::jsonb, %s::jsonb)",
        (
            user_id,
            action,
            target_type,
            target_id,
            None if before is None else json.dumps(before),
            None if after is None else json.dumps(after),
        ),
    )
