import uuid

from popper_upgrade.app import auth
from popper_upgrade.app.auth import require_admin
from popper_upgrade.db.audit import log_event
from popper_upgrade.db.bundle import import_bundle


def list_users(ctx, user):
    require_admin(user)
    with ctx.conn() as conn:
        return auth.list_users(conn)


def create_user_h(ctx, user, username, password, role):
    require_admin(user)
    with ctx.conn() as conn:
        return auth.create_user(conn, username, password, role, user)


def set_role_h(ctx, user, user_id, role):
    require_admin(user)
    with ctx.conn() as conn:
        auth.set_role(conn, user_id, role, user)


def set_active_h(ctx, user, user_id, active):
    require_admin(user)
    with ctx.conn() as conn:
        auth.set_active(conn, user_id, active, user)


def upsert_model(
    ctx,
    user,
    name,
    backend,
    served_model_id,
    size_class,
    environment,
    port=None,
    default_temperature=None,
    base_url=None,
    api_key_env=None,
    strip_tools=True,
    min_disk_gb=None,
):
    require_admin(user)
    with ctx.conn() as conn:
        before = conn.execute("select to_jsonb(m) as row from registry.models m where name = %s", (name,)).fetchone()
        row = conn.execute(
            "insert into registry.models as m (id, name, backend, served_model_id, size_class, environment, port, "
            "default_temperature, base_url, api_key_env, strip_tools, min_disk_gb) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "on conflict (name) do update set backend = excluded.backend, "
            "served_model_id = excluded.served_model_id, size_class = excluded.size_class, "
            "environment = excluded.environment, port = excluded.port, "
            "default_temperature = excluded.default_temperature, base_url = excluded.base_url, "
            "api_key_env = excluded.api_key_env, strip_tools = excluded.strip_tools, "
            "min_disk_gb = excluded.min_disk_gb returning to_jsonb(m) as row",
            (
                str(uuid.uuid4()),
                name,
                backend,
                served_model_id,
                size_class,
                environment,
                port,
                default_temperature,
                base_url or None,
                api_key_env or None,
                strip_tools,
                min_disk_gb,
            ),
        ).fetchone()
        log_event(
            conn,
            user.id,
            "model.upsert",
            "model",
            name,
            before=before["row"] if before else None,
            after=row["row"],
        )
        return name


def list_models(ctx, user):
    with ctx.conn() as conn:
        return ctx.registry(conn).models()


def list_prompt_versions(ctx, user):
    require_admin(user)
    with ctx.conn() as conn:
        return conn.execute(
            "select id::text as id, role, hash, text, created_at from registry.prompt_versions order by created_at"
        ).fetchall()


def add_prompt_version(ctx, user, role, text):
    require_admin(user)
    if not text.strip():
        raise ValueError("prompt text is required")
    with ctx.conn() as conn:
        version_id = ctx.registry(conn).prompt_version(role, text)
        log_event(conn, user.id, "prompt_version.add", "prompt_version", version_id, after={"role": role})
        return version_id


def import_bundle_h(ctx, user, path):
    require_admin(user)
    with ctx.conn() as conn:
        counts = import_bundle(conn, path, ctx.trajectory_dir)
        log_event(conn, user.id, "bundle.import", "bundle", str(path), after=counts)
        return counts
