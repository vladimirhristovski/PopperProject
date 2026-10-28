
from popper_upgrade.app import auth
from popper_upgrade.app.auth import require_admin
from popper_upgrade.db.audit import log_event
from popper_upgrade.db.bundle import import_bundle
from popper_upgrade.db.registry_repo import upsert_model_row
from popper_upgrade.serving.pricing import check_pricing


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
    quantization=None,
    dtype=None,
    max_model_len=None,
    system_prefix=None,
    request_overrides=None,
    serving=None,
    pricing=None,
    budget_eur=None,
    revision=None,
):
    require_admin(user)
    with ctx.conn() as conn:
        before = conn.execute("select to_jsonb(m) as row from registry.models m where name = %s", (name,)).fetchone()
        row = {
            "row": upsert_model_row(
                conn,
                {
                    "name": name,
                    "backend": backend,
                    "served_model_id": served_model_id,
                    "size_class": size_class,
                    "environment": environment,
                    "port": port,
                    "default_temperature": default_temperature,
                    "base_url": base_url or None,
                    "api_key_env": api_key_env or None,
                    "strip_tools": strip_tools,
                    "min_disk_gb": min_disk_gb,
                    "quantization": quantization or None,
                    "dtype": dtype or None,
                    "max_model_len": max_model_len,
                    "system_prefix": system_prefix or None,
                    "request_overrides": request_overrides or None,
                    "serving": serving or None,
                    "pricing": check_pricing(pricing) if pricing else None,
                    "budget_eur": budget_eur,
                    "revision": revision or None,
                },
            )
        }
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
