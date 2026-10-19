import psycopg
import pytest

from popper_upgrade.db.migrate import migrate

pytestmark = pytest.mark.db


def test_migrate_creates_all_schemas(db):
    schemas = {r["schema_name"] for r in db.execute("select schema_name from information_schema.schemata").fetchall()}
    extensions = {r["extname"] for r in db.execute("select extname from pg_extension").fetchall()}

    assert {"auth", "skills", "runs", "registry", "audit"} <= schemas
    assert "vector" in extensions


def test_migrate_is_idempotent(db):
    assert migrate(db) == []


def test_audit_events_are_append_only(db):
    db.execute("insert into audit.events (action, target_type, target_id) values ('a', 't', '1')")

    with pytest.raises(psycopg.errors.RaiseException):
        db.execute("update audit.events set action = 'b'")
    db.rollback()
    db.execute("insert into audit.events (action, target_type, target_id) values ('a', 't', '1')")
    with pytest.raises(psycopg.errors.RaiseException):
        db.execute("delete from audit.events")


def test_models_accept_openai_compatible_with_base_url(db):
    columns = "id, name, backend, served_model_id, size_class, environment, base_url, api_key_env"
    db.execute(
        f"insert into registry.models ({columns}) values (gen_random_uuid(), 'm1', 'openai_compatible', 'm1', "
        "'small', 'laptop', 'http://host:1/v1', 'KEY_ENV')"
    )
    row = db.execute("select strip_tools, api_key_env from registry.models where name = 'm1'").fetchone()

    assert row == {"strip_tools": True, "api_key_env": "KEY_ENV"}


def test_openai_compatible_requires_base_url(db):
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "insert into registry.models (id, name, backend, served_model_id, size_class, environment) "
            "values (gen_random_uuid(), 'm2', 'openai_compatible', 'm2', 'small', 'laptop')"
        )
    db.rollback()
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "insert into registry.models (id, name, backend, served_model_id, size_class, environment) "
            "values (gen_random_uuid(), 'm3', 'unknown', 'm3', 'small', 'laptop')"
        )
