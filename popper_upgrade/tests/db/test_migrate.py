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


def test_anthropic_messages_backend_needs_a_base_url_and_accepts_pricing(db):
    columns = "id, name, backend, served_model_id, size_class, environment, base_url, pricing, budget_eur"
    db.execute(
        f"insert into registry.models ({columns}) values (gen_random_uuid(), 'm4', 'anthropic_messages', 'm4', "
        "'large', 'laptop', 'https://host/v1', '{\"input\": 1, \"output\": 2}'::jsonb, 20)"
    )
    row = db.execute("select pricing, budget_eur from registry.models where name = 'm4'").fetchone()

    assert row == {"pricing": {"input": 1, "output": 2}, "budget_eur": 20}
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "insert into registry.models (id, name, backend, served_model_id, size_class, environment) "
            "values (gen_random_uuid(), 'm5', 'anthropic_messages', 'm5', 'large', 'laptop')"
        )
    db.rollback()
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "insert into registry.models (id, name, backend, served_model_id, size_class, environment, budget_eur) "
            "values (gen_random_uuid(), 'm6', 'ollama', 'm6', 'small', 'laptop', 0)"
        )


def test_model_revision_must_be_a_full_lowercase_commit_sha(db):
    columns = "id, name, backend, served_model_id, size_class, environment, revision"
    db.execute(
        f"insert into registry.models ({columns}) values (gen_random_uuid(), 'r1', 'vllm', 'org/r1', 'large', "
        f"'cluster', '{'a1' * 20}')"
    )
    assert db.execute("select revision from registry.models where name = 'r1'").fetchone() == {"revision": "a1" * 20}
    for bad in ("main", "A1" * 20, "a1" * 19):
        db.rollback()
        with pytest.raises(psycopg.errors.CheckViolation):
            db.execute(
                f"insert into registry.models ({columns}) values (gen_random_uuid(), 'r2', 'vllm', 'org/r2', "
                f"'large', 'cluster', '{bad}')"
            )
