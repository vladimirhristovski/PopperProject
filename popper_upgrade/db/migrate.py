from pathlib import Path

from pgvector.psycopg import register_vector

from popper_upgrade.db.connection import connect

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def migrate(conn):
    conn.execute(
        "create table if not exists public.schema_migrations "
        "(name text primary key, applied_at timestamptz not null default now())"
    )
    conn.commit()
    applied = {row["name"] for row in conn.execute("select name from public.schema_migrations").fetchall()}
    conn.commit()
    done = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if path.name in applied:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("insert into public.schema_migrations (name) values (%s)", (path.name,))
        conn.commit()
        done.append(path.name)
    if done:
        register_vector(conn)
    return done


if __name__ == "__main__":
    with connect() as connection:
        for name in migrate(connection):
            print(name)
