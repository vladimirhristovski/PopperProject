import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

DEFAULT_DSN = "postgresql://popper:popper@localhost:5433/popper"


def connect(dsn=None):
    conn = psycopg.connect(dsn or os.environ.get("POPPER_DB_DSN", DEFAULT_DSN), row_factory=dict_row)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        conn.rollback()
    return conn
