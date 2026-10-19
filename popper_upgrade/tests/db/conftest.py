import contextlib
import os

import pytest

from popper_upgrade.app.auth import create_user
from popper_upgrade.app.context import AppContext
from popper_upgrade.db.connection import DEFAULT_DSN, connect
from popper_upgrade.db.migrate import migrate
from popper_upgrade.tests.fakes import FakeEmbedder


@pytest.fixture(scope="session")
def dsn():
    value = os.environ.get("POPPER_DB_DSN", DEFAULT_DSN)
    with connect(value) as conn:
        migrate(conn)
    return value


@pytest.fixture
def db(dsn):
    conn = connect(dsn)
    conn.execute("select 1")
    yield conn
    conn.rollback()
    conn.close()


class SharedContext(AppContext):
    def __init__(self, conn, embedder, trajectory_dir, **kwargs):
        super().__init__("shared", embedder, trajectory_dir, **kwargs)
        self._shared = conn

    @contextlib.contextmanager
    def conn(self):
        yield self._shared


@pytest.fixture
def make_ctx(db, tmp_path):
    def build(**kwargs):
        return SharedContext(db, FakeEmbedder(), tmp_path / "trajectories", **kwargs)

    return build


@pytest.fixture
def ctx(make_ctx):
    return make_ctx()


@pytest.fixture
def admin_user(db):
    return create_user(db, "admin", "admin-password-1", "admin", None)


@pytest.fixture
def plain_user(db):
    return create_user(db, "alice", "alice-password-1", "user", None)
