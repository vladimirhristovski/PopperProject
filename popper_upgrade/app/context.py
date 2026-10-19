import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from popper_upgrade.db.connection import connect
from popper_upgrade.db.registry_repo import RegistryRepository
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.skills_repo import SkillRepository

REGISTRY_DIR = Path(__file__).resolve().parent.parent / "registry"


@dataclass
class AppContext:
    dsn: str
    embedder: object
    trajectory_dir: Path
    curator_factory: Callable[[str], object] | None = None
    default_curator_model: str | None = None

    @contextlib.contextmanager
    def conn(self):
        connection = connect(self.dsn)
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def skills(self, conn):
        return SkillRepository(conn, self.embedder)

    def runs(self, conn):
        return RunRepository(conn, self.skills(conn))

    def registry(self, conn):
        return RegistryRepository(conn)
