import json
import os
import uuid
from dataclasses import dataclass

import pytest
from psycopg import connect as raw_connect
from psycopg import sql
from psycopg.conninfo import make_conninfo

from popper_upgrade.app.auth import create_user
from popper_upgrade.app.context import REGISTRY_DIR, AppContext
from popper_upgrade.app.main import build_app, credentials_check
from popper_upgrade.db.connection import DEFAULT_DSN, connect
from popper_upgrade.db.migrate import migrate
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeCurator, FakeEmbedder, run_config

ADMIN_PW = "admin-password-1"
USER_PW = "user-password-1"
APPROVED_TEXT = "compare outcome rates between two demographic groups"
REJECTED_TEXT = "regress outcome on a single numeric column"
PENDING_TEXT = "permutation test on matched pairs"
DESIGNER_LOG = "designer proposes a two sample test"


@dataclass
class LiveApp:
    url: str
    ctx: AppContext
    skills: dict
    run_id: str
    single_run_id: str


def _skill(applicability):
    return Skill(
        id=str(uuid.uuid4()),
        applicability=applicability,
        design_template="compare group means",
        preconditions="two groups",
        domain="social science",
        source_hypothesis="",
    )


def _seed(ctx):
    with ctx.conn() as conn:
        ctx.registry(conn).seed_from(REGISTRY_DIR)
        create_user(conn, "admin", ADMIN_PW, "admin", None)
        create_user(conn, "alice", USER_PW, "user", None)
        repo = ctx.skills(conn)
        ids = {}
        for key, text, approved in (
            ("approved", APPROVED_TEXT, True),
            ("rejected", REJECTED_TEXT, False),
            ("pending", PENDING_TEXT, None),
        ):
            ids[key] = repo.add_candidate_skill(_skill(text))
            if approved is not None:
                repo.set_approval(ids[key], approved, "curator", "seed", "entry")
        runs = ctx.runs(conn)
        run_id = runs.create_run(run_config(model_name="claude-sonnet-4-5"))
        ctx.trajectory_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.trajectory_dir / f"{run_id}.json"
        trajectory = {
            "log": {"designer": [DESIGNER_LOG], "executor": ["executor ran code"]},
            "tracked_tests": ["t1", "t2"],
            "tracked_stat": [0.03, 0.2],
            "res": False,
            "res_stat": 1.5,
            "parsed_result": {},
            "last_message": "done",
        }
        path.write_text(json.dumps(trajectory), encoding="utf-8")
        runs.add_design_reviews(
            run_id,
            [
                {
                    "round": 1,
                    "attempt": 1,
                    "proposal": "p",
                    "approved": True,
                    "reason": "looks fine",
                    "overridden": False,
                }
            ],
        )
        runs.finish_run(run_id, {"res": False, "res_stat": 1.5, "n_tests": 2, "trajectory_path": str(path)})
        single_id = runs.create_run(run_config(model_name="claude-sonnet-4-5", hypothesis="single test run"))
        single_path = ctx.trajectory_dir / f"{single_id}.json"
        single_path.write_text(
            json.dumps({**trajectory, "tracked_tests": ["t1"], "tracked_stat": [0.03]}), encoding="utf-8"
        )
        runs.finish_run(single_id, {"res": False, "res_stat": 3.0, "n_tests": 1, "trajectory_path": str(single_path)})
    return ids, run_id, single_id


@pytest.fixture(scope="module")
def e2e_dsn():
    base = os.environ.get("POPPER_DB_DSN", DEFAULT_DSN)
    name = f"popper_e2e_{uuid.uuid4().hex[:10]}"
    with raw_connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("create database {}").format(sql.Identifier(name)))
    dsn = make_conninfo(base, dbname=name)
    with connect(dsn) as conn:
        migrate(conn)
    yield dsn
    with raw_connect(base, autocommit=True) as admin:
        admin.execute(sql.SQL("drop database {} with (force)").format(sql.Identifier(name)))


@pytest.fixture(scope="module")
def live_app(e2e_dsn, tmp_path_factory):
    ctx = AppContext(
        e2e_dsn,
        FakeEmbedder(),
        tmp_path_factory.mktemp("trajectories"),
        curator_factory=lambda name: FakeCurator(),
        default_curator_model="claude-sonnet-4-5",
    )
    ids, run_id, single_run_id = _seed(ctx)
    demo = build_app(ctx)
    _, url, _ = demo.queue().launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        prevent_thread_lock=True,
        auth=credentials_check(ctx),
        quiet=True,
    )
    yield LiveApp(url.rstrip("/"), ctx, ids, run_id, single_run_id)
    demo.close()
