import contextlib
import threading
import time
from types import SimpleNamespace

import psycopg
import pytest

from popper_upgrade.app.context import REGISTRY_DIR
from popper_upgrade.app.handlers import admin as admin_handlers
from popper_upgrade.app.handlers import curation as curation_handlers
from popper_upgrade.app.handlers import launch as launch_module
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.fakes import FakeCurator

pytestmark = pytest.mark.db


class _Connection:
    autocommit = False

    def close(self):
        pass


class _FakeRun:
    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = []
        self.log = {"designer": ["first design"], "executor": []}
        self.review_gate = SimpleNamespace(
            reviews=[{"round": 1, "attempt": 1, "approved": False, "reason": "weak", "overridden": False}]
        )

    def __call__(self, model, hypotheses, arm, source, sink, environment, **kwargs):
        self.calls.append((model, hypotheses, arm, environment, kwargs))
        sink.create_run({"hypothesis": hypotheses[0]["text"]})
        runner = SimpleNamespace(agent=SimpleNamespace(log=self.log), review_gate=self.review_gate)
        kwargs["on_session_start"](SimpleNamespace(runner=runner))
        self.started.set()
        self.release.wait(10)
        return ["run"]


class _Sink:
    def create_run(self, config):
        return "run-id"


@pytest.fixture
def fake_run(monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(launch_module, "run_hypotheses", fake)
    monkeypatch.setattr(launch_module, "connect", lambda dsn: _Connection())
    monkeypatch.setattr(launch_module, "RunRepository", lambda conn, skills: _Sink())
    monkeypatch.setattr(launch_module, "SkillRepository", lambda conn, embedder: object())

    @contextlib.contextmanager
    def serving(model):
        yield UsageCounter()

    monkeypatch.setattr(launch_module, "serving_for", serving)
    yield fake
    fake.release.set()


@pytest.fixture
def launcher(ctx, db):
    ctx.registry(db).seed_from(REGISTRY_DIR)
    manager = launch_module.LaunchManager(ctx)
    yield manager
    manager.stop()


def _wait_done(manager):
    deadline = time.monotonic() + 10
    while not manager.status()["done"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_second_launch_refused(fake_run, launcher, plain_user):
    launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [0], "baseline", False, None)
    assert fake_run.started.wait(10)
    with pytest.raises(RuntimeError, match="a run is already in progress"):
        launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [1], "baseline", False, None)
    fake_run.release.set()
    _wait_done(launcher)
    launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [1], "baseline", False, None)
    _wait_done(launcher)
    assert len(fake_run.calls) == 2


def test_status_exposes_live_log(fake_run, launcher, plain_user):
    assert launcher.status()["done"] is True
    launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [0, 2], "skills_reviewer", False, None)
    assert fake_run.started.wait(10)
    status = launcher.status()
    assert status["done"] is False
    assert status["run_id"] == "run-id"
    assert status["hypothesis"] == fake_run.calls[0][1][0]["text"]
    assert status["log"]["designer"] == ["first design"]
    assert status["log"]["reviewer"] == ["round 1 attempt 1: rejected - weak"]
    assert set(status["log"]) == {
        "designer",
        "executor",
        "relevance_checker",
        "summarizer",
        "sequential_testing",
        "reviewer",
    }
    fake_run.log["designer"].append("second design")
    assert launcher.status()["log"]["designer"] == ["first design", "second design"]
    fake_run.release.set()
    _wait_done(launcher)
    model, hypotheses, arm, environment, kwargs = fake_run.calls[0]
    assert (model["name"], arm, environment) == ("claude-sonnet-4-5", "skills_reviewer", "laptop")
    assert len(hypotheses) == 2
    assert kwargs["launched_by"] == plain_user.id
    assert kwargs["human_review"] is False


def test_curation_token_supplies_overrides_and_human_review(fake_run, launcher, ctx, plain_user):
    curator = FakeCurator()
    ctx.curator_factory = lambda name: curator
    token = curation_handlers.preview_curation(ctx, plain_user, "claude-sonnet-4-5", ["H"], [])["token"]
    curation_handlers.apply_overrides(token, {})
    launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [0], "skills", False, token)
    assert fake_run.started.wait(10)
    kwargs = fake_run.calls[0][4]
    assert kwargs["human_review"] is True
    assert kwargs["overrides_provider"](None) == {}
    fake_run.release.set()
    _wait_done(launcher)


def test_launch_validates_inputs(fake_run, launcher, plain_user):
    with pytest.raises(ValueError):
        launcher.start(plain_user, "casperhansen/llama-3.3-70b-instruct-awq", "gender_v1", [0], "baseline", False, None)
    with pytest.raises(ValueError):
        launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [], "baseline", False, None)
    with pytest.raises(KeyError):
        launcher.start(plain_user, "claude-sonnet-4-5", "gender_v1", [0], "baseline", False, "unknown-token")
    assert launcher.status()["done"] is True


def test_import_bundle_admin_only(ctx, db, admin_user, plain_user, tmp_path):
    (tmp_path / "manifest.json").write_text('{"snapshot_id": "snap"}', encoding="utf-8")
    with pytest.raises(PermissionError):
        admin_handlers.import_bundle_h(ctx, plain_user, str(tmp_path))
    assert db.execute("select count(*) as n from audit.events where action like 'bundle%'").fetchone()["n"] == 0
    counts = admin_handlers.import_bundle_h(ctx, admin_user, str(tmp_path))
    assert counts["runs"] == 0
    actions = db.execute("select action, user_id::text as user_id from audit.events where action like 'bundle%'")
    assert {(row["action"], row["user_id"]) for row in actions.fetchall()} == {
        ("bundle_import", None),
        ("bundle.import", admin_user.id),
    }


def test_admin_handlers_reject_users(ctx, plain_user):
    for call in (
        lambda: admin_handlers.list_users(ctx, plain_user),
        lambda: admin_handlers.create_user_h(ctx, plain_user, "carol", "carol-password-1", "user"),
        lambda: admin_handlers.set_role_h(ctx, plain_user, plain_user.id, "admin"),
        lambda: admin_handlers.set_active_h(ctx, plain_user, plain_user.id, False),
        lambda: admin_handlers.upsert_model(ctx, plain_user, "m", "ollama", "m", "small", "laptop"),
        lambda: admin_handlers.list_prompt_versions(ctx, plain_user),
        lambda: admin_handlers.add_prompt_version(ctx, plain_user, "curator", "text"),
    ):
        with pytest.raises(PermissionError):
            call()


def test_admin_model_and_prompt_management(ctx, db, admin_user):
    admin_handlers.upsert_model(ctx, admin_user, "tiny:1b", "ollama", "tiny:1b", "small", "laptop", 11435, 0.2)
    admin_handlers.upsert_model(ctx, admin_user, "tiny:1b", "ollama", "tiny:1b", "small", "laptop", 11436, 0.2)
    assert ctx.registry(db).model("tiny:1b")["port"] == 11436
    events = db.execute("select before, after from audit.events where action = 'model.upsert' order by id")
    first, second = events.fetchall()
    assert first["before"] is None
    assert second["before"]["port"] == 11435
    version_id = admin_handlers.add_prompt_version(ctx, admin_user, "curator", "be strict")
    assert admin_handlers.add_prompt_version(ctx, admin_user, "curator", "be strict") == version_id
    assert [row["id"] for row in admin_handlers.list_prompt_versions(ctx, admin_user)] == [version_id]
    with pytest.raises(ValueError):
        admin_handlers.add_prompt_version(ctx, admin_user, "curator", "  ")


def test_admin_saves_openai_compatible_model(ctx, db, admin_user):
    admin_handlers.upsert_model(
        ctx,
        admin_user,
        "endpoint-model",
        "openai_compatible",
        "endpoint-model",
        "large",
        "cluster",
        None,
        None,
        "http://node:8000/v1",
        "ENDPOINT_KEY",
        False,
    )
    row = ctx.registry(db).model("endpoint-model")

    assert (row["backend"], row["base_url"], row["api_key_env"], row["strip_tools"]) == (
        "openai_compatible",
        "http://node:8000/v1",
        "ENDPOINT_KEY",
        False,
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        admin_handlers.upsert_model(ctx, admin_user, "bad", "openai_compatible", "bad", "small", "laptop")
