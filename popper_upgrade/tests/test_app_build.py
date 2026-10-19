import gradio as gr
import pytest

import popper_upgrade.app.main as main_module
from popper_upgrade.app.context import AppContext
from popper_upgrade.app.main import build_app, make_curator_factory
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.fakes import COMPATIBLE_MODEL, OLLAMA_MODEL

HANDLER_API_NAMES = {
    "list_skills",
    "skill_detail",
    "propose_skill",
    "flag_skill",
    "set_library_approval",
    "edit_skill",
    "restore_skill",
    "delete_skill",
    "retrieval_preview",
    "preview_curation",
    "apply_overrides",
    "list_runs",
    "run_detail",
    "launch_start",
    "launch_stop",
    "launch_status",
    "list_users",
    "create_user_h",
    "set_role_h",
    "set_active_h",
    "upsert_model",
    "list_prompt_versions",
    "add_prompt_version",
    "import_bundle_h",
    "import_archive_h",
}


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    ctx = AppContext("postgresql://unused", None, tmp_path_factory.mktemp("trajectories"))
    return build_app(ctx)


def test_build_app_returns_blocks(demo):
    assert isinstance(demo, gr.Blocks)


def test_every_handler_has_api_name(demo):
    api_names = {fn.api_name for fn in demo.fns.values()}
    assert HANDLER_API_NAMES <= api_names


def test_no_event_has_an_implicit_api_name(demo):
    unnamed = [fn.name for fn in demo.fns.values() if fn.api_name not in HANDLER_API_NAMES and fn.api_name is not False]
    assert unnamed == []


class _Registry:
    def __init__(self, conn):
        self.models = {m["name"]: m for m in (COMPATIBLE_MODEL, OLLAMA_MODEL)}

    def model(self, name):
        return self.models[name]


def _patch_factory(monkeypatch):
    entered = []
    curators = []

    class Context:
        def __enter__(self):
            entered.append(1)
            counter = UsageCounter()
            counter.port = 5555
            return counter

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(main_module, "connect", lambda dsn: Connection())
    monkeypatch.setattr(main_module, "RegistryRepository", _Registry)
    monkeypatch.setattr(main_module, "serving_for", lambda model: Context())
    monkeypatch.setattr(main_module, "SkillCurator", lambda llm, port, api_key: curators.append((llm, port, api_key)))
    return entered, curators


def test_curator_for_a_compatible_model_goes_through_one_cached_proxy(monkeypatch):
    entered, curators = _patch_factory(monkeypatch)
    factory = make_curator_factory("dsn")

    factory(COMPATIBLE_MODEL["name"])
    factory(COMPATIBLE_MODEL["name"])

    assert entered == [1]
    assert curators == [("custom-endpoint-model", 5555, "EMPTY")] * 2


def test_curator_for_a_local_model_uses_its_registry_port(monkeypatch):
    entered, curators = _patch_factory(monkeypatch)

    make_curator_factory("dsn")(OLLAMA_MODEL["name"])

    assert entered == []
    assert curators == [("llama3.1:8b", 11435, "EMPTY")]
