import json
import sys
import types
from pathlib import Path

import pytest

import popper_upgrade
from local import common, run_anthropic, run_openai, run_other_api

REGISTRY = Path(popper_upgrade.__file__).parent / "registry" / "models.json"


@pytest.fixture
def no_database(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("the database must not be touched before the key check")

    stub = types.ModuleType("popper_upgrade.db.connection")
    stub.connect = refuse
    monkeypatch.setitem(sys.modules, "popper_upgrade.db.connection", stub)


@pytest.mark.parametrize(
    ("module", "argv", "variable"),
    [
        (run_openai, ["--model", "gpt-6-sol", "--library-from", "lib.json"], "OPENAI_API_KEY"),
        (run_anthropic, ["--model", "claude-sonnet-5-5", "--library-from", "lib.json"], "ANTHROPIC_API_KEY"),
        (run_other_api, ["--provider", "grok", "--model", "grok-4.7"], "XAI_API_KEY"),
        (run_other_api, ["--provider", "deepseek", "--model", "deepseek-flash"], "DEEPSEEK_API_KEY"),
    ],
)
def test_missing_key_refuses_before_any_work_and_names_the_variable(
    module, argv, variable, monkeypatch, capsys, no_database
):
    monkeypatch.delenv(variable, raising=False)

    assert module.main(argv) == 2

    err = capsys.readouterr().err
    assert variable in err
    assert "Nothing was run" in err


def test_models_are_limited_per_script(capsys):
    with pytest.raises(SystemExit):
        run_openai.main(["--model", "claude-sonnet-5-5", "--library-from", "lib.json"])
    with pytest.raises(SystemExit):
        run_anthropic.main(["--model", "gpt-6-sol", "--library-from", "lib.json"])
    with pytest.raises(SystemExit):
        run_other_api.main(["--provider", "grok", "--model", "deepseek-flash"])
    with pytest.raises(SystemExit):
        run_other_api.main(["--provider", "other", "--model", "x"])
    assert "invalid choice" in capsys.readouterr().err


def test_key_variables_match_the_registry():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    envs = {row["name"]: row.get("api_key_env") for row in registry}
    for provider, models in (
        ("openai", run_openai.MODELS),
        ("anthropic", run_anthropic.MODELS),
        ("grok", run_other_api.PROVIDER_MODELS["grok"]),
        ("deepseek", run_other_api.PROVIDER_MODELS["deepseek"]),
    ):
        assert {envs[name] for name in models} == {common.KEY_ENVS[provider]}


def test_paid_scripts_default_to_the_qwen_round_spec_and_need_a_library(capsys):
    for module, model in ((run_openai, "gpt-6-sol"), (run_anthropic, "claude-sonnet-5-5")):
        with pytest.raises(SystemExit):
            module.main(["--model", model])
        assert "--library-from" in capsys.readouterr().err
    assert common.QWEN_SPEC_PATH.name == "qwen_round.json"
    assert common.QWEN_SPEC_PATH.is_file()
    assert common.build_parser("p", ("m",), common.QWEN_SPEC_PATH, True).parse_args(
        ["--model", "m", "--library-from", "x"]
    ).spec == str(common.QWEN_SPEC_PATH)
    assert run_openai.MODELS == ("gpt-6-sol",)
    assert run_anthropic.MODELS == ("claude-sonnet-5-5",)
    spec = json.loads(common.QWEN_SPEC_PATH.read_text(encoding="utf-8"))
    assert set(run_openai.MODELS + run_anthropic.MODELS) <= set(spec["models"])


def test_library_must_exist(tmp_path):
    assert common.load_library(None, object()) is None
    with pytest.raises(ValueError, match="library snapshot not found"):
        common.load_library(str(tmp_path / "missing.json"), object())


@pytest.fixture
def wired(monkeypatch, tmp_path):
    calls = {}
    model = next(row for row in json.loads(REGISTRY.read_text(encoding="utf-8")) if row["name"] == "gpt-6-sol")

    calls["base_model"] = model

    class Conn:
        autocommit = False

        def close(self):
            calls["closed"] = True

    class Repo:
        def __init__(self, conn):
            pass

        def model(self, name):
            return calls.get("model", model)

    class Library:
        def __init__(self, path, embedder):
            calls["library"] = path

    def fake_run_experiment(*args, **kwargs):
        calls["kwargs"] = kwargs
        calls["budget_class"] = type(kwargs["budget"]).__name__
        if kwargs["budget"] is not None:
            kwargs["budget"].begin({"evaluate": {"baseline": 1}}, args[0], model)
            kwargs["budget"].exhausted = calls.get("exhausted", False)
        return "exp-1"

    def forbidden(*args, **kwargs):
        pytest.fail("learning must not be built when a library is supplied")

    modules = {
        "popper_upgrade.db.connection": {"connect": lambda: Conn()},
        "popper_upgrade.db.registry_repo": {"RegistryRepository": Repo},
        "popper_upgrade.db.snapshot": {"SnapshotSkillSource": Library},
        "popper_upgrade.eval.learning": {"db_learning": forbidden},
        "popper_upgrade.eval.report": {"build_report": lambda conn, eid: {}, "write_report": lambda report, out: None},
        "popper_upgrade.eval.run_experiment": {
            "_build_sink": lambda spec, conn, source, embedder: types.SimpleNamespace(model_costs=lambda name: []),
            "_freeze_source": lambda *args: object(),
            "default_snapshot_dir": lambda: tmp_path,
            "run_experiment": fake_run_experiment,
        },
        "popper_upgrade.serving.model_status": {
            "check_model_status": lambda m: {"model": m["name"], "state": "listed", "notes": {}, "detail": ""},
        },
        "popper_upgrade.skills.embedding": {"Embedder": lambda: object()},
    }
    for name, attrs in modules.items():
        __import__(name)
        for attr, value in attrs.items():
            monkeypatch.setattr(sys.modules[name], attr, value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    (tmp_path / "lib.json").write_text("{}", encoding="utf-8")
    calls["library_path"] = str(tmp_path / "lib.json")
    return calls


def test_paid_run_uses_the_library_the_cost_guard_and_the_status_check(wired, capsys):
    code = run_openai.main(["--model", "gpt-6-sol", "--library-from", wired["library_path"]])

    out = capsys.readouterr().out
    assert code == 0
    assert wired["library"] == wired["library_path"]
    assert wired["kwargs"]["library"] is not None
    assert wired["kwargs"]["learning"] is None
    assert wired["budget_class"] == "CostGuard"
    assert wired["kwargs"]["on_serving"] == wired["kwargs"]["budget"].attach
    assert "[model status] gpt-6-sol: listed" in out
    assert "total spent EUR 0.00 of EUR" in out
    assert wired["closed"] is True


def test_exhausted_budget_ends_with_a_distinct_exit_code(wired, capsys):
    wired["exhausted"] = True

    code = run_openai.main(["--model", "gpt-6-sol", "--library-from", wired["library_path"]])

    assert code == 3
    assert "budget stopped the experiment" in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["anthropic", "openai", "openai_compatible", "anthropic_messages"])
def test_paid_model_without_pricing_is_refused_before_anything_runs(wired, capsys, backend):
    wired["model"] = {k: v for k, v in wired["base_model"].items() if k != "pricing"} | {"backend": backend}

    code = run_openai.main(["--model", "gpt-6-sol", "--library-from", wired["library_path"]])

    err = capsys.readouterr().err
    assert code == 2
    assert "has no pricing" in err
    assert "--allow-unpriced" in err
    assert "Nothing was run" in err
    assert "kwargs" not in wired
    assert wired["closed"] is True


def test_allow_unpriced_runs_unguarded_with_a_loud_warning(wired, capsys):
    wired["model"] = {k: v for k, v in wired["base_model"].items() if k != "pricing"}

    code = run_openai.main(["--model", "gpt-6-sol", "--library-from", wired["library_path"], "--allow-unpriced"])

    captured = capsys.readouterr()
    assert code == 0
    assert "WARNING" in captured.err
    assert "no budget limit" in captured.err
    assert wired["kwargs"]["budget"] is None
    assert wired["kwargs"]["on_serving"] is None


def test_priced_models_do_not_need_the_flag_and_stay_guarded(wired):
    assert run_openai.main(["--model", "gpt-6-sol", "--library-from", wired["library_path"]]) == 0
    assert wired["budget_class"] == "CostGuard"

