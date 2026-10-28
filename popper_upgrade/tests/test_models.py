import json
import math
from pathlib import Path

import pytest

import popper_upgrade.models as models_module
from popper_upgrade.models import ServingError, check_environment, popper_kwargs_for, serving_for
from popper_upgrade.serving import vllm as vllm_module
from popper_upgrade.serving.rewrite import CHAT_PATH, request_rewriter
from popper_upgrade.tests.fakes import (
    COMPATIBLE_MODEL,
    HOSTED_MODEL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
    VLLM_MODEL,
    FakeUpstream,
    post_chat,
)


def test_kwargs_for_each_backend():
    assert popper_kwargs_for(OLLAMA_MODEL) == {
        "llm": "llama3.1:8b",
        "is_locally_served": True,
        "server_port": 11435,
        "api_key": "EMPTY",
    }
    assert popper_kwargs_for(VLLM_MODEL) == {
        "llm": "casperhansen/llama-3.3-70b-instruct-awq",
        "is_locally_served": True,
        "server_port": 8001,
        "api_key": "EMPTY",
    }
    assert popper_kwargs_for(HOSTED_MODEL) == {"llm": "claude-sonnet-4-5"}
    assert popper_kwargs_for(OPENAI_MODEL) == {"llm": "gpt-4o"}


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        popper_kwargs_for({**OLLAMA_MODEL, "backend": "other"})


def test_hosted_model_refused_on_cluster():
    with pytest.raises(ValueError, match="model claude-sonnet-4-5 is not available on cluster"):
        check_environment(HOSTED_MODEL, "cluster")
    with pytest.raises(ValueError, match="model casperhansen/llama-3.3-70b-instruct-awq is not available on laptop"):
        check_environment(VLLM_MODEL, "laptop")
    check_environment(HOSTED_MODEL, "laptop")
    check_environment(VLLM_MODEL, "cluster")


def test_hosted_serving_yields_empty_counter():
    with serving_for(HOSTED_MODEL) as counter:
        assert counter.snapshot() == (None, None)


def test_ollama_serving_fails_fast_when_daemon_is_closed(monkeypatch):
    monkeypatch.setenv("POPPER_OLLAMA_UPSTREAM", "http://127.0.0.1:1")

    with pytest.raises(ServingError, match="Ollama is not reachable"), serving_for(OLLAMA_MODEL):
        pytest.fail("serving must not start")


def test_ollama_serving_starts_and_stops_proxy(monkeypatch):
    upstream = FakeUpstream(b"{}")
    monkeypatch.setenv("POPPER_OLLAMA_UPSTREAM", f"http://127.0.0.1:{upstream.port}")
    model = {**OLLAMA_MODEL, "port": 0}
    try:
        with serving_for(model) as counter:
            assert counter.snapshot() == (None, None)
    finally:
        upstream.close()


class _FakeServer:
    instances = []
    start_result = True

    def __init__(self, model, port, host="localhost", proxy_port=None):
        self.args = (model, port, proxy_port)
        self.calls = []
        type(self).instances.append(self)

    def start(self):
        self.calls.append("start")
        return type(self).start_result

    def start_proxy(self):
        self.calls.append("start_proxy")

    def stop_proxy(self):
        self.calls.append("stop_proxy")

    def stop(self):
        self.calls.append("stop")


class _FakeUsageProxy:
    instances = []

    def __init__(self, listen_port, upstream_port, counter):
        self.args = (listen_port, upstream_port)
        self.calls = []
        type(self).instances.append(self)

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")


@pytest.fixture
def fake_vllm(monkeypatch):
    server = type("Server", (_FakeServer,), {"instances": [], "start_result": True})
    proxy = type("Proxy", (_FakeUsageProxy,), {"instances": []})
    monkeypatch.setattr(models_module, "VllmServer", server)
    monkeypatch.setattr(models_module, "UsageProxy", proxy)
    return server, proxy


def test_vllm_serving_layers_proxies_and_stops_everything(fake_vllm):
    server, proxy = fake_vllm

    with serving_for(VLLM_MODEL) as counter:
        assert counter.snapshot() == (None, None)
        assert server.instances[0].calls == ["start", "start_proxy"]

    assert server.instances[0].args == ("casperhansen/llama-3.3-70b-instruct-awq", 8000, 8002)
    assert proxy.instances[0].args == (8001, 8002)
    assert server.instances[0].calls == ["start", "start_proxy", "stop", "stop_proxy"]
    assert proxy.instances[0].calls == ["start", "stop"]


def test_vllm_server_stops_before_proxies_wait_on_in_flight_requests(fake_vllm):
    server, proxy = fake_vllm
    order = []
    for owner, label in ((server, "server"), (proxy, "usage_proxy")):
        owner.stop = lambda self, stop=owner.stop, label=label: (order.append(label), stop(self))[1]

    with serving_for(VLLM_MODEL):
        pass

    assert order == ["server", "usage_proxy"]


def test_vllm_start_failure_raises_and_stops(fake_vllm):
    server, proxy = fake_vllm
    server.start_result = False

    with pytest.raises(ServingError, match="vLLM did not start"), serving_for(VLLM_MODEL):
        pytest.fail("serving must not start")

    assert server.instances[0].calls == ["start", "stop", "stop_proxy"]
    assert proxy.instances[0].calls == ["stop"]


def test_compatible_kwargs_need_a_serving_port():
    assert popper_kwargs_for(COMPATIBLE_MODEL, 40123) == {
        "llm": "custom-endpoint-model",
        "is_locally_served": True,
        "server_port": 40123,
        "api_key": "EMPTY",
    }
    with pytest.raises(ValueError, match="serving port"):
        popper_kwargs_for(COMPATIBLE_MODEL)


def test_kwargs_use_the_serving_port_for_local_backends():
    assert popper_kwargs_for(OLLAMA_MODEL, 4242)["server_port"] == 4242


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"base_url": None}, "needs a base_url"),
        ({"base_url": "ftp://host/v1"}, "invalid base_url"),
        ({"served_model_id": "gpt-4o-mini"}, "bypass the local endpoint"),
        ({"served_model_id": "claude-x"}, "bypass the local endpoint"),
    ],
)
def test_compatible_model_is_validated(changes, message):
    model = {**COMPATIBLE_MODEL, **changes}

    with pytest.raises(ValueError, match=message):
        popper_kwargs_for(model, 1)
    with pytest.raises(ValueError, match=message):
        check_environment(model, "laptop")


def test_compatible_environment_and_key_checks(monkeypatch):
    monkeypatch.delenv("CUSTOM_ENDPOINT_KEY", raising=False)

    with pytest.raises(ValueError, match="CUSTOM_ENDPOINT_KEY is not set"):
        check_environment(COMPATIBLE_MODEL, "laptop")
    with pytest.raises(ValueError, match="not available on cluster"):
        check_environment(COMPATIBLE_MODEL, "cluster")
    monkeypatch.setenv("CUSTOM_ENDPOINT_KEY", "secret")
    check_environment(COMPATIBLE_MODEL, "laptop")
    check_environment({**COMPATIBLE_MODEL, "api_key_env": None}, "laptop")


def _compatible(upstream, **changes):
    return {**COMPATIBLE_MODEL, "base_url": f"http://127.0.0.1:{upstream.port}/v1", **changes}


def test_compatible_serving_relays_with_key_and_tool_fixes(monkeypatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT_KEY", "secret")
    body = json.dumps(
        {
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            "choices": [{"message": {"tool_calls": [{"function": {"arguments": '{"a": 1}'}}]}}],
        }
    ).encode()
    upstream = FakeUpstream(body)
    try:
        with serving_for(_compatible(upstream)) as counter:
            assert counter.port not in (None, 0)
            returned = json.loads(
                post_chat(counter.port, {"messages": [], "tools": [{"type": "function"}], "tool_choice": "auto"})
            )
            assert counter.snapshot() == (7, 3)
        path, sent = upstream.requests[0]
        assert path == "/v1/chat/completions"
        assert "tools" not in json.loads(sent)
        assert "tool_choice" not in json.loads(sent)
        assert upstream.headers[0]["Authorization"] == "Bearer secret"
        assert returned["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    finally:
        upstream.close()


def test_compatible_serving_can_keep_tools_and_skip_the_key():
    upstream = FakeUpstream(b'{"x": "{}"}')
    try:
        model = _compatible(upstream, strip_tools=False, api_key_env=None)
        with serving_for(model) as counter:
            post_chat(counter.port, {"messages": [], "tools": [{"type": "function"}]})
        _, sent = upstream.requests[0]
        assert json.loads(sent)["tools"] == [{"type": "function"}]
        assert "Authorization" not in upstream.headers[0]
    finally:
        upstream.close()


def test_compatible_serving_fails_fast_when_endpoint_is_closed(monkeypatch):
    monkeypatch.setenv("CUSTOM_ENDPOINT_KEY", "secret")
    model = {**COMPATIBLE_MODEL, "base_url": "http://127.0.0.1:1/v1"}

    with pytest.raises(ServingError, match="Endpoint for custom-endpoint-model is not reachable"), serving_for(model):
        pytest.fail("serving must not start")


def test_serving_exposes_the_port_popper_should_use(monkeypatch):
    upstream = FakeUpstream(b"{}")
    monkeypatch.setenv("POPPER_OLLAMA_UPSTREAM", f"http://127.0.0.1:{upstream.port}")
    try:
        with serving_for({**OLLAMA_MODEL, "port": 0}) as counter:
            assert counter.port not in (None, 0)
        with serving_for(HOSTED_MODEL) as counter:
            assert counter.port is None
    finally:
        upstream.close()


class _LaunchServer(_FakeServer):
    instances = []

    def __init__(self, model, port, host="localhost", proxy_port=None, launch=None):
        super().__init__(model, port, host, proxy_port)
        self.launch = launch


def test_models_with_launch_settings_use_the_launch_server(monkeypatch):
    server = type("Server", (_LaunchServer,), {"instances": [], "start_result": True})
    plain = type("Plain", (_FakeServer,), {"instances": [], "start_result": True})
    monkeypatch.setattr(models_module, "VllmServer", plain)
    monkeypatch.setattr(models_module, "ModelVllmServer", server)
    monkeypatch.setattr(models_module, "UsageProxy", type("Proxy", (_FakeUsageProxy,), {"instances": []}))
    tuned = {**VLLM_MODEL, "name": "tuned", "served_model_id": "tuned", "dtype": "bfloat16"}

    with serving_for(tuned):
        pass
    with serving_for(VLLM_MODEL):
        pass

    assert [instance.launch for instance in server.instances] == [tuned]
    assert len(plain.instances) == 1


def test_vllm_proxy_applies_model_request_settings(monkeypatch):
    upstream = FakeUpstream(b"{}")
    captured = {}

    class Proxy(_FakeUsageProxy):
        instances = []

        def __init__(self, listen_port, upstream_port, counter, **options):
            super().__init__(listen_port, upstream_port, counter)
            captured.update(options)

    server = type("Server", (_FakeServer,), {"instances": [], "start_result": True})
    monkeypatch.setattr(models_module, "VllmServer", server)
    monkeypatch.setattr(models_module, "UsageProxy", Proxy)
    model = {
        **VLLM_MODEL,
        "request_overrides": {"set": {"chat_template_kwargs": {"enable_thinking": False}}},
        "system_prefix": "PREFIX",
    }
    try:
        with serving_for(model):
            body = captured["rewrite_request"]("/v1/chat/completions", json.dumps({"messages": []}).encode())
        assert json.loads(body) == {
            "messages": [{"role": "system", "content": "PREFIX"}],
            "chat_template_kwargs": {"enable_thinking": False},
        }
    finally:
        upstream.close()


def test_compatible_serving_substitutes_the_real_model_id_and_vendor_fixes():
    upstream = FakeUpstream(b"{}")
    try:
        overrides = {
            "set": {"model": "gpt-9"},
            "drop": ["temperature"],
            "rename": {"max_tokens": "max_completion_tokens"},
        }
        model = _compatible(upstream, strip_tools=False, api_key_env=None, request_overrides=overrides)
        with serving_for(model) as counter:
            post_chat(counter.port, {"model": "openai/gpt-9", "temperature": 0.7, "max_tokens": 5, "messages": []})
        _, sent = upstream.requests[0]
        assert json.loads(sent) == {"model": "gpt-9", "max_completion_tokens": 5, "messages": []}
    finally:
        upstream.close()


def _registry_models():
    path = Path(__file__).resolve().parents[1] / "registry" / "models.json"
    return {model["name"]: model for model in json.loads(path.read_text(encoding="utf-8"))}


def test_only_qwen_large_is_pinned_to_a_revision():
    pinned = {name: model["revision"] for name, model in _registry_models().items() if model.get("revision")}

    assert pinned == {"qwen-large": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"}


def test_llama_anchor_registry_entry():
    anchor = _registry_models()["llama-anchor"]
    plan = vllm_module.plan_launch(anchor, 80.0)
    command = vllm_module.ModelVllmServer("m", 8000, launch=anchor).command(plan)

    assert "revision" not in anchor
    assert plan["args"] == []
    assert not set(vllm_module.REVISION_FLAGS) & set(command)

    assert (anchor["backend"], anchor["environment"]) == ("vllm", "cluster")
    assert anchor["served_model_id"] == "casperhansen/llama-3.3-70b-instruct-awq"
    check_environment(anchor, "cluster")


def test_registry_uses_only_laptop_and_cluster_environments():
    models = _registry_models()

    assert {model["environment"] for model in models.values()} == {"laptop", "cluster"}
    assert "llama-small" not in models


def test_qwen_large_registry_entry_matches_the_checkpoint():
    model = _registry_models()["qwen-large"]
    serving = model["serving"]
    layers, full_attention_interval, kv_heads, head_dim, kv_bytes = 64, 4, 4, 256, 2
    kv_kb = 2 * kv_heads * head_dim * kv_bytes * (layers // full_attention_interval) // 1024
    weight_bytes = 55_563_006_776

    assert (model["backend"], model["environment"], model["served_model_id"]) == ("vllm", "cluster", "Qwen/Qwen3.8-27B")
    assert (model["quantization"], model["dtype"], model["max_model_len"]) == ("none", "bfloat16", 16384)
    assert serving["kv_kb_per_token"] == kv_kb == 64
    assert serving["weights_gb"] == math.ceil(weight_bytes / 1024**3 * 10) / 10
    assert serving["gated"] is False
    assert serving["fallback"] == "run popper_upgrade.sif llama-anchor instead"
    assert model["min_disk_gb"] * 1000**3 > weight_bytes
    check_environment(model, "cluster")

    plan = vllm_module.plan_launch(model, 80.0)
    assert (plan["max_model_len"], plan["quantization"], plan["dtype"]) == (16384, None, "bfloat16")
    assert model["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    assert plan["args"] == [
        "--revision",
        model["revision"],
        "--tokenizer-revision",
        model["revision"],
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
    ]
    with pytest.raises(vllm_module.LaunchError, match="llama-anchor"):
        vllm_module.plan_launch(model, 40.0)


def test_qwen_large_turns_thinking_off_server_wide_and_per_request(monkeypatch):
    model = _registry_models()["qwen-large"]
    monkeypatch.setenv("POPPER_VLLM_PYTHON", "/opt/vllm-env/bin/python")
    server = vllm_module.ModelVllmServer(model["served_model_id"], 8000, launch=model)

    command = server.command(vllm_module.plan_launch(model, 80.0))
    flag = command.index("--default-chat-template-kwargs")
    assert json.loads(command[flag + 1]) == {"enable_thinking": False}
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "16384"
    assert "--quantization" not in command
    assert command[command.index("--revision") + 1] == model["revision"]
    assert command[command.index("--tokenizer-revision") + 1] == model["revision"]
    assert command.count("--revision") == command.count("--tokenizer-revision") == 1

    body = json.dumps({"model": "Qwen/Qwen3.8-27B", "messages": []}).encode()
    sent = json.loads(request_rewriter(model)(CHAT_PATH, body))
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_paid_registry_entries_pin_the_round_settings():
    models = _registry_models()
    sol, sonnet = models["gpt-6-sol"], models["claude-sonnet-5-5"]

    assert (sol["backend"], sol["environment"], sol["served_model_id"]) == (
        "openai_compatible",
        "laptop",
        "openai/gpt-6-sol",
    )
    assert sol["request_overrides"]["set"] == {"model": "gpt-6-sol", "reasoning_effort": "none"}
    assert sol["request_overrides"]["max_output_tokens"] == 4096
    assert sol["request_overrides"]["drop"] == ["temperature"]
    assert sol["request_overrides"]["rename"] == {"max_tokens": "max_completion_tokens"}
    assert sol["api_key_env"] == "OPENAI_API_KEY"
    assert (sonnet["backend"], sonnet["environment"]) == ("anthropic_messages", "laptop")
    assert sonnet["served_model_id"] == "anthropic/claude-sonnet-5-5"
    assert sonnet["base_url"] == "https://api.anthropic.com/v1"
    assert sonnet["request_overrides"]["set"] == {"model": "claude-sonnet-5-5", "thinking": {"type": "between_tools"}}
    assert sonnet["request_overrides"]["max_output_tokens"] == 4096
    assert sonnet["api_key_env"] == "ANTHROPIC_API_KEY"


def test_qwen_large_registry_entry_pins_non_thinking_sampling_and_the_output_cap():
    overrides = _registry_models()["qwen-large"]["request_overrides"]

    assert overrides["set"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
    }
    assert (overrides["max_output_tokens"], overrides["context_margin"]) == (4096, 256)
