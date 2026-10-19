import json

import pytest

import popper_upgrade.models as models_module
from popper_upgrade.models import ServingError, check_environment, popper_kwargs_for, serving_for
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
    assert server.instances[0].calls == ["start", "start_proxy", "stop_proxy", "stop"]
    assert proxy.instances[0].calls == ["start", "stop"]


def test_vllm_start_failure_raises_and_stops(fake_vllm):
    server, proxy = fake_vllm
    server.start_result = False

    with pytest.raises(ServingError, match="vLLM did not start"), serving_for(VLLM_MODEL):
        pytest.fail("serving must not start")

    assert server.instances[0].calls == ["start", "stop_proxy", "stop"]
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
