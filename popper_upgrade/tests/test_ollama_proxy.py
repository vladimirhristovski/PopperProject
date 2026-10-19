import json

import pytest

from popper_upgrade.serving.ollama_proxy import OllamaProxy
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.fakes import FakeUpstream, post_chat


@pytest.fixture
def make_proxy():
    started = []

    def make(body):
        upstream = FakeUpstream(body)
        counter = UsageCounter()
        proxy = OllamaProxy(0, f"http://127.0.0.1:{upstream.port}", counter)
        proxy.start()
        started.append((upstream, proxy))
        return upstream, proxy, counter

    yield make
    for upstream, proxy in started:
        proxy.stop()
        upstream.close()


def _response(**extra):
    message = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "f", "arguments": '{"x": 1, "y": [2]}'}}],
    }
    return json.dumps({"choices": [{"message": message}], **extra}).encode()


def test_strips_tools_from_request(make_proxy):
    upstream, proxy, _ = make_proxy(_response())

    post_chat(proxy.port, {"model": "m", "messages": [], "tools": [{"type": "function"}], "tool_choice": "auto"})

    [(path, sent)] = upstream.requests
    assert path == "/v1/chat/completions"
    assert json.loads(sent) == {"model": "m", "messages": []}


def test_reencodes_string_arguments(make_proxy):
    _, proxy, _ = make_proxy(_response())

    returned = json.loads(post_chat(proxy.port, {"messages": []}))

    assert returned["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == {"x": 1, "y": [2]}


def test_undecodable_arguments_are_left_alone(make_proxy):
    body = json.dumps(
        {"choices": [{"message": {"tool_calls": [{"function": {"arguments": "not json"}}]}}]}
    ).encode()
    _, proxy, _ = make_proxy(body)

    returned = json.loads(post_chat(proxy.port, {"messages": []}))

    assert returned["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_counts_usage(make_proxy):
    _, proxy, counter = make_proxy(_response(usage={"prompt_tokens": 11, "completion_tokens": 4}))

    post_chat(proxy.port, {"messages": []})
    post_chat(proxy.port, {"messages": []})

    assert counter.snapshot() == (22, 8)


def test_missing_usage_stays_none(make_proxy):
    _, proxy, counter = make_proxy(_response())

    post_chat(proxy.port, {"messages": []})

    assert counter.snapshot() == (None, None)


def test_upstream_defaults_to_environment(monkeypatch):
    monkeypatch.setenv("POPPER_OLLAMA_UPSTREAM", "http://host.docker.internal:11434")

    assert OllamaProxy()._upstream == "http://host.docker.internal:11434"
