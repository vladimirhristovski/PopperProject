import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from popper_upgrade.models import ServingError, check_environment, popper_kwargs_for, serving_for
from popper_upgrade.serving.anthropic import to_chat_response, to_chat_usage, to_messages_request
from popper_upgrade.serving.rewrite import CHAT_PATH, request_rewriter
from popper_upgrade.tests.fakes import post_chat
from popper_upgrade.tests.test_models import _registry_models

CACHE = {"type": "ephemeral"}
PRICING = {"input": 2, "cache_read": 0.1, "cache_write": 2.5, "output": 10}


def _block(text, cached=False):
    return {"type": "text", "text": text, **({"cache_control": CACHE} if cached else {})}


def test_single_turn_with_system_caches_the_system_prefix():
    request = to_messages_request(
        {
            "model": "claude-x",
            "max_tokens": 77,
            "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
        }
    )

    assert request == {
        "model": "claude-x",
        "max_tokens": 77,
        "system": [_block("be brief", cached=True)],
        "messages": [{"role": "user", "content": [_block("hi")]}],
    }


def test_multi_turn_marks_system_and_the_last_earlier_turn_only():
    request = to_messages_request(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "developer", "content": [{"type": "text", "text": "s2"}]},
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": [{"type": "text", "text": "q2a"}, {"type": "text", "text": "q2b"}]},
            ],
        }
    )

    assert request["system"] == [_block("s1"), _block("s2", cached=True)]
    assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]
    assert request["messages"][0]["content"] == [_block("q1")]
    assert request["messages"][1]["content"] == [_block("a1", cached=True)]
    assert request["messages"][2]["content"] == [_block("q2a"), _block("q2b")]
    marks = json.dumps(request).count("cache_control")
    assert marks == 2


def test_stop_temperature_and_default_max_tokens():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}

    one = to_messages_request({**base, "stop": "END", "temperature": 1.7})
    many = to_messages_request({**base, "stop": ["A", "B"], "max_completion_tokens": 9})

    assert one["stop_sequences"] == ["END"]
    assert one["temperature"] == 1.0
    assert one["max_tokens"] == 8192
    assert many["stop_sequences"] == ["A", "B"]
    assert many["max_tokens"] == 9
    assert "temperature" not in many


def test_tools_and_openai_only_fields_are_not_forwarded():
    request = to_messages_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function"}],
            "tool_choice": "auto",
            "presence_penalty": 0.5,
            "seed": 3,
            "n": 1,
        }
    )

    assert set(request) == {"model", "max_tokens", "messages"}


def test_empty_text_is_dropped_and_bad_requests_are_refused():
    request = to_messages_request(
        {"model": "m", "messages": [{"role": "user", "content": " "}, {"role": "user", "content": "real"}]}
    )
    assert request["messages"] == [{"role": "user", "content": [_block("real")]}]
    for bad in (
        {"model": "m", "messages": [{"role": "system", "content": "only system"}]},
        {"model": "m", "messages": [{"role": "tool", "content": "x"}]},
        {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"model": "m", "n": 2, "messages": [{"role": "user", "content": "x"}]},
        {"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}]},
    ):
        with pytest.raises(ValueError):
            to_messages_request(bad)


def test_thinking_payload_comes_from_the_registry_overrides():
    model = _registry_models()["claude-sonnet-5-5"]
    body = json.dumps(
        {"model": "anthropic/claude-sonnet-5-5", "n": 1, "messages": [{"role": "user", "content": "x"}]}
    ).encode()

    rewritten = request_rewriter(model)(CHAT_PATH, body)
    request = to_messages_request(json.loads(rewritten))

    assert model["request_overrides"]["set"]["thinking"] == {"type": "between_tools"}
    assert request["thinking"] == {"type": "between_tools"}
    assert request["model"] == "claude-sonnet-5-5"

    corrected = {**model, "request_overrides": {"set": {"model": "claude-sonnet-5-5", "thinking": {"type": "other"}}}}
    again = to_messages_request(json.loads(request_rewriter(corrected)(CHAT_PATH, body)))
    assert again["thinking"] == {"type": "other"}


def test_response_translation_carries_text_finish_reason_and_cache_usage():
    chat = to_chat_response(
        {
            "id": "msg_1",
            "model": "claude-x",
            "stop_reason": "max_tokens",
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 3000,
                "output_tokens": 40,
            },
        }
    )

    assert chat["object"] == "chat.completion"
    assert chat["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert chat["choices"][0]["finish_reason"] == "length"
    assert chat["usage"] == {
        "prompt_tokens": 3210,
        "completion_tokens": 40,
        "total_tokens": 3250,
        "input_tokens": 10,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 3000,
        "prompt_tokens_details": {"cached_tokens": 3000},
    }
    assert to_chat_usage({})["prompt_tokens"] == 0
    assert to_chat_response({"content": [], "stop_reason": "end_turn"})["choices"][0]["finish_reason"] == "stop"
    assert "usage" not in to_chat_response({"content": []})


class MessagesUpstream:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.headers = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _answer(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.headers.append(dict(self.headers))
                outer.requests.append((self.path, json.loads(self.rfile.read(length))))
                status, payload = outer.responses.pop(0)
                self._answer(status, json.dumps(payload).encode())

            def do_GET(self):
                outer.headers.append(dict(self.headers))
                outer.requests.append((self.path, None))
                self._answer(200, json.dumps({"data": [{"id": "claude-sonnet-5-5"}]}).encode())

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _sonnet(upstream, **changes):
    return {
        **_registry_models()["claude-sonnet-5-5"],
        "id": "id-sonnet",
        "base_url": f"http://127.0.0.1:{upstream.port}/v1",
        **changes,
    }


def _answer(text="ok", **usage):
    return (
        200,
        {
            "id": "msg_1",
            "model": "claude-sonnet-5-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 100, "output_tokens": 20, **usage},
        },
    )


def test_serving_translates_calls_sends_native_headers_and_prices_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    upstream = MessagesUpstream(
        [_answer("first", cache_creation_input_tokens=1000), _answer("second", cache_read_input_tokens=1000)]
    )
    try:
        with serving_for(_sonnet(upstream)) as counter:
            first = json.loads(
                post_chat(
                    counter.port,
                    {
                        "model": "anthropic/claude-sonnet-5-5",
                        "n": 1,
                        "max_tokens": 50,
                        "stop": ["STOP"],
                        "tools": [{"type": "function"}],
                        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
                    },
                )
            )
            second = json.loads(post_chat(counter.port, {"messages": [{"role": "user", "content": "q2"}]}))
            assert counter.snapshot() == (1100 + 1100, 40)
            assert counter.cache_snapshot() == (1000, 1000)
            expected = (100 * 2 + 1000 * 2.5 + 20 * 10) / 1e6 + (100 * 2 + 1000 * 0.1 + 20 * 10) / 1e6
            assert counter.cost_usd() == pytest.approx(expected)
            assert counter.total_cost_usd() == pytest.approx(expected)
            counter.reset()
            assert counter.cost_usd() is None
            assert counter.total_cost_usd() == pytest.approx(expected)
    finally:
        upstream.close()

    path, sent = upstream.requests[0]
    assert path == "/v1/messages"
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["thinking"] == {"type": "between_tools"}
    assert sent["stop_sequences"] == ["STOP"]
    assert sent["max_tokens"] == 50
    assert sent["system"] == [_block("sys", cached=True)]
    assert "tools" not in sent
    headers = {k.lower(): v for k, v in upstream.headers[0].items()}
    assert headers["x-api-key"] == "sk-test"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers
    assert first["choices"][0]["message"]["content"] == "first"
    assert first["usage"]["cache_creation_input_tokens"] == 1000
    assert second["usage"]["cache_read_input_tokens"] == 1000


def test_upstream_errors_keep_status_and_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    upstream = MessagesUpstream(
        [(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "thinking.type: bad value"}})]
    )
    try:
        with serving_for(_sonnet(upstream)) as counter:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post_chat(counter.port, {"messages": [{"role": "user", "content": "q"}]})
            assert counter.snapshot() == (None, None)
    finally:
        upstream.close()

    body = json.loads(caught.value.read())
    assert caught.value.code == 400
    assert body["error"]["message"] == "thinking.type: bad value"
    assert body["error"]["type"] == "invalid_request_error"


def test_untranslatable_request_is_a_400_without_calling_upstream(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    upstream = MessagesUpstream([])
    try:
        with serving_for(_sonnet(upstream)) as counter:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post_chat(counter.port, {"messages": [{"role": "tool", "content": "x"}]})
    finally:
        upstream.close()

    assert caught.value.code == 400
    assert upstream.requests == []


def test_models_listing_is_relayed_with_native_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    upstream = MessagesUpstream([])
    try:
        with serving_for(_sonnet(upstream)) as counter:
            url = f"http://127.0.0.1:{counter.port}/v1/models"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read())
    finally:
        upstream.close()

    assert data["data"][0]["id"] == "claude-sonnet-5-5"
    assert {k.lower(): v for k, v in upstream.headers[0].items()}["x-api-key"] == "sk-test"


def test_backend_validation_and_serving_checks(monkeypatch):
    model = _registry_models()["claude-sonnet-5-5"]
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert model["backend"] == "anthropic_messages"
    assert popper_kwargs_for(model, 4000)["server_port"] == 4000
    with pytest.raises(ValueError, match="serving port"):
        popper_kwargs_for(model)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        check_environment(model, "laptop")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    check_environment(model, "laptop")
    with pytest.raises(ValueError, match="needs an api_key_env"), serving_for({**model, "api_key_env": None}):
        pytest.fail("serving must not start")
    closed = {**model, "base_url": "http://127.0.0.1:1/v1"}
    with pytest.raises(ServingError, match="not reachable"), serving_for(closed):
        pytest.fail("serving must not start")


def test_registry_entries_carry_prices_and_budgets():
    models = _registry_models()
    sol, sonnet = models["gpt-6-sol"], models["claude-sonnet-5-5"]

    assert sol["pricing"] == {"input": 2, "cache_read": 0.2, "output": 10}
    assert sonnet["pricing"] == PRICING
    assert (sol["budget_eur"], sonnet["budget_eur"]) == (20, 20)


def test_failed_and_usageless_calls_are_charged_at_the_worst_case(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    ok_without_usage = (200, {"id": "m", "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})
    upstream = MessagesUpstream([(500, {"error": {"message": "overloaded"}}), ok_without_usage])
    request = {"messages": [{"role": "user", "content": "q"}], "max_tokens": 1000}
    output_price = _registry_models()["claude-sonnet-5-5"]["pricing"]["output"]
    floor = 1000 * output_price / 1e6
    try:
        with serving_for(_sonnet(upstream)) as counter:
            with pytest.raises(urllib.error.HTTPError) as caught:
                post_chat(counter.port, request)
            assert caught.value.code == 500
            assert floor < counter.total_cost_usd() < floor * 1.05
            after_error = counter.total_cost_usd()
            post_chat(counter.port, request)
            assert counter.total_cost_usd() - after_error > floor
    finally:
        upstream.close()

    out = capsys.readouterr().out
    assert "worst-case" in out
    assert "no usage" in out


def test_streaming_is_a_400_and_the_spend_limit_is_a_429(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    upstream = MessagesUpstream([_answer("one", output_tokens=100000)])
    request = {"messages": [{"role": "user", "content": "q"}]}
    try:
        with serving_for(_sonnet(upstream)) as counter:
            counter.set_limit(0.5)
            with pytest.raises(urllib.error.HTTPError) as streamed:
                post_chat(counter.port, {**request, "stream": True})
            assert streamed.value.code == 400
            assert "streaming" in json.loads(streamed.value.read())["error"]["message"]
            post_chat(counter.port, request)
            with pytest.raises(urllib.error.HTTPError) as limited:
                post_chat(counter.port, request)
    finally:
        upstream.close()

    body = json.loads(limited.value.read())
    assert limited.value.code == 429
    assert body["error"]["type"] == "budget_exceeded"
    assert len(upstream.requests) == 1
