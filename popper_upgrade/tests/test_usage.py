import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from popper_upgrade.serving.usage import UsageCounter, UsageProxy
from popper_upgrade.tests.fakes import FakeUpstream, post_chat


@pytest.fixture
def make_proxy():
    started = []

    def make(body):
        upstream = FakeUpstream(body)
        counter = UsageCounter()
        proxy = UsageProxy(0, upstream.port, counter, upstream_host="127.0.0.1")
        proxy.start()
        started.append((upstream, proxy))
        return upstream, proxy, counter

    yield make
    for upstream, proxy in started:
        proxy.stop()
        upstream.close()


def test_counter_is_none_until_first_usage():
    counter = UsageCounter()

    assert counter.snapshot() == (None, None)
    counter.add(None)
    counter.add({})
    assert counter.snapshot() == (None, None)
    counter.add({"prompt_tokens": 3, "completion_tokens": 2})
    counter.add({"prompt_tokens": 4})
    assert counter.snapshot() == (7, 2)
    counter.reset()
    assert counter.snapshot() == (None, None)


def test_counter_is_thread_safe():
    counter = UsageCounter()

    def work():
        for _ in range(500):
            counter.add({"prompt_tokens": 1, "completion_tokens": 2})

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert counter.snapshot() == (4000, 8000)


def test_usage_proxy_is_transparent(make_proxy):
    tool_call = {"function": {"arguments": '{"a": 1}'}}
    calls = json.dumps({"tool_calls": [tool_call]})
    usage = json.dumps({"prompt_tokens": 5, "completion_tokens": 6})
    body = f'{{"choices": [{{"message": {calls}}}],   "usage":\t{usage}}}\n'.encode()
    upstream, proxy, counter = make_proxy(body)
    request = {"messages": [], "tools": [{"type": "function"}], "tool_choice": "auto"}

    returned = post_chat(proxy.port, request)

    assert returned == body
    assert upstream.requests == [("/v1/chat/completions", json.dumps(request).encode())]
    assert counter.snapshot() == (5, 6)


def test_usage_proxy_reports_upstream_failure_as_502():
    counter = UsageCounter()
    proxy = UsageProxy(0, 1, counter, upstream_host="127.0.0.1")
    proxy.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as info:
            post_chat(proxy.port, {})
    finally:
        proxy.stop()

    assert info.value.code == 502
    assert counter.snapshot() == (None, None)


PRICING = {"input": 2, "output": 10}
PAID_REQUEST = {"messages": [{"role": "user", "content": "x" * 300}], "max_tokens": 1000}
WORST_CASE_LOW = 1000 * 10 / 1e6


class ScriptedUpstream:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.calls += 1
                status, body = outer.responses.pop(0)
                if status is None:
                    self.connection.close()
                    return
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def paid_proxy():
    started = []

    def make(responses, pricing=PRICING, upstream_port=None):
        upstream = ScriptedUpstream(responses)
        counter = UsageCounter(pricing)
        proxy = UsageProxy(0, upstream_port or upstream.port, counter, upstream_host="127.0.0.1")
        proxy.start()
        started.append((upstream, proxy))
        return upstream, proxy, counter

    yield make
    for upstream, proxy in started:
        proxy.stop()
        upstream.close()


def _usage_body(prompt=10, completion=5):
    return json.dumps({"usage": {"prompt_tokens": prompt, "completion_tokens": completion}}).encode()


@pytest.mark.parametrize("failure", [(500, b'{"error": "boom"}'), (None, b"")])
def test_upstream_failure_is_charged_at_the_worst_case_and_logged(paid_proxy, capsys, failure):
    _, proxy, counter = paid_proxy([failure])

    with pytest.raises(urllib.error.HTTPError) as caught:
        post_chat(proxy.port, PAID_REQUEST)

    assert caught.value.code == 502
    assert WORST_CASE_LOW < counter.cost_usd() < WORST_CASE_LOW * 1.05
    assert counter.total_cost_usd() == counter.cost_usd()
    assert counter.snapshot() == (None, None)
    assert "worst-case" in capsys.readouterr().out


def test_unreachable_upstream_is_not_charged(paid_proxy):
    _, proxy, counter = paid_proxy([], upstream_port=1)

    with pytest.raises(urllib.error.HTTPError):
        post_chat(proxy.port, PAID_REQUEST)

    assert counter.total_cost_usd() == 0.0


def test_response_without_usage_is_charged_at_the_worst_case(paid_proxy, capsys):
    _, proxy, counter = paid_proxy([(200, b'{"choices": []}')])

    post_chat(proxy.port, PAID_REQUEST)

    assert WORST_CASE_LOW < counter.cost_usd() < WORST_CASE_LOW * 1.05
    assert "no usage" in capsys.readouterr().out


def test_response_with_usage_is_charged_exactly(paid_proxy):
    _, proxy, counter = paid_proxy([(200, _usage_body(10, 5))])

    post_chat(proxy.port, PAID_REQUEST)

    assert counter.cost_usd() == pytest.approx((10 * 2 + 5 * 10) / 1e6)


def test_the_worst_case_uses_the_default_cap_when_the_request_names_none(paid_proxy):
    _, proxy, counter = paid_proxy([(200, b"{}")])

    post_chat(proxy.port, {"messages": [{"role": "user", "content": "hi"}]})

    assert counter.cost_usd() == pytest.approx(WORST_CASE_DEFAULT, rel=0.01)


WORST_CASE_DEFAULT = 8192 * 10 / 1e6


def test_unpriced_serving_is_never_charged(paid_proxy):
    _, proxy, counter = paid_proxy([(500, b"{}"), (200, b"{}")], pricing=None)

    with pytest.raises(urllib.error.HTTPError):
        post_chat(proxy.port, PAID_REQUEST)
    post_chat(proxy.port, PAID_REQUEST)

    assert counter.cost_usd() is None
    assert counter.total_cost_usd() == 0.0


def test_streaming_is_refused_for_paid_models_and_passed_through_for_free_ones(paid_proxy):
    upstream, proxy, counter = paid_proxy([])

    with pytest.raises(urllib.error.HTTPError) as caught:
        post_chat(proxy.port, {**PAID_REQUEST, "stream": True})

    body = json.loads(caught.value.read())
    assert caught.value.code == 400
    assert "streaming" in body["error"]["message"]
    assert upstream.calls == 0
    assert counter.total_cost_usd() == 0.0

    free_upstream, free_proxy, _ = paid_proxy([(200, b"data: x")], pricing=None)
    assert post_chat(free_proxy.port, {**PAID_REQUEST, "stream": True}) == b"data: x"


def test_calls_past_the_spend_limit_get_a_429_without_reaching_upstream(paid_proxy):
    upstream, proxy, counter = paid_proxy([(200, _usage_body(0, 10000)), (200, _usage_body())])
    counter.set_limit(0.05)

    post_chat(proxy.port, PAID_REQUEST)
    with pytest.raises(urllib.error.HTTPError) as caught:
        post_chat(proxy.port, PAID_REQUEST)

    body = json.loads(caught.value.read())
    assert caught.value.code == 429
    assert body["error"]["type"] == "budget_exceeded"
    assert "spend limit" in body["error"]["message"]
    assert upstream.calls == 1
    assert counter.total_cost_usd() == pytest.approx(0.1)


def test_calls_below_the_spend_limit_go_through(paid_proxy):
    upstream, proxy, counter = paid_proxy([(200, _usage_body()), (200, _usage_body())])
    counter.set_limit(0.05)

    post_chat(proxy.port, PAID_REQUEST)
    post_chat(proxy.port, PAID_REQUEST)

    assert upstream.calls == 2
