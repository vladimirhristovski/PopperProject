import json
import threading
import urllib.error

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
