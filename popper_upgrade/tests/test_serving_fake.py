import json

from popper_upgrade.models import serving_for
from popper_upgrade.serving.fake import USAGE, FakeVllmServer
from popper_upgrade.tests.fakes import VLLM_MODEL, post_chat


def _free_port_model():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return {**VLLM_MODEL, "port": port + 1}


def test_fake_backend_answers_through_the_real_proxy_chain(monkeypatch):
    monkeypatch.setenv("POPPER_FAKE_BACKEND", "1")
    model = _free_port_model()

    with serving_for(model) as counter:
        body = json.loads(post_chat(counter.port, {"model": model["served_model_id"], "messages": []}))
        assert body["choices"][0]["message"]["content"]
        assert counter.snapshot() == (USAGE["prompt_tokens"], USAGE["completion_tokens"])


def test_fake_server_stops_cleanly():
    server = FakeVllmServer("fake-model", 0, host="127.0.0.1")
    server.port = 0

    assert server.start() is True
    server.stop()
    server.stop()
