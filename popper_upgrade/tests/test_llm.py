import os

from popper_upgrade.llm import ensure_local_llm_api_key


def test_sets_api_key_when_port_given(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    ensure_local_llm_api_key(11434, api_key="EMPTY")

    assert os.environ["OPENAI_API_KEY"] == "EMPTY"


def test_does_not_touch_env_when_port_is_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    ensure_local_llm_api_key(None, api_key="EMPTY")

    assert "OPENAI_API_KEY" not in os.environ


def test_does_not_override_an_existing_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "real-cloud-key")

    ensure_local_llm_api_key(11434, api_key="EMPTY")

    assert os.environ["OPENAI_API_KEY"] == "real-cloud-key"
