import json

import pytest

from popper_upgrade.serving.model_status import check_model_status, format_status, upstream_model_id
from popper_upgrade.tests.test_models import _registry_models


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeOpener:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return FakeResponse(page)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")


def test_upstream_id_comes_from_the_request_overrides():
    models = _registry_models()

    assert upstream_model_id(models["gpt-6-sol"]) == "gpt-6-sol"
    assert upstream_model_id(models["claude-sonnet-5-5"]) == "claude-sonnet-5-5"
    assert upstream_model_id({"served_model_id": "vendor/some-model"}) == "some-model"


def test_openai_models_are_listed_with_a_bearer_key_and_deprecation_fields_are_reported(keys):
    opener = FakeOpener(
        [
            {
                "data": [
                    {"id": "gpt-other"},
                    {"id": "gpt-6-sol", "deprecation": {"shutdown_date": "2027-03-01", "replacement": None}},
                ]
            }
        ]
    )

    status = check_model_status(_registry_models()["gpt-6-sol"], opener)

    request = opener.requests[0]
    assert request.full_url == "https://api.openai.com/v1/models"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer sk-openai"
    assert status["state"] == "listed"
    assert status["notes"] == {"deprecation.shutdown_date": "2027-03-01"}
    assert "deprecation.shutdown_date=2027-03-01" in format_status(status)


def test_anthropic_models_are_paged_with_native_headers(keys):
    opener = FakeOpener(
        [
            {"data": [{"id": "claude-other"}], "has_more": True, "last_id": "claude-other"},
            {"data": [{"id": "claude-sonnet-5-5", "display_name": "Sonnet 5.5"}], "has_more": False},
        ]
    )

    status = check_model_status(_registry_models()["claude-sonnet-5-5"], opener)

    first, second = opener.requests
    assert first.full_url == "https://api.anthropic.com/v1/models?limit=1000"
    assert second.full_url == "https://api.anthropic.com/v1/models?limit=1000&after_id=claude-other"
    assert first.get_header("X-api-key") == "sk-anthropic"
    assert first.get_header("Anthropic-version") == "2023-06-01"
    assert status == {"model": "claude-sonnet-5-5", "state": "listed", "notes": {}, "detail": ""}
    assert "no shutdown or deprecation info exposed" in format_status(status)


def test_unlisted_model_is_reported(keys):
    status = check_model_status(_registry_models()["gpt-6-sol"], FakeOpener([{"data": [{"id": "gpt-other"}]}]))

    assert status["state"] == "not_listed"
    assert "gpt-6-sol is not in the models list" in format_status(status)


def test_endpoint_failure_is_reported_not_raised(keys):
    for failure in (OSError("connection refused"), ValueError("bad json")):
        status = check_model_status(_registry_models()["gpt-6-sol"], FakeOpener([failure]))

        assert status["state"] == "unavailable"
        assert str(failure) in format_status(status)


def test_without_a_key_nothing_is_called(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    opener = FakeOpener([])

    status = check_model_status(_registry_models()["gpt-6-sol"], opener)

    assert status["state"] == "skipped"
    assert opener.requests == []
