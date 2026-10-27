import json
import os
import urllib.parse
import urllib.request

from popper_upgrade.config import ANTHROPIC_VERSION
from popper_upgrade.models import ANTHROPIC_BACKEND

MARKERS = ("deprecat", "shutdown", "sunset", "retire", "end_of_life", "eol")
PAGE_LIMIT = 1000
MAX_PAGES = 10
TIMEOUT = 30


def upstream_model_id(model):
    overrides = (model.get("request_overrides") or {}).get("set") or {}
    return overrides.get("model") or model["served_model_id"].split("/")[-1]


def _headers(model):
    key = os.environ[model["api_key_env"]]
    if model["backend"] == ANTHROPIC_BACKEND:
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
    return {"Authorization": f"Bearer {key}"}


def fetch_models(model, opener=urllib.request.urlopen):
    base = model["base_url"].rstrip("/") + "/models"
    paged = model["backend"] == ANTHROPIC_BACKEND
    entries = []
    after = None
    for _ in range(MAX_PAGES):
        url = base
        if paged:
            query = {"limit": PAGE_LIMIT, **({"after_id": after} if after else {})}
            url = f"{base}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, method="GET", headers=_headers(model))
        with opener(request, timeout=TIMEOUT) as response:
            payload = json.loads(response.read())
        entries.extend(item for item in payload.get("data") or [] if isinstance(item, dict))
        if not paged or not payload.get("has_more") or not payload.get("last_id"):
            break
        after = payload["last_id"]
    return entries


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from _flatten(inner, f"{prefix}{key}.")
    else:
        yield prefix.rstrip("."), value


def lifecycle_notes(entry):
    return {
        key: value
        for key, value in _flatten(entry)
        if any(marker in key.lower() for marker in MARKERS) and value not in (None, False, "", [])
    }


def check_model_status(model, opener=urllib.request.urlopen):
    name = model["name"]
    key_env = model.get("api_key_env")
    if not model.get("base_url") or not key_env or not os.environ.get(key_env):
        return {"model": name, "state": "skipped", "notes": {}, "detail": "no key or endpoint to check with"}
    wanted = upstream_model_id(model)
    try:
        entries = fetch_models(model, opener)
    except (OSError, ValueError) as e:
        return {"model": name, "state": "unavailable", "notes": {}, "detail": f"models endpoint failed: {e}"}
    entry = next((item for item in entries if item.get("id") == wanted), None)
    if entry is None:
        return {"model": name, "state": "not_listed", "notes": {}, "detail": f"{wanted} is not in the models list"}
    return {"model": name, "state": "listed", "notes": lifecycle_notes(entry), "detail": ""}


def format_status(status):
    name, state = status["model"], status["state"]
    if state == "listed":
        if status["notes"]:
            notes = ", ".join(f"{key}={value}" for key, value in status["notes"].items())
            return f"[model status] {name}: listed, lifecycle info: {notes}"
        return f"[model status] {name}: listed, no shutdown or deprecation info exposed"
    return f"[model status] {name}: {state.replace('_', ' ')}: {status['detail']}"
