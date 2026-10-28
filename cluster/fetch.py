import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

HUB_TIMEOUT_S = 15
DEFAULT_ENDPOINT = "https://huggingface.co"
GB = 1024**3
SERVING_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py"]
WEIGHT_INDEX = "model.safetensors.index.json"
SINGLE_WEIGHTS = "model.safetensors"
CONFIG = "config.json"


class DownloadError(Exception):
    pass


def hf_home():
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def _model_dir(model_id):
    return hf_home() / "hub" / f"models--{model_id.replace('/', '--')}"


def is_cached(model_id, revision=None):
    if revision:
        return (_model_dir(model_id) / "snapshots" / revision).is_dir()
    return (_model_dir(model_id) / "refs" / "main").is_file()


def _snapshot_dir(model_id, revision=None):
    if revision:
        return _model_dir(model_id) / "snapshots" / revision
    try:
        sha = (_model_dir(model_id) / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return _model_dir(model_id) / "snapshots" / sha if sha else None


def weights_cached(model_id, revision=None):
    if not is_cached(model_id, revision):
        return False
    snapshot = _snapshot_dir(model_id, revision)
    if snapshot is None or not (snapshot / CONFIG).exists():
        return False
    index = snapshot / WEIGHT_INDEX
    if not index.is_file():
        return (snapshot / SINGLE_WEIGHTS).exists()
    try:
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return False
    return bool(shards) and all((snapshot / name).exists() for name in shards)


def hub_endpoint():
    return os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT


def hub_unreachable():
    request = urllib.request.Request(hub_endpoint(), method="HEAD")
    try:
        urllib.request.urlopen(request, timeout=HUB_TIMEOUT_S).close()
    except urllib.error.HTTPError:
        return None
    except (OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def folder_size_gb(path):
    total = sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
    return total / GB


def download_model(model_id, revision=None, say=print):
    problem = hub_unreachable()
    if problem is not None:
        if is_cached(model_id, revision):
            say(f"Hugging Face is not reachable ({problem}); using the copy of {model_id} already in {hf_home()}.")
            return _model_dir(model_id)
        raise DownloadError(
            f"cannot reach Hugging Face at {hub_endpoint()} ({problem}); "
            f"the server needs internet access to it, or {model_id} must already be in {hf_home()}"
        )
    say(f"Downloading {model_id} into {hf_home()} (files already there are skipped)...")
    try:
        path = snapshot_download(
            model_id, revision=revision, cache_dir=str(hf_home() / "hub"), allow_patterns=SERVING_PATTERNS
        )
    except Exception as e:
        raise DownloadError(f"downloading {model_id} failed ({type(e).__name__}: {e})") from e
    say(f"{model_id} is ready in {path}.")
    return Path(path)
