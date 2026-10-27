import contextlib
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

from popper_upgrade.config import ANTHROPIC_VERSION
from popper_upgrade.serving import ollama_proxy
from popper_upgrade.serving.anthropic import AnthropicProxy
from popper_upgrade.serving.fake import FakeVllmServer
from popper_upgrade.serving.logs import PROXY_LOG, VLLM_LOG, LogTrimmer, UpstreamErrors, capture_proxy_log
from popper_upgrade.serving.ollama_proxy import DEFAULT_UPSTREAM, OllamaProxy
from popper_upgrade.serving.rewrite import request_rewriter
from popper_upgrade.serving.usage import UsageCounter, UsageProxy, identity_request, identity_response
from popper_upgrade.serving.vllm import ModelVllmServer, VllmServer, has_launch_overrides

LOCAL_BACKENDS = ("ollama", "vllm")
HOSTED_BACKENDS = ("anthropic", "openai")
COMPATIBLE_BACKEND = "openai_compatible"
ANTHROPIC_BACKEND = "anthropic_messages"
PROXIED_BACKENDS = (COMPATIBLE_BACKEND, ANTHROPIC_BACKEND)
POPPER_HOSTED_PREFIXES = ("claude-", "gpt-", "o1")


class ServingError(RuntimeError):
    pass


def _check_compatible(model, require_key=False):
    name = model["name"]
    base_url = model.get("base_url")
    if not base_url:
        raise ValueError(f"model {name} needs a base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"model {name} has an invalid base_url {base_url}")
    if model["served_model_id"].startswith(POPPER_HOSTED_PREFIXES):
        raise ValueError(f"model {name}: POPPER would bypass the local endpoint for id {model['served_model_id']}")
    key_env = model.get("api_key_env")
    if require_key and key_env and not os.environ.get(key_env):
        raise ValueError(f"model {name}: environment variable {key_env} is not set")


def popper_kwargs_for(model, port=None):
    backend = model["backend"]
    if backend in LOCAL_BACKENDS:
        return {
            "llm": model["served_model_id"],
            "is_locally_served": True,
            "server_port": port if port is not None else model["port"],
            "api_key": "EMPTY",
        }
    if backend in PROXIED_BACKENDS:
        _check_compatible(model)
        if port is None:
            raise ValueError(f"model {model['name']} is only reachable through a serving port")
        return {
            "llm": model["served_model_id"],
            "is_locally_served": True,
            "server_port": port,
            "api_key": "EMPTY",
        }
    if backend in HOSTED_BACKENDS:
        return {"llm": model["served_model_id"]}
    raise ValueError(f"unknown backend {backend} for model {model['name']}")


def check_environment(model, environment):
    if model["environment"] != environment:
        raise ValueError(f"model {model['name']} is not available on {environment}")
    if model["backend"] in PROXIED_BACKENDS:
        _check_compatible(model, require_key=True)


def _require_reachable(url, label="Ollama"):
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)), 2):
            pass
    except OSError as e:
        raise ServingError(f"{label} is not reachable at {url}: {e}") from e


@contextlib.contextmanager
def _serve_ollama(model):
    upstream = os.environ.get("POPPER_OLLAMA_UPSTREAM", DEFAULT_UPSTREAM)
    _require_reachable(upstream)
    counter = UsageCounter()
    proxy = OllamaProxy(model["port"], upstream, counter)
    proxy.start()
    counter.port = proxy.port
    try:
        yield counter
    finally:
        proxy.stop()


@contextlib.contextmanager
def _serve_vllm(model, log_dir=None):
    counter = UsageCounter()
    counter.port = model["port"]
    server_class = FakeVllmServer if os.environ.get("POPPER_FAKE_BACKEND") else VllmServer
    extra = {}
    overridden = has_launch_overrides(model)
    if (overridden or log_dir is not None) and server_class is VllmServer:
        server_class = ModelVllmServer
        extra["launch"] = model if overridden else {"name": model["name"]}
        if log_dir is not None:
            extra["log_path"] = Path(log_dir) / VLLM_LOG
    trimmer = None
    if log_dir is not None:
        trimmer = LogTrimmer([Path(log_dir) / VLLM_LOG, Path(log_dir) / PROXY_LOG])
        counter.error_log = UpstreamErrors(Path(log_dir) / VLLM_LOG, trimmer)
    server = server_class(model["served_model_id"], model["port"] - 1, proxy_port=model["port"] + 1, **extra)
    rewrite = request_rewriter(model)
    proxy_options = {} if rewrite is identity_request else {"rewrite_request": rewrite}
    usage_proxy = UsageProxy(model["port"], model["port"] + 1, counter, **proxy_options)
    with _proxy_log(log_dir), _trimming(trimmer):
        try:
            if not server.start():
                raise ServingError(f"vLLM did not start for {model['name']}")
            server.start_proxy()
            usage_proxy.start()
            yield counter
        finally:
            server.stop()
            usage_proxy.stop()
            server.stop_proxy()


@contextlib.contextmanager
def _trimming(trimmer):
    if trimmer is None:
        yield
        return
    trimmer.start()
    try:
        yield
    finally:
        trimmer.stop()


def _proxy_log(log_dir):
    return contextlib.nullcontext() if log_dir is None else capture_proxy_log(Path(log_dir) / PROXY_LOG)


@contextlib.contextmanager
def _serve_openai_compatible(model):
    _check_compatible(model, require_key=True)
    base_url = model["base_url"]
    _require_reachable(base_url, f"Endpoint for {model['name']}")
    counter = UsageCounter(model.get("pricing"))
    key_env = model.get("api_key_env")
    headers = {"Authorization": f"Bearer {os.environ[key_env]}"} if key_env else {}
    rewrites = {"rewrite_request": identity_request, "rewrite_response": identity_response}
    if model.get("strip_tools") is not False:
        rewrites = {
            "rewrite_request": ollama_proxy.strip_tools,
            "rewrite_response": ollama_proxy.decode_arguments,
        }
    upstream = base_url.rstrip("/").removesuffix("/v1")
    rewrites["rewrite_request"] = request_rewriter(model, rewrites["rewrite_request"])
    proxy = UsageProxy(0, None, counter, upstream_url=upstream, headers=headers, **rewrites)
    proxy.start()
    counter.port = proxy.port
    try:
        yield counter
    finally:
        proxy.stop()


@contextlib.contextmanager
def _serve_anthropic_messages(model):
    _check_compatible(model, require_key=True)
    base_url = model["base_url"]
    _require_reachable(base_url, f"Endpoint for {model['name']}")
    counter = UsageCounter(model.get("pricing"))
    key_env = model.get("api_key_env")
    if not key_env:
        raise ValueError(f"model {model['name']} needs an api_key_env")
    headers = {"anthropic-version": ANTHROPIC_VERSION, "x-api-key": os.environ[key_env]}
    upstream = base_url.rstrip("/").removesuffix("/v1")
    proxy = AnthropicProxy(0, upstream, counter, rewrite_request=request_rewriter(model), headers=headers)
    proxy.start()
    counter.port = proxy.port
    try:
        yield counter
    finally:
        proxy.stop()


@contextlib.contextmanager
def _serve_hosted(model):
    yield UsageCounter()


def serving_for(model, log_dir=None):
    backend = model["backend"]
    if backend == "ollama":
        return _serve_ollama(model)
    if backend == "vllm":
        return _serve_vllm(model, log_dir)
    if backend == COMPATIBLE_BACKEND:
        return _serve_openai_compatible(model)
    if backend == ANTHROPIC_BACKEND:
        return _serve_anthropic_messages(model)
    if backend in HOSTED_BACKENDS:
        return _serve_hosted(model)
    raise ValueError(f"unknown backend {backend} for model {model['name']}")
