import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from popper_upgrade.config import ANTHROPIC_DEFAULT_MAX_TOKENS
from popper_upgrade.serving.usage import RelayProxy

CHAT_PATH = "/v1/chat/completions"
MESSAGES_PATH = "/v1/messages"
CACHE_CONTROL = {"type": "ephemeral"}
SYSTEM_ROLES = ("system", "developer")
CHAT_ROLES = ("user", "assistant")
PASSTHROUGH_KEYS = ("thinking", "metadata", "top_k", "top_p")
MAX_TEMPERATURE = 1.0
HTTP_OK = 200
FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _texts(content):
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        raise ValueError("message content must be text or a list of text parts")
    texts = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text") or "")
        else:
            kind = part.get("type") if isinstance(part, dict) else type(part).__name__
            raise ValueError(f"unsupported content part {kind}")
    return texts


def _blocks(content):
    return [{"type": "text", "text": text} for text in _texts(content) if text.strip()]


def _mark(blocks):
    if blocks:
        blocks[-1] = {**blocks[-1], "cache_control": CACHE_CONTROL}


def _split_messages(raw):
    system = []
    messages = []
    for message in raw or []:
        role = message.get("role")
        blocks = _blocks(message.get("content"))
        if role in SYSTEM_ROLES:
            system += blocks
        elif role in CHAT_ROLES:
            if blocks:
                messages.append({"role": role, "content": blocks})
        else:
            raise ValueError(f"unsupported message role {role}")
    return system, messages


def _check_supported(data, messages):
    if not messages:
        raise ValueError("a request needs at least one user or assistant message")
    if not data.get("model"):
        raise ValueError("a request needs a model")
    if data.get("n") not in (None, 1):
        raise ValueError("n above 1 is not supported")
    if data.get("stream"):
        raise ValueError("streaming is not supported")


def to_messages_request(data):
    system, messages = _split_messages(data.get("messages"))
    _check_supported(data, messages)
    _mark(system)
    if len(messages) > 1:
        _mark(messages[-2]["content"])
    request = {
        "model": data["model"],
        "max_tokens": data.get("max_tokens") or data.get("max_completion_tokens") or ANTHROPIC_DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if system:
        request["system"] = system
    stop = data.get("stop")
    if stop:
        request["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if data.get("temperature") is not None:
        request["temperature"] = min(float(data["temperature"]), MAX_TEMPERATURE)
    for key in PASSTHROUGH_KEYS:
        if data.get(key) is not None:
            request[key] = data[key]
    return request


def _count(usage, key):
    value = usage.get(key)
    return value if isinstance(value, int) else 0


def to_chat_usage(usage):
    fresh = _count(usage, "input_tokens")
    read = _count(usage, "cache_read_input_tokens")
    write = _count(usage, "cache_creation_input_tokens")
    output = _count(usage, "output_tokens")
    prompt = fresh + read + write
    return {
        "prompt_tokens": prompt,
        "completion_tokens": output,
        "total_tokens": prompt + output,
        "input_tokens": fresh,
        "cache_creation_input_tokens": write,
        "cache_read_input_tokens": read,
        "prompt_tokens_details": {"cached_tokens": read},
    }


def to_chat_response(data):
    text = "".join(block.get("text", "") for block in data.get("content") or [] if block.get("type") == "text")
    response = {
        "id": data.get("id"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": FINISH_REASONS.get(data.get("stop_reason"), "stop"),
            }
        ],
    }
    if isinstance(data.get("usage"), dict):
        response["usage"] = to_chat_usage(data["usage"])
    return response


def error_message(status, body):
    try:
        error = json.loads(body).get("error") or {}
        return {"message": error.get("message") or f"HTTP {status}", "type": error.get("type") or "api_error"}
    except (ValueError, AttributeError):
        return {"message": f"HTTP {status}", "type": "api_error"}


def error_body(status, message, kind="api_error"):
    return json.dumps({"error": {"message": message, "type": kind, "code": status}}).encode()


def _make_handler(upstream, counter, rewrite_request, headers):
    class _AnthropicHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _call(self, method, path, body):
            request = urllib.request.Request(f"{upstream}{path}", data=body, method=method)
            request.add_header("Content-Type", "application/json")
            for name, value in headers.items():
                request.add_header(name, value)
            try:
                with urllib.request.urlopen(request, timeout=300) as r:
                    return r.status, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
            except Exception as e:
                print(f"  [proxy] {method} {path} failed: {e}")
                return None, None

        def _relay(self, method, body):
            status, resp = self._call(method, self.path, body)
            if status is None:
                self._send(502, error_body(502, "upstream unreachable"))
            else:
                self._send(status, resp)

        def do_GET(self):
            self._relay("GET", None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path != CHAT_PATH:
                self._relay("POST", rewrite_request(self.path, body))
                return
            try:
                request = to_messages_request(json.loads(rewrite_request(self.path, body)))
            except (ValueError, TypeError, AttributeError) as e:
                print(f"  [proxy] request not translatable: {e}")
                self._send(400, error_body(400, f"request not translatable: {e}", "invalid_request_error"))
                return
            status, resp = self._call("POST", MESSAGES_PATH, json.dumps(request).encode())
            if status is None:
                self._send(502, error_body(502, "upstream unreachable"))
                return
            if status != HTTP_OK:
                error = error_message(status, resp)
                print(f"  [proxy] POST {MESSAGES_PATH} -> {status}: {error['message']}")
                self._send(status, error_body(status, error["message"], error["type"]))
                return
            try:
                chat = to_chat_response(json.loads(resp))
            except (ValueError, AttributeError) as e:
                print(f"  [proxy] response not translatable: {e}")
                self._send(502, error_body(502, "upstream response not translatable"))
                return
            counter.add(chat.get("usage"))
            self._send(HTTP_OK, json.dumps(chat).encode())

    return _AnthropicHandler


class AnthropicProxy(RelayProxy):
    def _build_handler(self):
        return _make_handler(self._upstream, self.counter, self._rewrite_request, self._headers)
