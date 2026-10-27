import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from popper_upgrade.serving.pricing import cost_usd, usage_parts, worst_case_cost

LOCAL_HOST = "127.0.0.1"


class UsageCounter:
    def __init__(self, pricing=None):
        self._lock = threading.Lock()
        self._prompt = None
        self._completion = None
        self._cache_read = None
        self._cache_write = None
        self._cost = None
        self._total_cost = 0.0
        self.pricing = pricing
        self.port = None
        self.error_log = None
        self.alive = None
        self._limit = None

    def add(self, usage):
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        parts = usage_parts(usage)
        call_cost = cost_usd(parts, self.pricing)
        with self._lock:
            if isinstance(prompt, int):
                self._prompt = (self._prompt or 0) + prompt
            if isinstance(completion, int):
                self._completion = (self._completion or 0) + completion
            if parts is not None:
                self._cache_read = (self._cache_read or 0) + parts["cache_read"]
                self._cache_write = (self._cache_write or 0) + parts["cache_write"]
            if call_cost is not None:
                self._cost = (self._cost or 0.0) + call_cost
                self._total_cost += call_cost

    def _charge(self, amount):
        with self._lock:
            self._cost = (self._cost or 0.0) + amount
            self._total_cost += amount

    def charge_worst_case(self, request_body, reason):
        amount = worst_case_cost(request_body, self.pricing)
        if amount is None:
            return None
        self._charge(amount)
        print(f"  [proxy] charged worst-case USD {amount:.4f}: {reason}")
        return amount

    def record(self, usage, request_body):
        if self.pricing and usage_parts(usage) is None:
            self.charge_worst_case(request_body, "response carried no usage")
            return
        self.add(usage)

    def set_limit(self, limit_usd):
        with self._lock:
            self._limit = limit_usd

    def over_limit(self):
        with self._lock:
            return self._limit is not None and self._total_cost >= self._limit

    def limit_message(self):
        with self._lock:
            return (
                f"spend limit reached: USD {self._total_cost:.4f} of USD {self._limit:.4f} for this model; "
                "no further calls are allowed"
            )

    def snapshot(self):
        with self._lock:
            return self._prompt, self._completion

    def cache_snapshot(self):
        with self._lock:
            return self._cache_read, self._cache_write

    def cost_usd(self):
        with self._lock:
            return self._cost

    def total_cost_usd(self):
        with self._lock:
            return self._total_cost

    def reset(self):
        with self._lock:
            self._prompt = None
            self._completion = None
            self._cache_read = None
            self._cache_write = None
            self._cost = None
        if self.error_log is not None:
            self.error_log.mark()

    def upstream_error(self):
        return self.error_log.last() if self.error_log is not None else None


def read_usage(body):
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data.get("usage") if isinstance(data, dict) else None


def error_body(status, message, kind="api_error"):
    return json.dumps({"error": {"message": message, "type": kind, "code": status}}).encode()


def is_streaming(body):
    try:
        data = json.loads(body)
    except (TypeError, ValueError):
        return False
    return isinstance(data, dict) and data.get("stream") is True


def reached_provider(error):
    return not isinstance(getattr(error, "reason", None), ConnectionRefusedError | socket.gaierror)


STREAMING_REFUSAL = "streaming is not supported for paid models because its usage cannot be metered"


def _make_handler(upstream, counter, rewrite_request, rewrite_response, headers):
    class _RelayHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _relay(self, method, body):
            req = urllib.request.Request(f"{upstream}{self.path}", data=body, method=method)
            content_type = self.headers.get("Content-Type")
            if body is not None:
                req.add_header("Content-Type", content_type or "application/json")
            for name, value in headers.items():
                req.add_header(name, value)
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    resp_body = r.read()
                    status = r.status
                    resp_type = r.headers.get("Content-Type", "application/json")
            except Exception as e:
                print(f"  [proxy] {method} {self.path} failed: {e}")
                if body is not None and reached_provider(e):
                    counter.charge_worst_case(body, f"upstream failed: {e}")
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            counter.record(read_usage(resp_body), body)
            resp_body = rewrite_response(resp_body)
            self.send_response(status)
            self.send_header("Content-Type", resp_type)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def do_GET(self):
            self._relay("GET", None)

        def _refuse(self, status, message, kind):
            payload = error_body(status, message, kind)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if counter.over_limit():
                self._refuse(429, counter.limit_message(), "budget_exceeded")
                return
            if counter.pricing and is_streaming(body):
                self._refuse(400, STREAMING_REFUSAL, "invalid_request_error")
                return
            self._relay("POST", rewrite_request(self.path, body))

    return _RelayHandler


def identity_request(path, body):
    return body


def identity_response(body):
    return body


class RelayProxy:
    def __init__(self, listen_port, upstream, counter, rewrite_request=identity_request,
                 rewrite_response=identity_response, headers=None):
        self.counter = counter
        self._headers = dict(headers or {})
        self.port = listen_port
        self._upstream = upstream
        self._rewrite_request = rewrite_request
        self._rewrite_response = rewrite_response
        self._server = None
        self._thread = None

    def _build_handler(self):
        return _make_handler(
            self._upstream, self.counter, self._rewrite_request, self._rewrite_response, self._headers
        )

    def start(self):
        handler = self._build_handler()
        self._server = HTTPServer((LOCAL_HOST, self.port), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None


class UsageProxy(RelayProxy):
    def __init__(
        self, listen_port, upstream_port, counter, upstream_host="localhost", upstream_url=None, **relay_options
    ):
        upstream = upstream_url or f"http://{upstream_host}:{upstream_port}"
        super().__init__(listen_port, upstream, counter, **relay_options)
