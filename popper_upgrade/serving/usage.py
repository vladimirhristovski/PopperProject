import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

LOCAL_HOST = "127.0.0.1"


class UsageCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._prompt = None
        self._completion = None
        self.port = None

    def add(self, usage):
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        with self._lock:
            if isinstance(prompt, int):
                self._prompt = (self._prompt or 0) + prompt
            if isinstance(completion, int):
                self._completion = (self._completion or 0) + completion

    def snapshot(self):
        with self._lock:
            return self._prompt, self._completion

    def reset(self):
        with self._lock:
            self._prompt = None
            self._completion = None


def read_usage(body):
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data.get("usage") if isinstance(data, dict) else None


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
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            counter.add(read_usage(resp_body))
            resp_body = rewrite_response(resp_body)
            self.send_response(status)
            self.send_header("Content-Type", resp_type)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def do_GET(self):
            self._relay("GET", None)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
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

    def start(self):
        handler = _make_handler(
            self._upstream, self.counter, self._rewrite_request, self._rewrite_response, self._headers
        )
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
