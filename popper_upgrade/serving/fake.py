import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from popper_upgrade.serving.vllm import VllmServer

ANSWER = "The fake serving backend has no analysis to offer."
USAGE = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}


class _FakeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "fake", "object": "model"}]})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        self._send(
            {
                "id": "fake-1",
                "object": "chat.completion",
                "model": request.get("model", "fake"),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": ANSWER}, "finish_reason": "stop"}
                ],
                "usage": USAGE,
            }
        )


class FakeVllmServer(VllmServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fake = None

    def start(self):
        self._fake = HTTPServer((self.host, self.port), _FakeHandler)
        threading.Thread(target=self._fake.serve_forever, daemon=True).start()
        return True

    def stop(self):
        if self._fake is not None:
            self._fake.shutdown()
            self._fake.server_close()
            self._fake = None
