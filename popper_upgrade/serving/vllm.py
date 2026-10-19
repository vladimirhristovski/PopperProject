import json as _json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request as _req
from http.server import BaseHTTPRequestHandler, HTTPServer

HF_TOKEN = os.environ.get("HF_TOKEN", "")


def make_proxy_handler(host, port):
    class _ProxyHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            url = f"http://{host}:{port}{self.path}"
            try:
                with _req.urlopen(url, timeout=10) as r:
                    body = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                print(f"  [proxy] GET {self.path} failed: {e}")
                self.send_response(502)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            url = f"http://{host}:{port}{self.path}"

            try:
                req_data = _json.loads(body)
                req_data.pop("tools", None)
                req_data.pop("tool_choice", None)
                body = _json.dumps(req_data).encode()
            except Exception as e:
                print(f"  [proxy] failed to strip tools from request: {e}")

            req = _req.Request(url, data=body, method="POST")
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))

            try:
                with _req.urlopen(req, timeout=300) as r:
                    resp_body = r.read()
                    status = r.status
                    ct = r.headers.get("Content-Type", "application/json")
            except Exception as e:
                print(f"  [proxy] POST {self.path} failed: {e}")
                self.send_response(502)
                self.end_headers()
                return

            try:
                data = _json.loads(resp_body)
                changed = False
                for choice in data.get("choices", []):
                    msg = choice.get("message") or choice.get("delta") or {}
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        args = fn.get("arguments")
                        if isinstance(args, str):
                            try:
                                fn["arguments"] = _json.loads(args)
                                changed = True
                            except Exception as e:
                                print(f"  [proxy] failed to decode tool_call arguments: {e}")
                if changed:
                    resp_body = _json.dumps(data).encode()
            except Exception as e:
                print(f"  [proxy] failed to post-process response: {e}")

            self.send_response(status)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", len(resp_body))
            self.end_headers()
            self.wfile.write(resp_body)

    return _ProxyHandler


class VllmServer:
    def __init__(self, model, port, host="localhost", proxy_port=None):
        self.model = model
        self.port = port
        self.host = host
        self.proxy_port = port + 1 if proxy_port is None else proxy_port
        self.proxy_handler = make_proxy_handler(host, port)
        self._proxy_server = None
        self.vllm_proc = None

    def start_proxy(self):
        self._proxy_server = HTTPServer((self.host, self.proxy_port), self.proxy_handler)
        t = threading.Thread(target=self._proxy_server.serve_forever, daemon=True)
        t.start()
        print(f"Proxy started on port {self.proxy_port} → vLLM on port {self.port}")

    def stop_proxy(self):
        if self._proxy_server:
            self._proxy_server.shutdown()

    def start(self):
        env = os.environ.copy()
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        env.pop("PYTHONPATH", None)

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False
            )
            vram_mb = int(result.stdout.strip().split("\n")[0].strip())
            vram_gb = vram_mb / 1024
            print(f"Detected GPU VRAM: {vram_gb:.1f} GB")
        except Exception:
            vram_gb = 40
            print(f"Could not detect GPU VRAM, assuming {vram_gb}GB")

        if vram_gb >= 70:
            max_model_len = "16384"
            gpu_util = "0.90"
            print("80GB GPU detected — using max-model-len=16384")
        else:
            max_model_len = "6144"
            gpu_util = "0.95"
            print("40GB GPU detected — using max-model-len=6144")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--host", self.host,
            "--dtype", "float16",
            "--max-model-len", max_model_len,
            "--gpu-memory-utilization", gpu_util,
            "--quantization", "awq",
            "--enforce-eager",
        ]

        print(f"Starting vLLM: {' '.join(cmd)}")
        self.vllm_proc = subprocess.Popen(cmd, env=env)

        print("Waiting for vLLM to be ready...")
        for _ in range(600):
            if self.vllm_proc.poll() is not None:
                print("ERROR: vLLM process exited unexpectedly. Check output above.")
                return False
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                if s.connect_ex((self.host, self.port)) == 0:
                    s.close()
                    print("vLLM is ready!")
                    return True
                s.close()
            except Exception:
                pass
            time.sleep(2)

        print("ERROR: vLLM did not start within 20 minutes.")
        return False

    def stop(self):
        if self.vllm_proc and self.vllm_proc.poll() is None:
            print("\nStopping vLLM...")
            self.vllm_proc.terminate()
            try:
                self.vllm_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.vllm_proc.kill()
            print("vLLM stopped.")
