import json as _json
import os
import signal
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


BASELINE_QUANTIZATION = "awq"
BASELINE_DTYPE = "float16"
LARGE_GPU_GB = 70
LARGE_CONTEXT = 16384
SMALL_CONTEXT = 6144
MIN_CONTEXT = SMALL_CONTEXT
OVERHEAD_GIB = 3.0
CONTEXT_STEP = 1024
NO_FLAG = "none"
REVISION_FLAGS = ("--revision", "--tokenizer-revision")
DEFAULT_FALLBACK = "run popper_upgrade.sif llama-anchor instead"
STOP_GRACE_S = 60
PROC_DIR = "/proc"


class LaunchError(ValueError):
    pass


def revision_args(revision):
    return [item for flag in REVISION_FLAGS for item in (flag, revision)]


def has_launch_overrides(model):
    serving = model.get("serving") or {}
    return any(model.get(key) is not None for key in ("quantization", "dtype", "max_model_len", "revision")) or bool(
        serving.get("vllm_args")
    )


def plan_launch(model, vram_gb):
    large = vram_gb >= LARGE_GPU_GB
    gpu_util = 0.90 if large else 0.95
    context = model.get("max_model_len") or (LARGE_CONTEXT if large else SMALL_CONTEXT)
    serving = model.get("serving") or {}
    weights = serving.get("weights_gb")
    kv = serving.get("kv_kb_per_token")
    if weights is not None and kv:
        room = vram_gb * gpu_util - weights - OVERHEAD_GIB
        fit = int(room * 1024 * 1024 // kv) // CONTEXT_STEP * CONTEXT_STEP
        if fit < MIN_CONTEXT:
            fallback = serving.get("fallback") or DEFAULT_FALLBACK
            raise LaunchError(
                f"{model['name']} needs about {weights + OVERHEAD_GIB:.0f} GB of GPU memory for its weights plus room "
                f"for a {MIN_CONTEXT}-token context, but this GPU has {vram_gb:.0f} GB. Please {fallback}."
            )
        context = min(context, fit)
    quantization = model.get("quantization") or BASELINE_QUANTIZATION
    return {
        "max_model_len": context,
        "gpu_util": gpu_util,
        "quantization": None if quantization == NO_FLAG else quantization,
        "dtype": model.get("dtype") or BASELINE_DTYPE,
        "args": [
            *(revision_args(model["revision"]) if model.get("revision") else []),
            *(str(arg) for arg in serving.get("vllm_args") or []),
        ],
    }


def detect_vram_gb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False
        )
        return int(result.stdout.strip().splitlines()[0].strip()) / 1024
    except Exception:
        return 40


class ModelVllmServer(VllmServer):
    def __init__(self, model, port, host="localhost", proxy_port=None, launch=None, log_path=None):
        super().__init__(model, port, host, proxy_port)
        self.launch = launch
        self.log_path = log_path

    def command(self, plan):
        python = os.environ.get("POPPER_VLLM_PYTHON") or sys.executable
        cmd = [
            python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--host", self.host,
            "--dtype", plan["dtype"],
            "--max-model-len", str(plan["max_model_len"]),
            "--gpu-memory-utilization", f"{plan['gpu_util']:.2f}",
        ]
        if plan["quantization"]:
            cmd += ["--quantization", plan["quantization"]]
        return cmd + ["--enforce-eager", *plan["args"]]

    def _spawn(self, cmd, env):
        if self.log_path is None:
            return subprocess.Popen(cmd, env=env, start_new_session=True)
        with open(self.log_path, "ab") as log:
            return subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )

    def start(self):
        env = os.environ.copy()
        env["HF_TOKEN"] = HF_TOKEN
        env["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        env.pop("PYTHONPATH", None)
        try:
            plan = plan_launch(self.launch, detect_vram_gb())
        except LaunchError as e:
            print(f"ERROR: {e}")
            return False
        cmd = self.command(plan)
        print(f"Starting vLLM: {' '.join(cmd)}")
        self.vllm_proc = self._spawn(cmd, env)
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

    def _signal_group(self, signum):
        try:
            os.killpg(self.vllm_proc.pid, signum)
        except (ProcessLookupError, PermissionError):
            pass

    def _group_alive(self):
        group = self.vllm_proc.pid
        if not os.path.isdir(PROC_DIR):
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True
        for entry in os.listdir(PROC_DIR):
            if not entry.isdigit():
                continue
            try:
                with open(f"{PROC_DIR}/{entry}/stat", encoding="utf-8", errors="replace") as handle:
                    fields = handle.read().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if fields[0] != "Z" and fields[2] == str(group):
                return True
        return False

    def stop(self):
        if self.vllm_proc is None:
            return
        if self.vllm_proc.poll() is None or self._group_alive():
            print("\nStopping vLLM...")
            self._signal_group(signal.SIGTERM)
            deadline = time.monotonic() + STOP_GRACE_S
            while time.monotonic() < deadline and (self.vllm_proc.poll() is None or self._group_alive()):
                time.sleep(0.2)
            if self.vllm_proc.poll() is None or self._group_alive():
                self._signal_group(signal.SIGKILL)
            self.vllm_proc.wait()
            print("vLLM stopped.")
