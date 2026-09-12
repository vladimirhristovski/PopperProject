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

import pandas as pd
from popper import Popper

from config import (
    ALPHA,
    DATA_DIR,
    HF_TOKEN,
    HYPOTHESES,
    LOCAL_HOST,
    LOCAL_MODEL,
    LOCAL_PORT,
    MAX_RETRY,
    MAX_TESTS,
    TIME_LIMIT,
)
from popper_common import check_data, determine_status, generate_report, parse_result

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

PROXY_PORT = LOCAL_PORT + 1


class _ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        url = f"http://{LOCAL_HOST}:{LOCAL_PORT}{self.path}"
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
        url = f"http://{LOCAL_HOST}:{LOCAL_PORT}{self.path}"

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


_proxy_server = None


def start_proxy():
    global _proxy_server
    _proxy_server = HTTPServer((LOCAL_HOST, PROXY_PORT), _ProxyHandler)
    t = threading.Thread(target=_proxy_server.serve_forever, daemon=True)
    t.start()
    print(f"Proxy started on port {PROXY_PORT} → vLLM on port {LOCAL_PORT}")


def stop_proxy():
    if _proxy_server:
        _proxy_server.shutdown()


vllm_proc = None


def start_vllm():
    global vllm_proc

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
        "--model", LOCAL_MODEL,
        "--port", str(LOCAL_PORT),
        "--host", LOCAL_HOST,
        "--dtype", "float16",
        "--max-model-len", max_model_len,
        "--gpu-memory-utilization", gpu_util,
        "--quantization", "awq",
        "--enforce-eager",
    ]

    print(f"Starting vLLM: {' '.join(cmd)}")
    vllm_proc = subprocess.Popen(cmd, env=env)

    print("Waiting for vLLM to be ready...")
    for _ in range(600):
        if vllm_proc.poll() is not None:
            print("ERROR: vLLM process exited unexpectedly. Check output above.")
            return False
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            if s.connect_ex((LOCAL_HOST, LOCAL_PORT)) == 0:
                s.close()
                print("vLLM is ready!")
                return True
            s.close()
        except Exception:
            pass
        time.sleep(2)

    print("ERROR: vLLM did not start within 20 minutes.")
    return False


def stop_vllm():
    if vllm_proc and vllm_proc.poll() is None:
        print("\nStopping vLLM...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
        print("vLLM stopped.")


def _handle_signal(sig, frame):
    stop_proxy()
    stop_vllm()
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


def run(results_file="results_local.csv"):
    print("=" * 70)
    print(f"POPPER — Test 3 vLLM: ({LOCAL_MODEL})")
    print("=" * 70)

    missing = check_data()
    if missing:
        print(f"ERROR: Missing data files: {missing}")
        print("Run: python3 prepare_data.py")
        return

    try:
        if not start_vllm():
            return

        start_proxy()
        os.environ["OPENAI_API_KEY"] = "vllm"
        os.environ["OPENAI_BASE_URL"] = f"http://{LOCAL_HOST}:{PROXY_PORT}/v1"

        results = []
        total_start = time.time()

        print(f"\nRunning {len(HYPOTHESES)} hypotheses\n")
        print("=" * 70)

        for i, hypothesis in enumerate(HYPOTHESES, 1):
            print(f"\n[{i}/{len(HYPOTHESES)}] {hypothesis[:65]}...")
            print("  Initializing POPPER agent...")

            try:
                agent = Popper(llm=LOCAL_MODEL, is_locally_served=True, server_port=PROXY_PORT,
                               plot_agent_architecture=False, domain="social science")
                agent.register_data(data_path=DATA_DIR, loader_type="custom")
                agent.configure(
                    alpha=ALPHA,
                    max_num_of_tests=MAX_TESTS,
                    max_retry=MAX_RETRY,
                    time_limit=TIME_LIMIT,
                    aggregate_test="E-value",
                    relevance_checker=True,
                    use_react_agent=True,
                )
            except Exception as e:
                print(f"  ERROR: Failed to initialize agent — {e}")
                results.append({"model": LOCAL_MODEL, "hypothesis": hypothesis,
                                "status": "ERROR", "e_value": 0.0,
                                "decision": "init_error", "time_min": 0.0})
                continue

            start = time.time()
            try:
                result = agent.validate(hypothesis=hypothesis)
                elapsed = (time.time() - start) / 60
                e_value, decision = parse_result(result)
                status = determine_status(e_value, decision)

                print(f"  Status  : {status}")
                print(f"  E-value : {e_value:.4f}")
                print(f"  Decision: {decision}")
                print(f"  Time    : {elapsed:.1f} min")

                results.append({"model": LOCAL_MODEL, "hypothesis": hypothesis,
                                "status": status, "e_value": e_value,
                                "decision": decision, "time_min": elapsed})

            except Exception as e:
                elapsed = (time.time() - start) / 60
                print(f"  ERROR: {str(e)[:120]}")
                results.append({"model": LOCAL_MODEL, "hypothesis": hypothesis,
                                "status": "ERROR", "e_value": 0.0,
                                "decision": "error", "time_min": elapsed})

        total_time = (time.time() - total_start) / 60

        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        print(f"\nCSV saved to: {results_file}")

        generate_report(results, total_time, results_file, LOCAL_MODEL)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            print(f"{r['status']:15} | E={r['e_value']:7.4f} | {r['time_min']:5.1f}m | {r['hypothesis'][:40]}")
        print("=" * 70)
        print(f"Total time : {total_time:.1f} min ({total_time / 60:.2f} hours)")
        print(f"Supported  : {sum(1 for r in results if r['status'] == 'SUPPORTED')}")
        print(f"Not Supp.  : {sum(1 for r in results if r['status'] == 'NOT SUPPORTED')}")
        print(f"Errors     : {sum(1 for r in results if r['status'] == 'ERROR')}")
        print("=" * 70)

    finally:
        stop_proxy()
        stop_vllm()


if __name__ == "__main__":
    run()
