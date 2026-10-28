import contextlib
import ctypes
import io
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cluster import fetch, preflight
from cluster.common import Interrupted, _signals
from popper_upgrade.models import serving_for
from popper_upgrade.serving import logs
from popper_upgrade.serving.vllm import revision_args

CANARY_JOB = "qwen-large"
CANARY_NAME = "qwen-canary"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
GPU_UTIL_ENV = "POPPER_CANARY_GPU_UTIL"
MAX_LEN_ENV = "POPPER_CANARY_MAX_MODEL_LEN"
EXTRA_ARGS_ENV = "POPPER_CANARY_EXTRA_ARGS"
DEFAULT_GPU_UTIL = "0.30"
DEFAULT_MAX_LEN = 8192
SERVE_TIMEOUT_S = 1500
REQUEST_TIMEOUT_S = 600
MIN_LONG_PROMPT_TOKENS = 2048
CHUNK_TOKENS = 2048
LONG_RECORDS = 260
EXCERPT_LINES = 30
EXCERPT_CHARS = 5000
MAX_SUMMARY_LINES = 10
REPORT_FILE = "canary.json"
LOG_DIR = "canary"
PASS_LINE = "PASS — run: singularity run --nv popper_qwen.sif"
FAIL_LINE = "FAIL — send canary.json to Vladimir; run popper_upgrade.sif llama-anchor instead"
INFORMATIONAL = ("clean_stop",)
ORDER = ("driver", "gpu", "software", "download", "server_start", "long_prompt", "short_chat", "clean_stop")
KERNEL_ERROR = re.compile(
    r"triton|flashinfer|ptxas|nvcc|nvrtc|\bPTX\b|no kernel image|CUDA error|cudaError|illegal memory|"
    r"gated.?delta|flash.?attn|\bfla\b|\bjit\b",
    re.IGNORECASE,
)
EXPLANATIONS = {
    "kernel": "A GPU kernel that vLLM compiles when it starts failed on this machine (see canary.json).",
    "memory": "vLLM ran out of GPU memory; another program may be using the GPU.",
    "download": "This is a download problem, not a GPU problem: the GPU itself was not tested.",
}
ROOT_CAUSE = re.compile(r"[\w.]*(?:Error|Exception|OutOfResources): ")
MEMORY_ERROR = re.compile(r"out of memory|\bOOM\b|Not enough memory|Free memory on device", re.IGNORECASE)
VERSION_PROBE = (
    "import importlib.metadata as m, json\n"
    "def v(n):\n"
    "    try:\n"
    "        return m.version(n)\n"
    "    except m.PackageNotFoundError:\n"
    "        return None\n"
    "print(json.dumps({n: v(n) for n in ('vllm', 'torch', 'triton', 'flashinfer-python')}))\n"
)


class Deadline(BaseException):
    pass


def _say(message):
    print(f"[canary] {message}", file=sys.stderr, flush=True)


def _fake():
    return bool(os.environ.get("POPPER_FAKE_BACKEND"))


def _smi(query):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip().splitlines()[0].strip()


def cuda_driver_api():
    try:
        version = ctypes.c_int()
        if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version)) == 0:
            return f"{version.value // 1000}.{version.value % 1000 // 10}"
    except (OSError, AttributeError):
        pass
    try:
        text = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", text)
    return match.group(1) if match else None


def versions():
    python = os.environ.get("POPPER_VLLM_PYTHON") or sys.executable
    out = subprocess.run([python, "-c", VERSION_PROBE], capture_output=True, text=True, timeout=120, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _driver_floor():
    floor = preflight.DRIVER_FLOORS.get(os.environ.get(preflight.IMAGE_ENV))
    return floor[1] if floor else None


def _canary_row():
    from cluster.job import load_job

    job = load_job(CANARY_JOB)
    row = preflight.model_row(job)
    util = os.environ.get(GPU_UTIL_ENV) or DEFAULT_GPU_UTIL
    max_len = int(os.environ.get(MAX_LEN_ENV) or DEFAULT_MAX_LEN)
    extra = shlex.split(os.environ.get(EXTRA_ARGS_ENV) or "")
    dropped = ("weights_gb", "kv_kb_per_token")
    serving = {key: value for key, value in (row.get("serving") or {}).items() if key not in dropped}
    serving["vllm_args"] = [
        *serving.get("vllm_args", []),
        *revision_args(REVISION),
        "--gpu-memory-utilization",
        util,
        "--max-num-batched-tokens",
        str(CHUNK_TOKENS),
        *extra,
    ]
    return {
        **row,
        "name": CANARY_NAME,
        "served_model_id": MODEL_ID,
        "max_model_len": max_len,
        "min_disk_gb": None,
        "revision": None,
        "serving": serving,
    }, {"gpu_memory_utilization": util, "max_model_len": max_len, "extra_args": extra}


def long_prompt():
    records = "\n".join(f"Record {i}: the sample value was {(i * 7) % 97} units." for i in range(1, LONG_RECORDS + 1))
    return f"{records}\n\nHow many records were listed above? Answer in one short sentence."


def _post(port, payload):
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return json.load(response)


def _message(body):
    choices = body.get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _is_info(line):
    return line.startswith("INFO") or " INFO " in line[:80]


def _last_error(lines, text):
    for line in reversed(lines):
        if not _is_info(line) and (logs.MESSAGE.search(line) or KERNEL_ERROR.search(line)):
            return line.strip()[: logs.MAX_DETAIL]
    return logs.last_error_line(text)


def _root_cause(lines):
    for line in lines:
        if not _is_info(line) and ROOT_CAUSE.search(line):
            return line.strip()[: logs.MAX_DETAIL]
    return None


def log_excerpt(path):
    text = logs.read_tail(path, 256 * 1024).decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "root_cause": _root_cause(lines),
        "last_error": _last_error(lines, text),
        "tail": "\n".join(lines[-EXCERPT_LINES:])[-EXCERPT_CHARS:],
    }


def classify(excerpt):
    quiet = [line for line in (excerpt.get("tail") or "").splitlines() if not _is_info(line)]
    blob = "\n".join([excerpt.get("root_cause") or "", excerpt.get("last_error") or "", *quiet])
    if MEMORY_ERROR.search(blob):
        return "memory"
    if KERNEL_ERROR.search(blob):
        return "kernel"
    return "other"


def _vllm_pids():
    proc = Path("/proc")
    if not proc.is_dir():
        return set()
    found = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except OSError:
            continue
        if "vllm" in cmdline.lower():
            found.add(int(entry.name))
    return found


def _reap(pids):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signum)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _vllm_pids() & pids:
            time.sleep(0.5)
        pids = _vllm_pids() & pids
        if not pids:
            return


@contextlib.contextmanager
def _deadline(seconds):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def fire(signum, frame):
        raise Deadline

    previous = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Report:
    def __init__(self):
        self.checks = []
        self.facts = {}
        self.model = {"id": MODEL_ID, "revision": REVISION}
        self.launch = {}
        self.failure = None
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.lap = time.monotonic()

    def add(self, name, ok, detail, **extra):
        now = time.monotonic()
        self.checks.append({"name": name, "ok": ok, "detail": detail, "seconds": round(now - self.lap, 1), **extra})
        self.lap = now

    def passed(self, name):
        return any(check["name"] == name and check["ok"] for check in self.checks)

    def recorded(self, name):
        return any(check["name"] == name for check in self.checks)

    def healthy(self):
        return all(check["ok"] for check in self.checks if check["name"] not in INFORMATIONAL)

    def finish(self):
        for name in ORDER:
            if not self.recorded(name):
                self.add(name, False, "not tried because an earlier check failed", skipped=True)
        ordered = sorted(self.checks, key=lambda check: ORDER.index(check["name"]))
        required = [check for check in ordered if check["name"] not in INFORMATIONAL]
        return {
            "verdict": "PASS" if all(check["ok"] for check in required) else "FAIL",
            "fake_backend": _fake(),
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": round(time.monotonic() - self.started, 1),
            "facts": self.facts,
            "model": self.model,
            "launch": self.launch,
            "checks": ordered,
            "failure": self.failure,
        }


def _hardware(report):
    if _fake():
        for name in ("driver", "gpu", "software"):
            report.add(name, True, "skipped (fake backend)")
        return
    _say("reading the GPU, driver and software versions")
    try:
        driver = preflight._driver_version()
        report.facts["driver"] = driver
        floor = _driver_floor()
        if floor is not None and preflight._version_tuple(driver) < floor:
            minimum = ".".join(str(part) for part in floor)
            report.add("driver", False, f"driver {driver} is older than the {minimum} this image needs")
        else:
            report.add("driver", True, f"driver {driver}")
    except Exception as e:
        report.add("driver", False, f"the GPU driver version could not be read ({type(e).__name__}: {e})")
        return
    report.facts["cuda_driver_api"] = cuda_driver_api()
    try:
        name, vram = preflight._gpu_query()
        capability = _smi("compute_cap")
        report.facts["gpu"] = {"name": name, "vram_gb": round(vram, 1), "compute_capability": capability}
        report.add("gpu", True, f"{name}, {vram:.0f} GB, compute capability {capability}")
    except Exception as e:
        report.add("gpu", False, f"no NVIDIA GPU could be read ({type(e).__name__}: {e})")
        return
    try:
        report.facts["software"] = versions()
        found = report.facts["software"]
        ok = bool(found.get("vllm") and found.get("torch"))
        detail = f"vLLM {found.get('vllm')}, torch {found.get('torch')}, triton {found.get('triton')}"
        report.add("software", ok, detail)
    except Exception as e:
        report.add("software", False, f"the vLLM and torch versions could not be read ({type(e).__name__}: {e})")


def _download(report):
    if _fake():
        report.add("download", True, "skipped (fake backend)")
        return
    _say(f"downloading {MODEL_ID} (about 1.75 GB, skipped when already present)")
    try:
        path = fetch.download_model(MODEL_ID, REVISION, say=_say)
        report.model["path"] = str(path)
        report.model["size_gb"] = round(fetch.folder_size_gb(path), 2)
    except Exception as e:
        report.failure = {"category": "download", "message": str(e)}
        report.add("download", False, f"the model could not be downloaded: {e}")
        return
    report.add("download", True, f"{MODEL_ID} revision {REVISION[:7]} ({report.model['size_gb']} GB)")


def _first_missing(report):
    return next(name for name in ("server_start", "long_prompt", "short_chat") if not report.recorded(name))


def _fail_server(report, name, detail, log_path):
    excerpt = log_excerpt(log_path)
    category = classify(excerpt)
    report.failure = {"category": category, "check": name, "message": detail, "vllm_log": excerpt}
    cause = excerpt.get("root_cause")
    report.add(name, False, f"{detail} ({cause})" if cause else detail, category=category)


def _exercise(report, row, log_dir):
    log_path = log_dir / logs.VLLM_LOG
    fake = _fake()
    began = time.monotonic()
    with serving_for(row, log_dir) as counter:
        port = counter.port or row["port"]
        report.add("server_start", True, f"vLLM was ready after {time.monotonic() - began:.0f} s")
        _say("vLLM is up; sending a long prompt")
        try:
            body = _post(
                port,
                {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": long_prompt()}],
                    "max_tokens": 16,
                },
            )
            usage = body.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or 0
            generated = usage.get("completion_tokens") or 0
            if not _message(body).get("content") and not generated:
                raise ValueError("the server answered without any text")
            if not fake and prompt_tokens < MIN_LONG_PROMPT_TOKENS:
                raise ValueError(f"the prompt counted only {prompt_tokens} tokens, not over {MIN_LONG_PROMPT_TOKENS}")
            report.add("long_prompt", True, f"{prompt_tokens} prompt tokens read, {generated} generated")
        except Exception as e:
            _fail_server(report, "long_prompt", f"the long prompt failed ({type(e).__name__}: {e})", log_path)
            return
        _say("sending a short chat with the qwen-large settings")
        try:
            body = _post(
                port,
                {
                    "model": MODEL_ID,
                    "messages": [{"role": "user", "content": "In one short sentence, what is 12 + 30?"}],
                    "max_tokens": 64,
                },
            )
            message = _message(body)
            content = message.get("content") or ""
            if not content.strip():
                raise ValueError("the server answered without any text")
            leaked = "<think>" in content or "</think>" in content
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            report.model["short_reply"] = content[:200]
            if leaked or reasoning:
                report.add("short_chat", False, "the model still produced thinking text although thinking is off")
                report.failure = {"category": "thinking", "check": "short_chat", "message": content[:500]}
                return
            report.add("short_chat", True, "answered, no thinking text")
        except Exception as e:
            _fail_server(report, "short_chat", f"the short chat failed ({type(e).__name__}: {e})", log_path)


def _serve(report, row, effective, workdir):
    log_dir = Path(workdir) / LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in (logs.VLLM_LOG, logs.PROXY_LOG):
        (log_dir / name).unlink(missing_ok=True)
    report.launch = {"effective": effective, "log_dir": str(log_dir)}
    before = _vllm_pids()
    printed = io.StringIO()
    interrupted = None
    _say("starting vLLM (first start compiles GPU kernels and can take several minutes)")
    began = time.monotonic()
    with _signals() as guard:
        guard.active = True
        try:
            with _deadline(SERVE_TIMEOUT_S), contextlib.redirect_stdout(printed):
                _exercise(report, row, log_dir)
        except Deadline:
            detail = f"timed out after {SERVE_TIMEOUT_S // 60} minutes"
            _fail_server(report, _first_missing(report), detail, log_dir / logs.VLLM_LOG)
        except Exception as e:
            _fail_server(report, _first_missing(report), str(e), log_dir / logs.VLLM_LOG)
        except Interrupted:
            interrupted = 128 + guard.signum
            report.failure = {"category": "interrupted", "message": f"stopped by signal {guard.signum}"}
        finally:
            guard.active = False
    report.facts["serve_seconds"] = round(time.monotonic() - began, 1)
    command = next((line for line in printed.getvalue().splitlines() if line.startswith("Starting vLLM:")), None)
    report.launch["command"] = command.removeprefix("Starting vLLM: ") if command else None
    leftover = _vllm_pids() - before
    if leftover:
        _reap(leftover)
        report.add("clean_stop", False, f"{len(leftover)} vLLM process(es) were left behind and had to be killed")
    else:
        report.add("clean_stop", True, "vLLM and its helper processes are gone")
    return interrupted


def differences(effective):
    extra = []
    if effective["extra_args"]:
        extra = [f"extra arguments from {EXTRA_ARGS_ENV}: {' '.join(effective['extra_args'])}"]
    return [
        *extra,
        f"model id {MODEL_ID} instead of the qwen-large model, pinned to revision {REVISION}",
        f"--max-model-len {effective['max_model_len']} instead of 16384",
        f"--gpu-memory-utilization {effective['gpu_memory_utilization']} instead of the 0.90/0.95 default",
        f"--max-num-batched-tokens {CHUNK_TOKENS}, so the long prompt is prefilled in chunks",
        "no memory-fit check (the 0.8B model needs about 2 GB)",
    ]


def summary(document, workdir):
    checks = {check["name"]: check for check in document["checks"]}
    lines = []
    facts = document["facts"]
    if document["fake_backend"]:
        lines.append("Fake backend: no GPU was used; this only tests the canary program itself.")
    else:
        gpu = facts.get("gpu")
        if gpu:
            lines.append(
                f"Machine: {gpu['name']}, {gpu['vram_gb']:g} GB, compute capability {gpu['compute_capability']}, "
                f"driver {facts.get('driver')}, CUDA driver API {facts.get('cuda_driver_api')}."
            )
        for name in ("driver", "gpu"):
            if not gpu and not checks[name]["ok"] and not checks[name].get("skipped"):
                lines.append(f"Machine: FAILED - {checks[name]['detail']}")
                break
        software = facts.get("software") or {}
        if software and checks["software"]["ok"]:
            lines.append(f"Software: vLLM {software.get('vllm')}, torch {software.get('torch')}.")
        elif not checks["software"]["ok"] and not checks["software"].get("skipped"):
            lines.append(f"Software: FAILED - {checks['software']['detail']}")
    labels = {
        "download": "Model download",
        "server_start": "vLLM start-up (kernel compilation)",
        "long_prompt": "Long prompt and decode",
        "short_chat": "Short chat, thinking off",
        "clean_stop": "Clean shutdown",
    }
    for name, label in labels.items():
        check = checks[name]
        if check.get("skipped"):
            continue
        lines.append(f"{label}: {'OK' if check['ok'] else 'FAILED'} - {check['detail']}"[:300])
    failure = document["failure"] or {}
    if document["verdict"] == "FAIL" and failure.get("category") in EXPLANATIONS:
        lines.append(EXPLANATIONS[failure["category"]])
    lines.append(f"Details: {Path(workdir).resolve() / REPORT_FILE}")
    lines = lines[: MAX_SUMMARY_LINES - 1]
    lines.append(PASS_LINE if document["verdict"] == "PASS" else FAIL_LINE)
    return lines


def run_canary(workdir="."):
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    report = Report()
    interrupted = None
    try:
        row, effective = _canary_row()
        report.launch = {"effective": effective}
    except Exception as e:
        report.add("software", False, f"the canary settings could not be built ({type(e).__name__}: {e})")
        row = None
    if row is not None:
        _hardware(report)
        if report.healthy():
            _download(report)
        if report.healthy():
            interrupted = _serve(report, row, effective, workdir)
        report.launch["differences_from_qwen_large"] = differences(effective)
    document = report.finish()
    (workdir / REPORT_FILE).write_text(json.dumps(document, indent=2), encoding="utf-8")
    print("\n".join(summary(document, workdir)))
    if interrupted is not None:
        return interrupted
    return 0 if document["verdict"] == "PASS" else 1
