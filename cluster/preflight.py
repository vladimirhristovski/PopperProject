import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from cluster import fetch
from cluster.common import LOCAL_HOST, load_baseline
from cluster.fetch import hf_home
from popper_upgrade.models import ServingError, check_environment, serving_for
from popper_upgrade.serving import logs
from popper_upgrade.serving.vllm import DEFAULT_FALLBACK, LaunchError, plan_launch
from popper_upgrade.skills.embedding import Embedder

GB = 1024**3
ANSWER_TIMEOUT = 180
MODEL_CHECK = "model_answers"
DRIVER_CHECK = "driver"
IMAGE_ENV = "POPPER_IMAGE"
DRIVER_FLOORS = {
    "modern": ("popper_modern.sif", (575, 51)),
    "qwen": ("popper_qwen.sif", (525, 60)),
}
TOKEN_ENVS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _result(name, ok, detail):
    return {"name": name, "ok": ok, "detail": detail}


def read_snapshot(job):
    return json.loads((Path(job["dir"]) / job["snapshot"]).read_text(encoding="utf-8"))


def model_row(job):
    row = next((m for m in read_snapshot(job)["models"] if m["name"] == job["model"]), None)
    if row is None:
        raise KeyError(f"model {job['model']} is not in the job's snapshot")
    return row


def _gpu_query():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    name, memory_mb = out.stdout.strip().splitlines()[0].rsplit(",", 1)
    return name.strip(), int(memory_mb) / 1024


def _driver_version():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return out.stdout.strip().splitlines()[0].strip()


def _version_tuple(text):
    return tuple(int(part) for part in text.split(".")[:2])


def check_driver(job):
    floor = DRIVER_FLOORS.get(os.environ.get(IMAGE_ENV))
    try:
        version = _driver_version()
        found = _version_tuple(version)
    except Exception as e:
        if floor:
            detail = f"the GPU driver version could not be read ({type(e).__name__}); {DEFAULT_FALLBACK}"
            return _result(DRIVER_CHECK, False, detail)
        return _result(DRIVER_CHECK, True, "driver version not readable; not needed for this image")
    if not floor:
        return _result(DRIVER_CHECK, True, f"driver {version}")
    image, minimum_version = floor
    minimum = ".".join(str(part) for part in minimum_version)
    if found < minimum_version:
        detail = (
            f"driver too old: this machine has {version} but {image} needs {minimum} or newer; "
            f"{DEFAULT_FALLBACK}"
        )
        return _result(DRIVER_CHECK, False, detail)
    return _result(DRIVER_CHECK, True, f"driver {version} (needs {minimum} or newer)")


def check_gpu_memory(job):
    model = model_row(job)
    if model["backend"] != "vllm":
        return _result("memory", True, "no local GPU model in this job")
    try:
        name, vram_gb = _gpu_query()
    except Exception:
        return _result("memory", True, "not checked: no GPU is visible")
    try:
        plan = plan_launch(model, vram_gb)
    except LaunchError as e:
        return _result("memory", False, str(e))
    detail = f"{model['name']} fits {vram_gb:.0f} GB with a {plan['max_model_len']}-token context"
    return _result("memory", True, detail)


def _model_cached(served_model_id, revision=None):
    return fetch.is_cached(served_model_id, revision)


def check_hf_token(job):
    model = model_row(job)
    serving = model.get("serving") or {}
    if not serving.get("gated"):
        return _result("token", True, "no Hugging Face token is needed for this model")
    has_token = any(os.environ.get(name) for name in TOKEN_ENVS)
    if has_token or _model_cached(model["served_model_id"], model.get("revision")):
        return _result("token", True, f"access to {model['served_model_id']} is available")
    fallback = serving.get("fallback") or DEFAULT_FALLBACK
    detail = f"{model['served_model_id']} needs a Hugging Face token (HF_TOKEN) and none is set; please {fallback}"
    return _result("token", False, detail)


def check_gpu(job):
    try:
        name, vram_gb = _gpu_query()
    except Exception as e:
        return _result("gpu", False, f"no NVIDIA GPU is visible ({type(e).__name__}); start the job with --nv")
    return _result("gpu", True, f"{name}, {vram_gb:.0f} GB")


def _existing_parent(path):
    path = Path(path).resolve()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_disk(job):
    model = model_row(job)
    required = model.get("min_disk_gb")
    target = hf_home()
    free = shutil.disk_usage(_existing_parent(target)).free / GB
    if model["backend"] == "vllm" and fetch.weights_cached(model["served_model_id"], model.get("revision")):
        detail = f"{free:.0f} GB free in {target}; the weights of {model['served_model_id']} are already there"
        return _result("disk", True, detail)
    if required is None:
        return _result("disk", True, f"{free:.0f} GB free in {target}; no minimum is set for this model")
    if free < required:
        detail = f"the model weights need {required} GB free in {target} but only {free:.0f} GB is free"
        return _result("disk", False, detail)
    return _result("disk", True, f"{free:.0f} GB free in {target}, {required} GB needed")


def prefetch_model(job, say=print):
    model = model_row(job)
    if model["backend"] != "vllm" or os.environ.get("POPPER_FAKE_BACKEND"):
        return None
    try:
        fetch.download_model(model["served_model_id"], model.get("revision"), say=say)
    except fetch.DownloadError as e:
        return str(e)
    return None


def check_download(job):
    model = model_row(job)
    if model["backend"] != "vllm" or os.environ.get("POPPER_FAKE_BACKEND"):
        return _result("download", True, "no model download is needed for this job")
    problem = prefetch_model(job)
    if problem is not None:
        return _result("download", False, problem)
    return _result("download", True, f"{model['served_model_id']} is in {hf_home()}")


def check_data(job):
    config, popper_common = load_baseline()
    missing = popper_common.check_data()
    spec = json.loads((Path(job["dir"]) / job["spec"]).read_text(encoding="utf-8"))
    source = spec["source"]
    for part in source["parts"] if source.get("kind") == "combined" else [source]:
        if part.get("kind") == "discoverybench":
            path = Path(config.DATA_DIR) / part["path"]
            if not path.exists():
                missing.append(str(path))
    if missing:
        return _result("data", False, f"missing data files: {', '.join(missing)}")
    return _result("data", True, f"data found in {config.DATA_DIR}")


def check_snapshot(job):
    from popper_upgrade.db.snapshot import SnapshotSkillSource

    source = SnapshotSkillSource(Path(job["dir"]) / job["snapshot"], Embedder())
    return _result(
        "snapshot", True, f"{len(source.list_skills(True))} approved skills, embedding model {source.embedding_model}"
    )


def _ask(port, served_model_id):
    payload = json.dumps(
        {"model": served_model_id, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 1}
    ).encode()
    request = urllib.request.Request(
        f"http://{LOCAL_HOST}:{port}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=ANSWER_TIMEOUT) as response:
        return json.load(response)


def _log_size(workdir):
    if workdir is None:
        return 0
    try:
        return (Path(workdir) / logs.VLLM_LOG).stat().st_size
    except OSError:
        return 0


def _new_log_error(workdir, start):
    if workdir is None:
        return None
    text = logs.read_tail(Path(workdir) / logs.VLLM_LOG, start=start).decode("utf-8", errors="replace")
    return logs.last_error_line(text)


def check_model_answers(job, workdir=None):
    model = model_row(job)
    check_environment(model, "cluster")
    start = _log_size(workdir)
    try:
        with serving_for(model, workdir) as counter:
            body = _ask(counter.port or model["port"], model["served_model_id"])
            if not body.get("choices"):
                return _result(MODEL_CHECK, False, "the model server answered without any text")
            prompt, completion = counter.snapshot()
    except ServingError as e:
        line = _new_log_error(workdir, start)
        if line:
            raise ServingError(f"{e} (last line of the vLLM log: {line})") from e
        raise
    if prompt is None or completion is None:
        return _result(MODEL_CHECK, False, "the model server answered but does not report token usage")
    return _result(MODEL_CHECK, True, f"{model['name']} answered ({prompt} prompt tokens, {completion} generated)")


CHECKS = (
    (DRIVER_CHECK, "check_driver"),
    ("gpu", "check_gpu"),
    ("memory", "check_gpu_memory"),
    ("token", "check_hf_token"),
    ("disk", "check_disk"),
    ("data", "check_data"),
    ("snapshot", "check_snapshot"),
    ("download", "check_download"),
    (MODEL_CHECK, "check_model_answers"),
)


def _run_check(name, function, *args):
    try:
        return function(*args)
    except Exception as e:
        return _result(name, False, f"{type(e).__name__}: {e}")


def run_preflight(job, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    checks = []
    for name, function_name in CHECKS:
        if checks and not checks[0]["ok"] and checks[0]["name"] == DRIVER_CHECK:
            checks.append(_result(name, False, "not tried because the driver is too old"))
            continue
        if name == MODEL_CHECK and not all(check["ok"] for check in checks):
            checks.append(_result(name, False, "not tried because an earlier check failed"))
            continue
        arguments = (job, workdir) if name == MODEL_CHECK else (job,)
        checks.append(_run_check(name, globals()[function_name], *arguments))
    result = {"ok": all(check["ok"] for check in checks), "checks": checks}
    (workdir / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
