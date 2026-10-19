import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from cluster.common import LOCAL_HOST, load_baseline
from popper_upgrade.models import check_environment, serving_for
from popper_upgrade.skills.embedding import Embedder

GB = 1024**3
ANSWER_TIMEOUT = 180
MODEL_CHECK = "model_answers"


def _result(name, ok, detail):
    return {"name": name, "ok": ok, "detail": detail}


def read_snapshot(job):
    return json.loads((Path(job["dir"]) / job["snapshot"]).read_text(encoding="utf-8"))


def model_row(job):
    row = next((m for m in read_snapshot(job)["models"] if m["name"] == job["model"]), None)
    if row is None:
        raise KeyError(f"model {job['model']} is not in the job's snapshot")
    return row


def hf_home():
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


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
    required = model_row(job).get("min_disk_gb")
    target = hf_home()
    free = shutil.disk_usage(_existing_parent(target)).free / GB
    if required is None:
        return _result("disk", True, f"{free:.0f} GB free in {target}; no minimum is set for this model")
    if free < required:
        detail = f"the model weights need {required} GB free in {target} but only {free:.0f} GB is free"
        return _result("disk", False, detail)
    return _result("disk", True, f"{free:.0f} GB free in {target}, {required} GB needed")


def check_data(job):
    config, popper_common = load_baseline()
    missing = popper_common.check_data()
    spec = json.loads((Path(job["dir"]) / job["spec"]).read_text(encoding="utf-8"))
    source = spec["source"]
    if source.get("kind") == "discoverybench":
        path = Path(config.DATA_DIR) / source["path"]
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


def check_model_answers(job):
    model = model_row(job)
    check_environment(model, "cluster")
    with serving_for(model) as counter:
        body = _ask(counter.port or model["port"], model["served_model_id"])
        if not body.get("choices"):
            return _result(MODEL_CHECK, False, "the model server answered without any text")
        prompt, completion = counter.snapshot()
    if prompt is None or completion is None:
        return _result(MODEL_CHECK, False, "the model server answered but does not report token usage")
    return _result(MODEL_CHECK, True, f"{model['name']} answered ({prompt} prompt tokens, {completion} generated)")


CHECKS = (
    ("gpu", "check_gpu"),
    ("disk", "check_disk"),
    ("data", "check_data"),
    ("snapshot", "check_snapshot"),
    (MODEL_CHECK, "check_model_answers"),
)


def _run_check(name, function, job):
    try:
        return function(job)
    except Exception as e:
        return _result(name, False, f"{type(e).__name__}: {e}")


def run_preflight(job, workdir):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    checks = []
    for name, function_name in CHECKS:
        if name == MODEL_CHECK and not all(check["ok"] for check in checks):
            checks.append(_result(name, False, "not tried because an earlier check failed"))
            continue
        checks.append(_run_check(name, globals()[function_name], job))
    result = {"ok": all(check["ok"] for check in checks), "checks": checks}
    (workdir / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
