import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from cluster import canary as canary_module
from cluster import parts as parts_module
from cluster import preflight as preflight_module
from cluster import store
from cluster.budget import Budget, resolve_max_hours
from cluster.common import Interrupted, _signals, run_experiment_job
from popper_upgrade.eval.report import report_from_bundle, write_report
from popper_upgrade.serving import logs

JOBS_DIR = Path(os.environ.get("POPPER_JOBS_DIR", Path(__file__).resolve().parent / "jobs"))
GIT_SHA_FILE = Path(__file__).resolve().parent / "GIT_SHA"
OPERATOR = "Vladimir"
MAX_LISTED_JOBS = 6


class JobError(Exception):
    pass


def list_jobs():
    if not JOBS_DIR.is_dir():
        return []
    return sorted(path.name for path in JOBS_DIR.iterdir() if (path / "job.json").is_file())


def load_job(name):
    candidates = [JOBS_DIR / name, Path(name)]
    path = next((candidate for candidate in candidates if (candidate / "job.json").is_file()), None)
    if path is None:
        raise JobError(f"there is no job called {name}")
    job = json.loads((path / "job.json").read_text(encoding="utf-8"))
    return {**job, "dir": str(path.resolve())}


def _sif_path():
    for name in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        path = os.environ.get(name)
        if path and Path(path).is_file():
            return Path(path)
    return None


class _SifHash:
    def __init__(self):
        self.path = _sif_path()
        self.value = None
        self._thread = None
        if self.path is not None:
            self._thread = threading.Thread(target=self._compute, daemon=True)
            self._thread.start()

    def _compute(self):
        digest = hashlib.sha256()
        try:
            with self.path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 24), b""):
                    digest.update(chunk)
        except OSError:
            return
        self.value = digest.hexdigest()

    def result(self):
        if self._thread is not None:
            self._thread.join()
        return self.value


def _popper_commit():
    try:
        text = importlib.metadata.distribution("popper_agent").read_text("direct_url.json")
        return json.loads(text)["vcs_info"]["commit_id"]
    except Exception:
        return None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _vllm_version():
    python = os.environ.get("POPPER_VLLM_PYTHON")
    if not python:
        return _package_version("vllm")
    try:
        out = subprocess.run(
            [python, "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _model_revision(served_model_id):
    ref = preflight_module.hf_home() / "hub" / f"models--{served_model_id.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _gpu():
    try:
        name, vram_gb = preflight_module._gpu_query()
    except Exception:
        return None
    return {"name": name, "vram_gb": round(vram_gb, 1)}


def build_manifest(base, job, runs, sif, exit_code, extra=None):
    completed = [run for run in runs if run.get("status") == "completed"]
    served = next(
        (m["served_model_id"] for m in preflight_module.read_snapshot(job)["models"] if m["name"] == job["model"]),
        job["model"],
    )
    return {
        **base,
        "job": job["name"],
        "model": job["model"],
        "finished_at": store.now().isoformat(),
        "exit_code": exit_code,
        "sif_path": str(sif.path) if sif.path else None,
        "sif_sha256": sif.result(),
        "popper_sha": _popper_commit(),
        "vllm_version": _vllm_version(),
        "gpu": _gpu(),
        "model_revision": _model_revision(served),
        "n_runs": len(runs),
        "n_completed": len(completed),
        "n_errors": sum(1 for run in runs if run.get("status") == "error"),
        "runs_without_tokens": sum(1 for run in completed if run.get("tokens_in") is None),
        **(extra or {}),
    }


@dataclass
class Packed:
    archive: Path
    runs: list
    part: int
    progress: dict
    complete: bool


def _frozen_snapshot_id(bundle):
    path = Path(bundle) / parts_module.FREEZE_INFO
    return json.loads(path.read_text(encoding="utf-8")).get("snapshot_id") if path.exists() else None


def _library_snapshot_id(job):
    if not job.get("library_from"):
        return None
    path = Path(job["dir"]) / job["library_from"]
    return json.loads(path.read_text(encoding="utf-8")).get("snapshot_id")


def pack_archive(bundle, workdir, job, sif, exit_code, budget):
    workdir = Path(workdir)
    store.close_stale_runs(bundle)
    if not store.final_runs(bundle):
        return None
    previous = parts_module.read_parts(workdir)
    number = max((part["part"] for part in previous), default=0) + 1
    stem = f"popper_results_{job['name']}_part{number:02d}_{store.now().strftime('%Y%m%dT%H%M%SZ')}"
    scratch = Path(tempfile.mkdtemp(dir=workdir, prefix=".stage_"))
    stage = scratch / stem
    try:
        runs, candidate_ids = parts_module.stage_part(bundle, stage, previous)
        parts_module.stage_cumulative(bundle, scratch / "view")
        write_report(report_from_bundle(scratch / "view"), stage)
        progress = budget.progress()
        complete = bool(progress) and progress["total"] > 0 and progress["done"] >= progress["total"]
        manifest = json.loads((stage / parts_module.MANIFEST).read_text(encoding="utf-8"))
        extra = {
            "part": number,
            "progress": progress,
            "complete": complete,
            "max_hours": budget.max_hours,
            "budget_reached": budget.exhausted,
            "frozen_snapshot_id": _frozen_snapshot_id(bundle) or _library_snapshot_id(job),
        }
        manifest = build_manifest(manifest, job, runs, sif, exit_code, extra)
        (stage / parts_module.MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if (workdir / "preflight.json").exists():
            shutil.copyfile(workdir / "preflight.json", stage / "preflight.json")
        for name in (logs.VLLM_LOG, logs.PROXY_LOG):
            logs.copy_tail(workdir / name, stage / name)
        archive = workdir / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage, arcname=stem)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record = {
        "part": number,
        "name": archive.name,
        "created_at": store.now().isoformat(),
        "runs": [run["id"] for run in runs],
        "candidates": candidate_ids,
    }
    parts_module.write_parts(workdir, [*previous, record])
    return Packed(archive, runs, number, progress, complete)


def pack_diagnostics(workdir, job, sif, exit_code, budget):
    workdir = Path(workdir)
    stem = f"popper_diagnostics_{job['name']}_{store.now().strftime('%Y%m%dT%H%M%SZ')}"
    scratch = Path(tempfile.mkdtemp(dir=workdir, prefix=".stage_"))
    stage = scratch / stem
    stage.mkdir()
    try:
        last_error = logs.last_error_line(logs.read_tail(workdir / logs.VLLM_LOG).decode("utf-8", errors="replace"))
        extra = {"diagnostics_only": True, "max_hours": budget.max_hours, "last_vllm_log_error": last_error}
        manifest = build_manifest({}, job, [], sif, exit_code, extra)
        (stage / parts_module.MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if (workdir / "preflight.json").exists():
            shutil.copyfile(workdir / "preflight.json", stage / "preflight.json")
        for name in (logs.VLLM_LOG, logs.PROXY_LOG):
            logs.copy_tail(workdir / name, stage / name)
        archive = workdir / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage, arcname=stem)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return archive


def diagnostics_message(archive, workdir):
    lines = ["No run was recorded, so there are no results to send."]
    base = Path(workdir).resolve()
    present = [str(base / name) for name in (logs.VLLM_LOG, logs.PROXY_LOG) if (base / name).exists()]
    if present:
        lines.append("The model server logs are in: " + ", ".join(present))
    lines.append(f"Please send this file to {OPERATOR}: {Path(archive).resolve()}")
    return lines


def preflight_message(result, workdir):
    failed = [check for check in result["checks"] if not check["ok"] and not check["detail"].startswith("not tried")]
    lines = ["The machine check failed, so nothing was run."]
    lines += [f"- {check['name']}: {check['detail']}" for check in failed[:5]]
    lines.append(f"Please send this file to {OPERATOR}: {Path(workdir).resolve() / 'preflight.json'}")
    return lines


def _progress_line(progress):
    phases = []
    for phase, entry in progress["phases"].items():
        arms = ", ".join(f"{arm} {item['done']}/{item['total']}" for arm, item in entry["arms"].items())
        phases.append(f"{phase} {entry['done']}/{entry['total']} ({arms})")
    return f"Progress so far: {progress['done']} of {progress['total']} runs: " + "; ".join(phases) + "."


def finish_message(packed, exit_code, budget):
    runs = packed.runs
    completed = sum(1 for run in runs if run.get("status") == "completed")
    errors = sum(1 for run in runs if run.get("status") == "error")
    tonight = f"Part {packed.part}: {len(runs)} runs tonight ({completed} completed, {errors} ended in an error)."
    send = f"Please send this file to {OPERATOR}: {packed.archive.resolve()}"
    if exit_code != 0:
        lines = [
            f"The job was stopped early (exit code {exit_code}) after {len(runs)} runs tonight.",
            "Progress is saved: run the same command again in this directory to continue.",
        ]
    elif packed.complete:
        lines = [tonight, "Everything is finished."]
        send = f"ALL DONE — send this file to {OPERATOR}: {packed.archive.resolve()}"
    elif budget.exhausted:
        lines = [
            tonight,
            f"The {budget.max_hours:g}-hour limit was reached (budget reached): no new run was started,",
            "and none was cut off.",
        ]
    else:
        lines = [tonight, "Every run was tried, but some did not complete; they will be tried again next time."]
    if completed == 0 and runs and not packed.complete:
        lines.append("No run completed successfully; the file is still needed to find out why.")
    if packed.progress:
        lines.append(_progress_line(packed.progress))
    if not packed.complete and exit_code == 0:
        lines.append("Run the same command again in this directory to continue.")
    lines.append(send)
    return lines


def _pick_job(name):
    if name is not None:
        return load_job(name)
    names = list_jobs()
    if len(names) == 1:
        return load_job(names[0])
    if not names:
        raise JobError("this image contains no job")
    shown = names[:MAX_LISTED_JOBS]
    more = [f"... and {len(names) - len(shown)} more"] if len(names) > len(shown) else []
    intro = "Several jobs are available. Run the command again with one of these names:"
    raise JobError("\n".join([intro, *shown, *more]))


def _run_preflight(job, workdir):
    with _signals() as guard:
        guard.active = True
        try:
            return preflight_module.run_preflight(job, workdir), None
        except Interrupted:
            return None, 128 + guard.signum


def _gate(job, workdir, skip_preflight):
    if skip_preflight:
        problem = preflight_module.prefetch_model(job)
        if problem is not None:
            print(f"The model could not be downloaded, so nothing was run: {problem}")
            return 1
        return None
    result, interrupted = _run_preflight(job, workdir)
    if interrupted is not None:
        print("Stopped while checking the machine; nothing was run.")
        return interrupted
    if not result["ok"]:
        print("\n".join(preflight_message(result, workdir)))
        return 1
    return None


def _apply_git_sha(job):
    if GIT_SHA_FILE.is_file():
        os.environ.setdefault("POPPER_GIT_SHA", GIT_SHA_FILE.read_text(encoding="utf-8").strip())
    os.environ.setdefault("POPPER_GIT_SHA", job.get("git_sha") or "unknown")


def run_job(name, workdir=".", skip_preflight=False, replicates=None, per_task=None, max_hours=None, clock=None):
    clock = clock or time.monotonic
    started = clock()
    try:
        job = _pick_job(name)
        hours = resolve_max_hours(max_hours)
    except (JobError, ValueError) as e:
        print(e)
        return 2
    _apply_git_sha(job)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    bundle = workdir / "bundle"
    blocked = _gate(job, workdir, skip_preflight)
    if blocked is not None:
        return blocked
    sif = _SifHash()
    if bundle.exists():
        store.close_stale_runs(bundle)
    budget = Budget(bundle, hours, clock, started)
    job_dir = Path(job["dir"])
    code = run_experiment_job(
        job_dir / job["spec"],
        job["model"],
        job_dir / job["snapshot"],
        bundle,
        replicates=replicates,
        per_task=per_task,
        budget=budget,
        library_from=job_dir / job["library_from"] if job.get("library_from") else None,
        log_dir=workdir,
    )
    packed = pack_archive(bundle, workdir, job, sif, code, budget) if bundle.exists() else None
    if packed is None:
        if budget.exhausted:
            print(f"The {hours:g}-hour limit is too short to start even one run, so nothing was run.")
            return code or 1
        archive = pack_diagnostics(workdir, job, sif, code, budget)
        print("\n".join(diagnostics_message(archive, workdir)))
        return code or 1
    print("\n".join(finish_message(packed, code, budget)))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m cluster.job")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("job", nargs="?")
    run.add_argument("--workdir", default=".")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--replicates", type=int)
    run.add_argument("--examples-per-task", type=int)
    run.add_argument("--max-hours", type=float)
    canary = commands.add_parser("canary")
    canary.add_argument("--workdir", default=".")
    args = parser.parse_args(argv)
    if args.command == "canary":
        return canary_module.run_canary(args.workdir)
    return run_job(args.job, args.workdir, args.skip_preflight, args.replicates, args.examples_per_task, args.max_hours)


if __name__ == "__main__":
    sys.exit(main())
