import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from cluster import preflight as preflight_module
from cluster.common import Interrupted, _signals, run_experiment_job
from cluster.upload import maybe_upload
from popper_upgrade.db.rows import run_key
from popper_upgrade.eval.report import report_from_bundle, write_report

JOBS_DIR = Path(os.environ.get("POPPER_JOBS_DIR", Path(__file__).resolve().parent / "jobs"))
GIT_SHA_FILE = Path(__file__).resolve().parent / "GIT_SHA"
OPERATOR = "Vladimir"
MAX_LISTED_JOBS = 6
RUNS = "runs.jsonl"
JSONL_FILES = ("run_skills.jsonl", "design_reviews.jsonl", "skill_decisions.jsonl")
COPIED_FILES = ("experiments.jsonl", "candidate_skills.jsonl", "manifest.json")


class JobError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc)


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


def _read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _merged_runs(directory):
    merged = {}
    for row in _read_lines(directory / RUNS):
        merged[row["id"]] = {**merged.get(row["id"], {}), **row}
    return merged


def close_stale_runs(bundle):
    stale = [run_id for run_id, row in _merged_runs(bundle).items() if row.get("status") == "running"]
    if stale:
        with (bundle / RUNS).open("a", encoding="utf-8") as handle:
            for run_id in stale:
                row = {
                    "id": run_id,
                    "status": "error",
                    "error": "the job stopped before this run finished",
                    "finished_at": _now().isoformat(),
                }
                handle.write(json.dumps(row) + "\n")
    return len(stale)


def _final_runs(bundle):
    best = {}
    for row in _merged_runs(bundle).values():
        key = run_key(row)
        held = best.get(key)
        if held is None or row.get("status") == "completed" or held.get("status") != "completed":
            best[key] = row
    return list(best.values())


def stage_results(bundle, stage):
    stage.mkdir(parents=True, exist_ok=True)
    runs = _final_runs(bundle)
    kept = {run["id"] for run in runs}
    _write_lines(stage / RUNS, runs)
    for name in JSONL_FILES:
        _write_lines(stage / name, [row for row in _read_lines(bundle / name) if row.get("run_id") in kept | {None}])
    for name in COPIED_FILES:
        if (bundle / name).exists():
            shutil.copyfile(bundle / name, stage / name)
    (stage / "trajectories").mkdir(exist_ok=True)
    for run_id in kept:
        source = bundle / "trajectories" / f"{run_id}.json"
        if source.is_file():
            shutil.copyfile(source, stage / "trajectories" / source.name)
    return runs


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


def build_manifest(base, job, runs, sif, exit_code):
    completed = [run for run in runs if run.get("status") == "completed"]
    served = next(
        (m["served_model_id"] for m in preflight_module.read_snapshot(job)["models"] if m["name"] == job["model"]),
        job["model"],
    )
    return {
        **base,
        "job": job["name"],
        "model": job["model"],
        "finished_at": _now().isoformat(),
        "exit_code": exit_code,
        "sif_path": str(sif.path) if sif.path else None,
        "sif_sha256": sif.result(),
        "popper_sha": _popper_commit(),
        "vllm_version": _package_version("vllm"),
        "gpu": _gpu(),
        "model_revision": _model_revision(served),
        "n_runs": len(runs),
        "n_completed": len(completed),
        "n_errors": sum(1 for run in runs if run.get("status") == "error"),
        "runs_without_tokens": sum(1 for run in completed if run.get("tokens_in") is None),
    }


def pack_archive(bundle, workdir, job, sif, exit_code):
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    stem = f"popper_results_{job['name']}_{stamp}"
    workdir = Path(workdir)
    close_stale_runs(bundle)
    stage = Path(tempfile.mkdtemp(dir=workdir, prefix=".stage_")) / stem
    try:
        runs = stage_results(bundle, stage)
        if not runs:
            return None, []
        write_report(report_from_bundle(stage), stage)
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        manifest = build_manifest(manifest, job, runs, sif, exit_code)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if (workdir / "preflight.json").exists():
            shutil.copyfile(workdir / "preflight.json", stage / "preflight.json")
        archive = workdir / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(stage, arcname=stem)
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)
    return archive, runs


def preflight_message(result, workdir):
    failed = [check for check in result["checks"] if not check["ok"]]
    lines = ["The machine check failed, so nothing was run."]
    lines += [f"- {check['name']}: {check['detail']}" for check in failed[:5]]
    lines.append(f"Please send this file to {OPERATOR}: {Path(workdir).resolve() / 'preflight.json'}")
    return lines


def finish_message(archive, runs, exit_code, url):
    completed = sum(1 for run in runs if run.get("status") == "completed")
    errors = sum(1 for run in runs if run.get("status") == "error")
    if exit_code == 0:
        lines = [f"Finished: {len(runs)} runs ({completed} completed, {errors} ended in an error)."]
        if completed == 0:
            lines.append("No run completed successfully; the file is still needed to find out why.")
    else:
        lines = [
            f"The job was stopped early (exit code {exit_code}) after {len(runs)} runs.",
            "Progress is saved: run the same command again in this directory to continue.",
        ]
    lines.append(f"Uploaded automatically: {url}" if url else "The results were not uploaded automatically.")
    lines.append(f"Please send this file to {OPERATOR}: {archive.resolve()}")
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


def _apply_git_sha(job):
    if GIT_SHA_FILE.is_file():
        os.environ.setdefault("POPPER_GIT_SHA", GIT_SHA_FILE.read_text(encoding="utf-8").strip())
    os.environ.setdefault("POPPER_GIT_SHA", job.get("git_sha") or "unknown")


def run_job(name, workdir=".", skip_preflight=False, replicates=None):
    try:
        job = _pick_job(name)
    except JobError as e:
        print(e)
        return 2
    _apply_git_sha(job)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    bundle = workdir / "bundle"
    if not skip_preflight:
        result, interrupted = _run_preflight(job, workdir)
        if interrupted is not None:
            print("Stopped while checking the machine; nothing was run.")
            return interrupted
        if not result["ok"]:
            print("\n".join(preflight_message(result, workdir)))
            return 1
    sif = _SifHash()
    if bundle.exists():
        close_stale_runs(bundle)
    job_dir = Path(job["dir"])
    code = run_experiment_job(
        job_dir / job["spec"],
        job["model"],
        job_dir / job["snapshot"],
        bundle,
        replicates=replicates,
    )
    archive, runs = pack_archive(bundle, workdir, job, sif, code) if bundle.exists() else (None, [])
    if archive is None:
        print("No run was recorded, so there are no results to send.")
        return code or 1
    url = maybe_upload(archive) if code == 0 else None
    print("\n".join(finish_message(archive, runs, code, url)))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m cluster.job")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("job", nargs="?")
    run.add_argument("--workdir", default=".")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--replicates", type=int)
    args = parser.parse_args(argv)
    return run_job(args.job, args.workdir, args.skip_preflight, args.replicates)


if __name__ == "__main__":
    sys.exit(main())
