import hashlib
import json
import signal
import sys
import tarfile
from collections import Counter
from types import SimpleNamespace

from cluster import job as job_module
from popper_upgrade.tests.fakes import write_job

TOTAL_RUNS = 6


def _run(env, **kwargs):
    kwargs.setdefault("skip_preflight", True)
    return job_module.run_job(env.job_dir, env.workdir, **kwargs)


def _latest_archive(workdir):
    return sorted(workdir.glob("popper_results_*.tar.gz"))[-1]


def _read(archive, name):
    with tarfile.open(archive) as tar:
        stem = tar.getnames()[0].split("/")[0]
        member = tar.extractfile(f"{stem}/{name}")
        return member.read().decode("utf-8")


def _runs(archive):
    return [json.loads(line) for line in _read(archive, "runs.jsonl").splitlines()]


def _identity(row):
    return (row["model_name"], row["arm"], row.get("task"), row["hypothesis"], row.get("replicate"), row["permuted"])


def _sigterm_at(env, count):
    env.popper.on_validate = staticmethod(lambda n: signal.raise_signal(signal.SIGTERM) if n == count else None)


def test_resume_skips_completed_runs(job_env):
    _sigterm_at(job_env, 3)

    first = _run(job_env)

    assert first == 128 + signal.SIGTERM
    assert len(job_env.popper.validations) == 3
    job_env.popper.on_validate = None
    job_env.popper.validations.clear()

    second = _run(job_env)

    assert second == 0
    assert len(job_env.popper.validations) == TOTAL_RUNS - 2
    rows = _runs(_latest_archive(job_env.workdir))
    assert len(rows) == TOTAL_RUNS
    assert {row["status"] for row in rows} == {"completed"}
    assert set(Counter(_identity(row) for row in rows).values()) == {1}


def test_sigterm_still_packs_archive(job_env, capsys):
    _sigterm_at(job_env, 2)

    code = _run(job_env)

    assert code == 128 + signal.SIGTERM
    archive = _latest_archive(job_env.workdir)
    rows = _runs(archive)
    assert Counter(row["status"] for row in rows) == {"completed": 1, "error": 1}
    assert "received signal" in next(row for row in rows if row["status"] == "error")["error"]
    manifest = json.loads(_read(archive, "manifest.json"))
    assert manifest["exit_code"] == code
    lines = capsys.readouterr().out.strip().splitlines()
    assert "run the same command again" in "\n".join(lines)
    assert lines[-1].endswith(archive.name)


def test_sigterm_is_not_swallowed_by_agent_code(job_env, monkeypatch):
    original = job_env.popper.validate

    def swallowing(self, hypothesis):
        try:
            return original(self, hypothesis)
        except Exception:
            return {"parsed_result": {"conclusion": True}, "last_message": "recovered"}

    monkeypatch.setattr(job_env.popper, "validate", swallowing)
    _sigterm_at(job_env, 2)

    code = _run(job_env)

    assert code == 128 + signal.SIGTERM
    assert len(job_env.popper.validations) == 2
    rows = _runs(_latest_archive(job_env.workdir))
    assert Counter(row["status"] for row in rows) == {"completed": 1, "error": 1}


def test_sigterm_while_checking_machine_runs_nothing(job_env, monkeypatch, capsys):
    def interrupted(job, workdir):
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(job_module.preflight_module, "run_preflight", interrupted)

    code = _run(job_env, skip_preflight=False)

    assert code == 128 + signal.SIGTERM
    assert job_env.popper.validations == []
    assert list(job_env.workdir.glob("popper_results_*")) == []
    assert "nothing was run" in capsys.readouterr().out


def test_manifest_fields(job_env, tmp_path, monkeypatch):
    sif = tmp_path / "popper_upgrade.sif"
    sif.write_bytes(b"sif bytes")
    monkeypatch.setenv("APPTAINER_CONTAINER", str(sif))
    monkeypatch.setattr(job_module, "_gpu", lambda: {"name": "Test GPU", "vram_gb": 80.0})
    monkeypatch.setattr(job_module, "_popper_commit", lambda: "popper-sha")

    assert _run(job_env) == 0

    manifest = json.loads(_read(_latest_archive(job_env.workdir), "manifest.json"))
    assert manifest["job"] == "demo"
    assert manifest["model"] == job_env.job["model"]
    assert manifest["snapshot_id"] == "snap-1"
    assert manifest["git_sha"] == "job-build-sha"
    assert manifest["sif_sha256"] == hashlib.sha256(b"sif bytes").hexdigest()
    assert manifest["sif_path"] == str(sif)
    assert manifest["popper_sha"] == "popper-sha"
    assert manifest["gpu"] == {"name": "Test GPU", "vram_gb": 80.0}
    assert manifest["exit_code"] == 0
    assert (manifest["n_runs"], manifest["n_completed"], manifest["n_errors"]) == (TOTAL_RUNS, TOTAL_RUNS, 0)
    extra = {"created_at", "host", "finished_at", "vllm_version", "model_revision", "runs_without_tokens"}
    assert extra <= set(manifest)


def test_archive_layout_and_summary(job_env):
    assert _run(job_env) == 0

    archive = _latest_archive(job_env.workdir)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    stem = archive.name.removesuffix(".tar.gz")
    expected = {stem, *(f"{stem}/{name}" for name in ("manifest.json", "runs.jsonl", "summary.md", "summary.csv"))}
    assert expected <= set(names)
    assert f"{stem}/experiments.jsonl" in names
    assert "demo_spec" in _read(archive, "summary.md")
    assert not [path for path in job_env.workdir.iterdir() if path.name.startswith(".stage_")]


def test_preflight_json_is_packed_when_checks_pass(job_env, monkeypatch):
    def passing(job, workdir):
        result = {"ok": True, "checks": [{"name": "gpu", "ok": True, "detail": "fine"}]}
        (workdir / "preflight.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(job_module.preflight_module, "run_preflight", passing)

    assert _run(job_env, skip_preflight=False) == 0

    assert json.loads(_read(_latest_archive(job_env.workdir), "preflight.json"))["ok"] is True


def test_failed_preflight_writes_no_archive_and_names_the_file(job_env, monkeypatch, capsys):
    def failing(job, workdir):
        result = {"ok": False, "checks": [{"name": "gpu", "ok": False, "detail": "no NVIDIA GPU is visible"}]}
        (workdir / "preflight.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(job_module.preflight_module, "run_preflight", failing)

    code = _run(job_env, skip_preflight=False)

    assert code == 1
    assert list(job_env.workdir.glob("popper_results_*")) == []
    assert job_env.popper.validations == []
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) <= 10
    assert "no NVIDIA GPU is visible" in lines[1]
    assert lines[-1].endswith("preflight.json")


def test_upload_failure_keeps_the_archive_prints_manual_hand_over_and_exits_zero(job_env, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("POPPER_RESULTS_REPO", "lab/results")

    class Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, **kwargs):
            raise OSError("network down")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api))

    code = _run(job_env)

    assert code == 0
    archive = _latest_archive(job_env.workdir)
    assert archive.exists()
    lines = capsys.readouterr().out.strip().splitlines()
    assert any("Automatic upload failed" in line for line in lines)
    assert lines[-2] == "The results were not uploaded automatically."
    assert lines[-1] == f"Please send this file to Vladimir: {archive.resolve()}"


def test_successful_upload_is_announced(job_env, monkeypatch, capsys):
    monkeypatch.setattr(job_module, "maybe_upload", lambda archive: "https://example.org/file")

    assert _run(job_env) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert "Uploaded automatically: https://example.org/file" in lines
    assert len(lines[-4:]) <= 10


def test_stale_running_rows_are_closed_before_resuming(job_env):
    bundle = job_env.workdir / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "runs.jsonl").write_text(json.dumps({"id": "r1", "status": "running"}) + "\n", encoding="utf-8")

    assert job_module.close_stale_runs(bundle) == 1

    merged = job_module._merged_runs(bundle)
    assert merged["r1"]["status"] == "error"
    assert job_module.close_stale_runs(bundle) == 0


def test_default_job_is_the_only_one_installed(job_env, monkeypatch):
    monkeypatch.setattr(job_module, "JOBS_DIR", job_env.tmp_path / "jobs")

    assert job_module.list_jobs() == ["demo"]
    assert job_module.run_job(None, job_env.workdir, skip_preflight=True) == 0


def test_several_jobs_are_listed(job_env, monkeypatch, capsys):
    write_job(job_env.tmp_path / "jobs" / "other", name="other")
    monkeypatch.setattr(job_module, "JOBS_DIR", job_env.tmp_path / "jobs")

    assert job_module.run_job(None, job_env.workdir, skip_preflight=True) == 2

    out = capsys.readouterr().out
    assert "demo" in out
    assert "other" in out
    assert job_env.popper.validations == []


def test_no_jobs_is_reported(job_env, monkeypatch, capsys):
    monkeypatch.setattr(job_module, "JOBS_DIR", job_env.tmp_path / "nowhere")

    assert job_module.run_job(None, job_env.workdir, skip_preflight=True) == 2
    assert "no job" in capsys.readouterr().out


def test_unknown_job_name_is_refused(job_env, monkeypatch, capsys):
    monkeypatch.setattr(job_module, "JOBS_DIR", job_env.tmp_path / "jobs")

    assert job_module.run_job("missing", job_env.workdir, skip_preflight=True) == 2
    assert "no job called missing" in capsys.readouterr().out


def test_nothing_recorded_means_no_archive(job_env, monkeypatch, capsys):
    monkeypatch.setattr(job_module, "run_experiment_job", lambda *args, **kwargs: 1)

    assert _run(job_env) == 1
    assert list(job_env.workdir.glob("popper_results_*")) == []
    assert "no results" in capsys.readouterr().out


def test_replicates_override_reaches_the_experiment(job_env):
    assert _run(job_env, replicates=1) == 0

    assert len(job_env.popper.validations) == 4


def test_main_parses_the_runscript_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(job_module, "run_job", lambda *args: calls.append(args) or 0)

    assert job_module.main(["run", "demo", "--skip-preflight", "--replicates", "1"]) == 0
    assert job_module.main(["run"]) == 0
    assert calls == [("demo", ".", True, 1), (None, ".", False, None)]
