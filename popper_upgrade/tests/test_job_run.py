import hashlib
import json
import signal
import subprocess
import sys
import tarfile
from collections import Counter

import popper_upgrade.run as run_module
from cluster import job as job_module
from cluster import store
from popper_upgrade.serving import logs
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeExtractor, write_job

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


def _read_bytes(archive, name):
    with tarfile.open(archive) as tar:
        stem = tar.getnames()[0].split("/")[0]
        return tar.extractfile(f"{stem}/{name}").read()


def _runs(archive):
    return [json.loads(line) for line in _read(archive, "runs.jsonl").splitlines()]


def _all_runs(workdir):
    rows = []
    for archive in sorted(workdir.glob("popper_results_*.tar.gz")):
        rows += _runs(archive)
    return rows


def _identity(row):
    return (row["model_name"], row["arm"], row.get("task"), row["hypothesis"], row.get("replicate"), row["permuted"])


def _sigterm_at(env, count):
    env.popper.on_validate = staticmethod(lambda n: signal.raise_signal(signal.SIGTERM) if n == count else None)


def test_archive_carries_the_log_tails(job_env):
    job_env.workdir.mkdir(parents=True)
    (job_env.workdir / "vllm.log").write_bytes(b"x" * logs.TAIL_BYTES + b"tail-of-vllm")
    (job_env.workdir / "proxy.log").write_text("proxy line\n", encoding="utf-8")

    _run(job_env)

    archive = _latest_archive(job_env.workdir)
    vllm_log = _read_bytes(archive, "vllm.log")
    assert len(vllm_log) == logs.TAIL_BYTES
    assert vllm_log.endswith(b"tail-of-vllm")
    assert _read(archive, "proxy.log").endswith("proxy line\n")
    assert job_env.server.log_paths[-1] == job_env.workdir / "vllm.log"


def test_archive_without_server_logs_has_none(job_env):
    _run(job_env)

    with tarfile.open(_latest_archive(job_env.workdir)) as tar:
        assert not [name for name in tar.getnames() if name.endswith(".log")]


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
    assert len(rows) == TOTAL_RUNS - 2
    assert {row["status"] for row in rows} == {"completed"}
    final = [row for row in _all_runs(job_env.workdir) if row["status"] == "completed"]
    assert len(final) == TOTAL_RUNS
    assert set(Counter(_identity(row) for row in final).values()) == {1}


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


def test_vllm_version_is_read_from_the_serving_interpreter(monkeypatch):
    monkeypatch.setenv("POPPER_VLLM_PYTHON", sys.executable)
    monkeypatch.setattr(
        job_module.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, stdout="9.9.9"),
    )

    assert job_module._vllm_version() == "9.9.9"


def test_vllm_version_is_none_when_the_serving_interpreter_is_missing(monkeypatch):
    monkeypatch.setenv("POPPER_VLLM_PYTHON", "/nonexistent/python")

    assert job_module._vllm_version() is None


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


def test_finish_message_names_the_archive_and_has_no_upload_line(job_env, capsys):
    assert _run(job_env) == 0

    archive = _latest_archive(job_env.workdir)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == f"ALL DONE — send this file to Vladimir: {archive.resolve()}"
    assert not any("upload" in line.lower() for line in lines)
    assert len(lines) <= 10


def test_stale_running_rows_are_closed_before_resuming(job_env):
    bundle = job_env.workdir / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "runs.jsonl").write_text(json.dumps({"id": "r1", "status": "running"}) + "\n", encoding="utf-8")

    assert store.close_stale_runs(bundle) == 1

    merged = store.merged_runs(bundle)
    assert merged["r1"]["status"] == "error"
    assert store.close_stale_runs(bundle) == 0


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
    assert job_module.main(["run", "--max-hours", "2.5"]) == 0
    assert calls == [
        ("demo", ".", True, 1, None, None),
        (None, ".", False, None, None, None),
        (None, ".", False, None, None, 2.5),
    ]


TRAIN = ["train one", "train two"]


def _lines(archive, name):
    return [json.loads(line) for line in _read(archive, name).splitlines()]


def _learning_job(env, monkeypatch, skills):
    monkeypatch.setattr(run_module, "SkillExtractor", lambda llm, **kwargs: FakeExtractor(skills))
    return write_job(env.tmp_path / "jobs" / "learn", name="learn", arms=("baseline", "skills"), train=TRAIN)


def test_two_phase_job_learns_freezes_and_evaluates(job_env, monkeypatch):
    skill = Skill("x", "compare groups", "template", "", "social science", "train one")
    job = _learning_job(job_env, monkeypatch, [skill])

    assert job_module.run_job(job["dir"], job_env.workdir, skip_preflight=True) == 0

    archive = _latest_archive(job_env.workdir)
    experiments = {row["spec"]["phase"]: row for row in _lines(archive, "experiments.jsonl")}
    assert set(experiments) == {"learn", "evaluate"}
    freeze = json.loads(_read(archive, "freeze.json"))
    assert (freeze["candidates"], freeze["learned_skills"], freeze["approved_skills"]) == (1, 1, 1)
    assert freeze["empty_library"] is False
    runs = _runs(archive)
    learn_runs = [row for row in runs if row["experiment_id"] == experiments["learn"]["id"]]
    test_runs = [row for row in runs if row["experiment_id"] == experiments["evaluate"]["id"]]
    assert {row["arm"] for row in learn_runs} == {"skills_reviewer"}
    assert len(learn_runs) == len(TRAIN)
    assert {row["library_snapshot_id"] for row in learn_runs} == {freeze["base_snapshot_id"]}
    assert {row["library_snapshot_id"] for row in test_runs} == {freeze["snapshot_id"]}
    assert not {row["hypothesis"] for row in test_runs} & set(TRAIN)
    summary = _read(archive, "summary.md")
    assert "Learning phase" in summary
    assert "empty_library" not in summary
    manifest = json.loads(_read(archive, "manifest.json"))
    assert manifest["snapshot_id"] == freeze["base_snapshot_id"]


def test_empty_learned_library_is_flagged_and_evaluation_still_runs(job_env, monkeypatch):
    job = _learning_job(job_env, monkeypatch, [])

    assert job_module.run_job(job["dir"], job_env.workdir, skip_preflight=True) == 0

    archive = _latest_archive(job_env.workdir)
    assert json.loads(_read(archive, "freeze.json"))["empty_library"] is True
    assert "empty_library" in _read(archive, "summary.md")
    evaluate = next(row for row in _lines(archive, "experiments.jsonl") if row["spec"]["phase"] == "evaluate")
    completed = [row for row in _runs(archive) if row["experiment_id"] == evaluate["id"]]
    assert completed
    assert {row["status"] for row in completed} == {"completed"}


def test_two_phase_job_resumes_without_relearning(job_env, monkeypatch):
    skill = Skill("x", "compare groups", "template", "", "social science", "train one")
    job = _learning_job(job_env, monkeypatch, [skill])
    _sigterm_at(job_env, len(TRAIN) + 2)

    first = job_module.run_job(job["dir"], job_env.workdir, skip_preflight=True)

    assert first == 128 + signal.SIGTERM
    job_env.popper.on_validate = None
    assert job_module.run_job(job["dir"], job_env.workdir, skip_preflight=True) == 0
    trained = [hypothesis for hypothesis in job_env.popper.validations if hypothesis in TRAIN]
    assert sorted(trained) == sorted(TRAIN)
    archive = _latest_archive(job_env.workdir)
    freeze = json.loads(_read(archive, "freeze.json"))
    evaluated = [row for row in _runs(archive) if row.get("library_snapshot_id") == freeze["snapshot_id"]]
    assert len(evaluated) == len({_identity(row) for row in evaluated})


def test_library_from_skips_learning_and_evaluates_with_the_supplied_library(job_env, monkeypatch):
    from popper_upgrade.tests.fakes import FakeEmbedder, write_snapshot

    skill = Skill("x", "compare groups", "template", "", "social science", "train one")
    job = _learning_job(job_env, monkeypatch, [skill])
    job_dir = job_env.tmp_path / "jobs" / "learn"
    write_snapshot(job_dir / "library.json", [], FakeEmbedder(), snapshot_id="shared-library")
    meta = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    (job_dir / "job.json").write_text(json.dumps({**meta, "library_from": "library.json"}), encoding="utf-8")

    assert job_module.run_job(job["dir"], job_env.workdir, skip_preflight=True) == 0

    archive = _latest_archive(job_env.workdir)
    experiments = _lines(archive, "experiments.jsonl")
    assert [row["spec"].get("phase") for row in experiments] == [None]
    runs = _runs(archive)
    assert {row["library_snapshot_id"] for row in runs} == {"shared-library"}
    assert not {row["hypothesis"] for row in runs} & set(TRAIN)
    assert "freeze.json" not in tarfile.open(archive).getnames()
    assert json.loads(_read(archive, "manifest.json"))["frozen_snapshot_id"] == "shared-library"
