import csv
import io
import json
import re
import signal
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import popper_upgrade.run as run_module
import popper_upgrade.session as session_module
from cluster import job as job_module
from cluster import parts as parts_module
from cluster import store
from cluster.budget import (
    DEFAULT_EST_MINUTES,
    MAX_HOURS_ENV,
    SAFETY_SECONDS,
    Budget,
    hint_seconds,
    resolve_max_hours,
)
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.report import report_from_parts
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeEmbedder, FakeExtractor, write_job, write_snapshot
from popper_upgrade.tests.part_fixtures import build_two_parts

TOTAL_RUNS = 6
RUN_SECONDS = 600
TRAIN = ["train one", "train two"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(job_env, monkeypatch):
    fake = Clock()
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=fake))
    job_env.popper.on_validate = staticmethod(lambda count: fake.advance(RUN_SECONDS))
    return fake


def _night(env, clock, hours, job=None, **kwargs):
    return job_module.run_job(
        job["dir"] if job else env.job_dir, env.workdir, skip_preflight=True, max_hours=hours, clock=clock, **kwargs
    )


def _archives(workdir):
    return sorted(workdir.glob("popper_results_*.tar.gz"))


def _member(archive, name):
    with tarfile.open(archive) as tar:
        stem = tar.getnames()[0].split("/")[0]
        member = tar.extractfile(f"{stem}/{name}")
        return member.read().decode("utf-8")


def _names(archive):
    with tarfile.open(archive) as tar:
        stem = tar.getnames()[0].split("/")[0]
        return {name.removeprefix(f"{stem}/") for name in tar.getnames()}


def _lines(archive, name):
    return [json.loads(line) for line in _member(archive, name).splitlines()]


def _ids(archive):
    return [row["id"] for row in _lines(archive, "runs.jsonl")]


def _manifest(archive):
    return json.loads(_member(archive, "manifest.json"))


def _learning_job(env, monkeypatch, arms=("baseline", "skills")):
    skill = Skill("x", "compare groups", "template", "", "social science", "train one")
    monkeypatch.setattr(run_module, "SkillExtractor", lambda llm, **kwargs: FakeExtractor([skill]))
    return write_job(env.tmp_path / "jobs" / "learn", name="learn", arms=arms, train=TRAIN)


def test_budget_stops_before_a_run_that_would_not_fit(job_env, clock, capsys):
    code = _night(job_env, clock, 0.7)

    assert code == 0
    assert len(job_env.popper.validations) == 3
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert "budget reached" in out
    assert len(lines) <= 10
    archive = _archives(job_env.workdir)[0]
    assert lines[-1] == f"Please send this file to Vladimir: {archive.resolve()}"
    rows = _lines(archive, "runs.jsonl")
    assert [row["status"] for row in rows] == ["completed"] * 3


def test_budget_never_interrupts_a_run_that_has_started(job_env, monkeypatch, capsys):
    fake = Clock()
    monkeypatch.setattr(session_module, "time", SimpleNamespace(monotonic=fake))
    job_env.popper.on_validate = staticmethod(lambda count: fake.advance(10 * 3600))

    code = _night(job_env, fake, 1.0)

    assert code == 0
    assert len(job_env.popper.validations) == 1
    rows = _lines(_archives(job_env.workdir)[0], "runs.jsonl")
    assert [row["status"] for row in rows] == ["completed"]
    assert rows[0]["wall_time_s"] == 10 * 3600
    assert "budget reached" in capsys.readouterr().out


def test_budget_too_small_for_any_run_says_so(job_env, clock, capsys):
    code = _night(job_env, clock, 0.1)

    assert code == 1
    assert job_env.popper.validations == []
    assert list(job_env.workdir.glob("popper_results_*")) == []
    assert "too short" in capsys.readouterr().out


def test_nightly_parts_hold_only_new_runs_and_are_numbered(job_env, clock, capsys):
    assert _night(job_env, clock, 0.7) == 0
    first_out = capsys.readouterr().out.strip().splitlines()
    assert _night(job_env, clock, 0.7) == 0
    second_out = capsys.readouterr().out.strip().splitlines()

    first, second = _archives(job_env.workdir)
    assert re.fullmatch(r"popper_results_demo_part01_\d{8}T\d{6}Z\.tar\.gz", first.name)
    assert re.fullmatch(r"popper_results_demo_part02_\d{8}T\d{6}Z\.tar\.gz", second.name)
    assert len(_ids(first)) == len(_ids(second)) == 3
    assert not set(_ids(first)) & set(_ids(second))
    assert [part["part"] for part in parts_module.read_parts(job_env.workdir)] == [1, 2]
    assert _manifest(first)["part"] == 1
    assert _manifest(second)["part"] == 2
    assert _manifest(first)["n_runs"] == _manifest(second)["n_runs"] == 3
    assert "budget reached" in "\n".join(first_out)
    assert second_out[-1].startswith("ALL DONE — send this file")
    assert second_out[-1].endswith(str(second.resolve()))
    assert len(store.final_runs(job_env.workdir / "bundle")) == TOTAL_RUNS


def test_part_manifest_reports_cumulative_progress(job_env, clock):
    _night(job_env, clock, 0.7)
    _night(job_env, clock, 0.7)

    first, second = (_manifest(path) for path in _archives(job_env.workdir))

    assert first["progress"]["done"] == 3
    assert first["progress"]["total"] == TOTAL_RUNS
    assert first["progress"]["phases"]["evaluate"]["arms"] == {"baseline": {"done": 3, "total": TOTAL_RUNS}}
    assert first["complete"] is False
    assert first["budget_reached"] is True
    assert first["max_hours"] == 0.7
    assert second["progress"]["done"] == TOTAL_RUNS
    assert second["complete"] is True


def test_default_has_no_time_limit(job_env, clock, monkeypatch, capsys):
    monkeypatch.delenv(MAX_HOURS_ENV, raising=False)
    job_env.popper.on_validate = staticmethod(lambda count: clock.advance(100 * 3600))

    assert _night(job_env, clock, None) == 0

    assert len(job_env.popper.validations) == TOTAL_RUNS
    archives = _archives(job_env.workdir)
    assert len(archives) == 1
    manifest = _manifest(archives[0])
    assert manifest["max_hours"] is None
    assert manifest["budget_reached"] is False
    assert manifest["complete"] is True
    out = capsys.readouterr().out
    assert "budget reached" not in out
    assert "nights" not in out
    assert out.strip().splitlines()[-1].startswith("ALL DONE — send this file")


def test_interrupted_run_continues_in_the_next_part_without_a_limit(job_env, clock, capsys):
    job_env.popper.on_validate = staticmethod(lambda count: signal.raise_signal(signal.SIGTERM) if count == 3 else None)

    assert _night(job_env, clock, None) == 128 + signal.SIGTERM
    job_env.popper.on_validate = None
    assert _night(job_env, clock, None) == 0

    first, second = _archives(job_env.workdir)
    assert [part["part"] for part in parts_module.read_parts(job_env.workdir)] == [1, 2]
    assert not set(_ids(first)) & set(_ids(second))
    assert len(store.final_runs(job_env.workdir / "bundle")) == TOTAL_RUNS
    assert "nights" not in capsys.readouterr().out


def test_a_completed_job_run_again_adds_an_empty_part(job_env, clock, capsys):
    _night(job_env, clock, 7)
    capsys.readouterr()
    before = job_env.popper.validations[:]

    assert _night(job_env, clock, 7) == 0

    assert job_env.popper.validations == before
    last = _archives(job_env.workdir)[-1]
    assert _ids(last) == []
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("ALL DONE — send this file")


def test_max_hours_env_opts_in_and_the_flag_wins(job_env, clock, monkeypatch):
    monkeypatch.setenv(MAX_HOURS_ENV, "0.7")

    _night(job_env, clock, None)
    assert len(job_env.popper.validations) == 3

    monkeypatch.setenv(MAX_HOURS_ENV, "0.4")
    _night(job_env, clock, 7)
    assert len(job_env.popper.validations) == TOTAL_RUNS


@pytest.mark.parametrize("value", ["abc", "0", "-1", "nan"])
def test_invalid_hours_are_refused(job_env, clock, capsys, value):
    assert _night(job_env, clock, value) == 2
    assert job_env.popper.validations == []
    assert "hours" in capsys.readouterr().out


def test_phase_b_snapshot_ships_in_every_part_from_the_freeze_on(job_env, monkeypatch, clock):
    job = _learning_job(job_env, monkeypatch)

    assert _night(job_env, clock, 1.1, job) == 0
    assert _night(job_env, clock, 30, job) == 0

    first, second = _archives(job_env.workdir)
    freeze = json.loads(_member(first, "freeze.json"))
    assert {"freeze.json", "frozen_snapshot.json"} <= _names(first) & _names(second)
    assert json.loads(_member(second, "freeze.json"))["snapshot_id"] == freeze["snapshot_id"]
    assert _manifest(first)["frozen_snapshot_id"] == freeze["snapshot_id"]
    first_runs, second_runs = _lines(first, "runs.jsonl"), _lines(second, "runs.jsonl")
    phases = store.experiment_phases(job_env.workdir / "bundle")
    assert [store.run_phase(row, phases) for row in first_runs].count("learn") == len(TRAIN)
    assert [store.run_phase(row, phases) for row in second_runs].count("learn") == 0
    assert len(first_runs) == len(TRAIN) + 3
    assert not set(_ids(first)) & set(_ids(second))
    assert {row["library_snapshot_id"] for row in second_runs} == {freeze["snapshot_id"]}
    assert _manifest(second)["complete"] is True
    assert sum(1 for hypothesis in job_env.popper.validations if hypothesis in TRAIN) == len(TRAIN)


def test_budget_stop_during_learning_defers_the_freeze(job_env, monkeypatch, clock):
    job = _learning_job(job_env, monkeypatch)

    assert _night(job_env, clock, 0.4, job) == 0

    first = _archives(job_env.workdir)[0]
    assert len(_ids(first)) == 1
    assert not {"freeze.json", "frozen_snapshot.json"} & _names(first)
    assert _manifest(first)["frozen_snapshot_id"] is None
    assert not (job_env.workdir / "bundle" / "freeze.json").exists()
    assert _night(job_env, clock, 30, job) == 0
    second = _archives(job_env.workdir)[1]
    assert "freeze.json" in _names(second)
    assert sum(1 for hypothesis in job_env.popper.validations if hypothesis in TRAIN) == len(TRAIN)


def test_report_works_on_parts_in_any_order_with_completeness(job_env, clock):
    _night(job_env, clock, 0.7)
    _night(job_env, clock, 0.7)
    first, second = _archives(job_env.workdir)

    both = report_from_parts([second, first])
    forward = report_from_parts([first, second])
    partial = report_from_parts([first])

    assert both["rows"] == forward["rows"]
    row = both["rows"][0]
    assert (row["n_runs"], row["n_completed"], row["n_planned"], row["completeness"]) == (
        TOTAL_RUNS,
        TOTAL_RUNS,
        TOTAL_RUNS,
        1.0,
    )
    assert partial["rows"][0]["n_completed"] == 3
    assert partial["rows"][0]["completeness"] == 0.5


def test_report_lists_planned_arms_that_have_no_run_yet(job_env, clock):
    job = write_job(job_env.tmp_path / "jobs" / "two", name="two", arms=("baseline", "skills"))

    assert _night(job_env, clock, 0.4, job) == 0
    part = _archives(job_env.workdir)[0]

    report = report_from_parts([part])

    by_arm = {row["arm"]: row for row in report["rows"]}
    assert set(by_arm) == {"baseline", "skills"}
    assert by_arm["baseline"]["n_runs"] == 1
    assert by_arm["skills"]["n_runs"] == 0
    assert by_arm["skills"]["completeness"] == 0.0
    assert by_arm["skills"]["n_planned"] == 6
    assert "planned" in _member(part, "summary.md")


def test_summary_in_a_part_covers_everything_so_far(job_env, clock):
    _night(job_env, clock, 0.7)
    _night(job_env, clock, 0.7)

    second = _archives(job_env.workdir)[1]

    rows = list(csv.DictReader(io.StringIO(_member(second, "summary.csv"))))
    assert [row["n_completed"] for row in rows] == [str(TOTAL_RUNS)]


def _bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({"snapshot_id": "base"}), encoding="utf-8")
    (bundle / "trajectories").mkdir()
    return bundle


def _write(bundle, name, rows):
    store.write_lines(bundle / name, rows)


def _run(run_id, experiment, arm="baseline", status="completed", hypothesis="h", replicate=0):
    return {
        "id": run_id,
        "experiment_id": experiment,
        "status": status,
        "model_name": "m",
        "arm": arm,
        "hypothesis": hypothesis,
        "task": "t",
        "replicate": replicate,
        "permuted": False,
    }


def test_stage_part_ships_unshipped_runs_and_what_they_reference(tmp_path):
    bundle = _bundle(tmp_path)
    _write(bundle, "experiments.jsonl", [{"id": "learn"}, {"id": "eval"}, {"id": "unused"}])
    _write(
        bundle,
        "runs.jsonl",
        [_run("L1", "learn", hypothesis="a"), _run("E1", "eval", hypothesis="b"), _run("E2", "eval", hypothesis="c")],
    )
    _write(bundle, "candidate_skills.jsonl", [{"id": "c1", "source_run_id": "L1"}, {"id": "c2", "source_run_id": "E2"}])
    _write(
        bundle,
        "skill_decisions.jsonl",
        [
            {"id": "d1", "skill_id": "c1", "run_id": None, "trigger": "entry"},
            {"id": "d2", "skill_id": "c1", "run_id": "E1", "trigger": "pre_run"},
            {"id": "d3", "skill_id": "c2", "run_id": None, "trigger": "entry"},
        ],
    )
    _write(bundle, "run_skills.jsonl", [{"run_id": "E1", "skill_id": "c1", "role": "injected"}])
    (bundle / "freeze.json").write_text(json.dumps({"snapshot_id": "frozen"}), encoding="utf-8")
    (bundle / "trajectories" / "E1.json").write_text("{}", encoding="utf-8")
    (bundle / "trajectories" / "L1.json").write_text("{}", encoding="utf-8")
    previous = [{"part": 1, "runs": ["L1"], "candidates": ["c1"]}]

    runs, candidates = parts_module.stage_part(bundle, tmp_path / "stage", previous)

    assert sorted(run["id"] for run in runs) == ["E1", "E2"]
    assert candidates == ["c2"]
    stage = tmp_path / "stage"
    assert {row["id"] for row in store.read_lines(stage / "experiments.jsonl")} == {"eval"}
    assert {row["id"] for row in store.read_lines(stage / "candidate_skills.jsonl")} == {"c1", "c2"}
    assert {row["id"] for row in store.read_lines(stage / "skill_decisions.jsonl")} == {"d2", "d3"}
    assert len(store.read_lines(stage / "run_skills.jsonl")) == 1
    assert (stage / "freeze.json").exists()
    assert sorted(path.name for path in (stage / "trajectories").iterdir()) == ["E1.json"]


def test_stage_part_after_everything_shipped_is_empty(tmp_path):
    bundle = _bundle(tmp_path)
    _write(bundle, "experiments.jsonl", [{"id": "eval"}])
    _write(bundle, "runs.jsonl", [_run("E1", "eval")])

    runs, candidates = parts_module.stage_part(bundle, tmp_path / "stage", [{"part": 1, "runs": ["E1"]}])

    assert (runs, candidates) == ([], [])
    assert store.read_lines(tmp_path / "stage" / "experiments.jsonl") == []


def test_an_error_attempt_and_its_later_completion_land_in_different_parts(tmp_path):
    bundle = _bundle(tmp_path)
    _write(bundle, "experiments.jsonl", [{"id": "eval"}])
    _write(bundle, "runs.jsonl", [_run("E1", "eval", status="error")])
    first, _ = parts_module.stage_part(bundle, tmp_path / "one", [])
    parts = [{"part": 1, "runs": [run["id"] for run in first]}]
    with (bundle / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_run("E2", "eval")) + "\n")

    second, _ = parts_module.stage_part(bundle, tmp_path / "two", parts)

    assert [run["id"] for run in first] == ["E1"]
    assert [(run["id"], run["status"]) for run in second] == [("E2", "completed")]


def _runs_file(bundle, rows):
    _write(bundle, "runs.jsonl", rows)


def _timed(run_id, experiment, seconds, status="completed"):
    return {**_run(run_id, experiment, hypothesis=run_id), "wall_time_s": seconds, "status": status}


def test_estimate_is_the_median_per_phase_and_falls_back_to_the_hint(tmp_path):
    bundle = _bundle(tmp_path)
    _write(
        bundle,
        "experiments.jsonl",
        [{"id": "learn", "spec": {"phase": "learn"}}, {"id": "eval", "spec": {"phase": "evaluate"}}],
    )
    _runs_file(
        bundle,
        [
            _timed("a", "learn", 100),
            _timed("b", "learn", 500),
            _timed("c", "learn", 200),
            _timed("d", "learn", 99999, status="error"),
        ],
    )
    budget = Budget(bundle, 7, Clock())
    budget.hint = 777

    assert budget.estimate("learn") == 200
    assert budget.estimate("evaluate") == 777


def test_estimate_uses_timestamps_when_wall_time_is_missing(tmp_path):
    bundle = _bundle(tmp_path)
    _write(bundle, "experiments.jsonl", [{"id": "eval"}])
    row = _run("a", "eval")
    row.update(created_at="2026-10-19T00:00:00+00:00", finished_at="2026-10-19T00:05:00+00:00")
    _runs_file(bundle, [row])

    assert Budget(bundle, 7, Clock()).estimate("evaluate") == 300


def test_gate_uses_elapsed_estimate_and_ten_minute_safety(tmp_path):
    clock = Clock()
    budget = Budget(tmp_path / "bundle", 1.0, clock)
    budget.hint = 1200

    assert SAFETY_SECONDS == 600
    assert budget.may_start("evaluate") is True
    clock.advance(1800)
    assert budget.may_start("evaluate") is True
    clock.advance(1)
    assert budget.may_start("evaluate") is False
    assert budget.exhausted is True
    clock.now = 0
    assert budget.may_start("evaluate") is False


def test_hint_comes_from_spec_then_model_then_default(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"est_run_minutes": 20}), encoding="utf-8")
    per_model = tmp_path / "per_model.json"
    per_model.write_text(json.dumps({"est_run_minutes": {"m": 30}}), encoding="utf-8")
    plain = tmp_path / "plain.json"
    plain.write_text("{}", encoding="utf-8")

    assert hint_seconds(spec, {"name": "m", "est_run_minutes": 5}) == 1200
    assert hint_seconds(per_model, {"name": "m"}) == 1800
    assert hint_seconds(plain, {"name": "m", "est_run_minutes": 5}) == 300
    assert hint_seconds(plain, {"name": "m"}) == DEFAULT_EST_MINUTES * 60
    assert hint_seconds(tmp_path / "missing.json", None) == DEFAULT_EST_MINUTES * 60


def test_resolve_max_hours(monkeypatch):
    monkeypatch.delenv(MAX_HOURS_ENV, raising=False)
    assert resolve_max_hours() is None
    monkeypatch.setenv(MAX_HOURS_ENV, "5.5")
    assert resolve_max_hours() == 5.5
    assert resolve_max_hours(3) == 3.0


def _packed(tmp_path, runs, complete=False, done=3, total=10):
    progress = {
        "done": done,
        "total": total,
        "phases": {"evaluate": {"done": done, "total": total, "arms": {"baseline": {"done": done, "total": total}}}},
    }
    return job_module.Packed(Path(tmp_path) / "part.tar.gz", runs, 2, progress, complete)


def _done(count, status="completed"):
    return [{"status": status}] * count


def test_message_when_the_budget_ran_out(tmp_path):
    budget = SimpleNamespace(max_hours=7.0, exhausted=True)

    lines = job_module.finish_message(_packed(tmp_path, _done(4)), 0, budget)

    assert len(lines) <= 10
    assert "budget reached" in "\n".join(lines)
    assert "4 runs tonight" in lines[0]
    assert any("3 of 10" in line for line in lines)
    assert not any("nights" in line for line in lines)
    assert lines[-1] == f"Please send this file to Vladimir: {(Path(tmp_path) / 'part.tar.gz').resolve()}"


def test_message_when_everything_is_done(tmp_path):
    budget = SimpleNamespace(max_hours=7.0, exhausted=False)

    lines = job_module.finish_message(_packed(tmp_path, _done(2), complete=True, done=10), 0, budget)

    assert len(lines) <= 10
    assert lines[-1] == f"ALL DONE — send this file to Vladimir: {(Path(tmp_path) / 'part.tar.gz').resolve()}"
    assert "budget reached" not in "\n".join(lines)
    assert not any("Run the same command again" in line for line in lines)


def test_message_when_runs_did_not_complete_without_a_budget_stop(tmp_path):
    budget = SimpleNamespace(max_hours=7.0, exhausted=False)

    lines = job_module.finish_message(_packed(tmp_path, _done(3, "error")), 0, budget)

    assert any("did not complete" in line for line in lines)
    assert any("No run completed successfully" in line for line in lines)
    assert any("Run the same command again" in line for line in lines)
    assert lines[-1].startswith("Please send this file to Vladimir:")
    assert len(lines) <= 10


def test_message_when_the_job_was_stopped(tmp_path):
    budget = SimpleNamespace(max_hours=7.0, exhausted=False)

    lines = job_module.finish_message(_packed(tmp_path, _done(1)), 143, budget)

    assert "stopped early (exit code 143)" in lines[0]
    assert any("run the same command again" in line for line in lines)
    assert lines[-1].startswith("Please send this file to Vladimir:")
    assert len(lines) <= 10


def test_later_part_carries_the_frozen_snapshot_and_the_learned_skill_it_uses(tmp_path):
    embedder = FakeEmbedder()
    snapshot = SnapshotSkillSource(write_snapshot(tmp_path / "snapshot.json", [], embedder), embedder)
    built = build_two_parts(tmp_path, snapshot, "base-skill")
    first, second = built["first"], built["second"]

    assert [row["id"] for row in store.read_lines(first / "runs.jsonl")] == [built["learn_id"]]
    assert not (first / "freeze.json").exists()
    assert [row["id"] for row in store.read_lines(second / "runs.jsonl")] == [built["eval_id"]]
    assert json.loads((second / "freeze.json").read_text(encoding="utf-8"))["snapshot_id"] == built["frozen_id"]
    assert [row["id"] for row in store.read_lines(second / "candidate_skills.jsonl")] == [built["candidate_id"]]
    assert [row["trigger"] for row in store.read_lines(second / "skill_decisions.jsonl")] == ["pre_run"]
    assert [row["trigger"] for row in store.read_lines(first / "skill_decisions.jsonl")] == ["entry"]
    assert len(store.read_lines(second / "run_skills.jsonl")) == 1
    assert len(store.read_lines(second / "design_reviews.jsonl")) == 1
    assert [row["id"] for row in store.read_lines(second / "experiments.jsonl")] == [built["experiment_id"]]
