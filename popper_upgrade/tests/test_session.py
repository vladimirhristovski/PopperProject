import json

import pytest

import popper_upgrade.runner as runner_module
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.eval.controls import WrongDomainSkillSource
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.session import RunSession, RunSpec
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import (
    FakeCurator,
    FakeEmbedder,
    FakeExtractor,
    FakeLoader,
    FakePopper,
    FakeSink,
    write_snapshot,
)

HYPOTHESIS = "men are paid more than women"


def _skill(skill_id, applicability, approved=True, source_hypothesis="unrelated source"):
    return Skill(
        id=skill_id,
        applicability=applicability,
        design_template="compare group means",
        preconditions="two groups",
        domain="social science",
        source_hypothesis=source_hypothesis,
        approved=approved,
    )


class _Reviewer:
    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    def review(self, main_hypothesis, proposal, prior_designs):
        return self._verdicts.pop(0)


class _BrokenExtractor(FakeExtractor):
    def extract(self, *args):
        raise RuntimeError("extractor down")


def _source(tmp_path, skills):
    embedder = FakeEmbedder()
    return SnapshotSkillSource(write_snapshot(tmp_path / "snap.json", skills, embedder), embedder)


def _spec(**overrides):
    fields = {
        "hypothesis": HYPOTHESIS,
        "split": "train",
        "arm": "skills",
        "model": {"name": "fake-model"},
        "popper_kwargs": {"llm": "fake-model", "domain": "social science"},
        "register_kwargs": {"data_path": "data/winobias", "loader_type": "custom"},
        "configure_kwargs": {"alpha": 0.1, "max_num_of_tests": 3, "max_retry": 1, "time_limit": 1},
        "skills_enabled": True,
    }
    fields.update(overrides)
    return RunSpec(**fields)


@pytest.fixture(autouse=True)
def fake_popper(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", FakePopper)
    monkeypatch.setattr(FakePopper, "failure", None)


def _session(tmp_path, spec, skills, curator=None, extractor=None, reviewer=None, **kwargs):
    sink = FakeSink()
    session = RunSession(
        spec,
        _source(tmp_path, skills),
        sink,
        curator or FakeCurator(),
        reviewer,
        extractor,
        tmp_path / "trajectories",
        **kwargs,
    )
    return session, sink


def test_session_records_curation_injection_reviews_and_result(tmp_path):
    kept = _skill("s1", HYPOTHESIS)
    dropped = _skill("s2", "something else")
    candidate = _skill("new", "a general strategy", approved=None)
    usage = []
    session, sink = _session(
        tmp_path,
        _spec(reviewer_enabled=True),
        [kept, dropped],
        curator=FakeCurator(rejected={"s2"}),
        extractor=FakeExtractor([candidate]),
        reviewer=_Reviewer([(False, "weak"), (True, "ok")]),
        usage_recorder=lambda skill_id, passed: usage.append((skill_id, passed)),
        prompt_version_resolver=lambda role, text: f"pv-{role}",
    )

    run_id = session.run()

    assert run_id == "run-1"
    assert {"git_sha", "popper_version", "pins_hash"} <= set(sink.config)
    assert sink.config["model_name"] == "fake-model"
    assert sink.config["datasets"] == ["winobias"]
    curated = {(r["skill_id"], r["role"]) for r in sink.run_skills if r["role"] != "injected"}
    assert curated == {("s1", "curator_approved"), ("s2", "curator_rejected")}
    injected = [r for r in sink.run_skills if r["role"] == "injected"]
    assert {r["skill_id"] for r in injected} == {"s1"}
    pre_run = {d["skill_id"]: d for d in sink.decisions if d["trigger"] == "pre_run"}
    assert pre_run["s1"]["verdict"] == "approve"
    assert pre_run["s2"]["verdict"] == "reject"
    assert pre_run["s2"]["model"] == "fake-curator"
    assert pre_run["s2"]["prompt_version_id"] == "pv-curator"
    assert [r["approved"] for r in sink.design_reviews] == [False, True]
    assert sink.result["status"] == "completed"
    assert sink.result["res"] is True
    assert sink.result["n_tests"] == 1
    assert sink.result["reviewer_rejections"] == 1
    assert sink.result["reviewer_overrides"] == 0
    trajectory = json.loads((tmp_path / "trajectories" / "run-1.json").read_text(encoding="utf-8"))
    assert trajectory["tracked_tests"] == ["design one"]
    assert sink.result["trajectory_path"].endswith("run-1.json")
    [entry] = [d for d in sink.decisions if d["trigger"] == "entry"]
    assert entry["skill_id"] == sink.candidates[0].id
    assert entry["verdict"] == "approve"
    assert usage == [("s1", True)]


def test_extractor_gets_designs_and_run_id_only(tmp_path):
    extractor = FakeExtractor([_skill("new", "a general strategy", approved=None)])
    session, _ = _session(tmp_path, _spec(), [], extractor=extractor)

    session.run()

    designs, hypothesis, domain, rejected, run_id = extractor.calls[0]
    assert designs == ["design one"]
    assert (hypothesis, domain, rejected, run_id) == (HYPOTHESIS, "social science", [], "run-1")


def test_zero_approved_skills_runs_without_injection(tmp_path):
    curator = FakeCurator()
    session, sink = _session(tmp_path, _spec(), [_skill("s1", HYPOTHESIS, approved=False)], curator=curator)

    session.run()

    assert curator.run_calls[0][0] == []
    assert [r for r in sink.run_skills if r["role"] == "injected"] == []
    assert sink.result["status"] == "completed"


def test_validate_error_marks_run_error_and_keeps_curation(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    session, sink = _session(tmp_path, _spec(), [_skill("s1", HYPOTHESIS)])

    run_id = session.run()

    assert run_id == "run-1"
    assert sink.result["status"] == "error"
    assert "model down" in sink.result["error"]
    assert [r["role"] for r in sink.run_skills] == ["curator_approved"]
    assert len(sink.decisions) == 1


def test_validate_error_reraises_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    session, sink = _session(tmp_path, _spec(), [], raise_on_error=True)

    with pytest.raises(RuntimeError, match="model down"):
        session.run()

    assert sink.result["status"] == "error"


def test_non_exception_interruption_marks_error_and_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", KeyboardInterrupt("stop"))
    session, sink = _session(tmp_path, _spec(), [])

    with pytest.raises(KeyboardInterrupt):
        session.run()

    assert sink.result["status"] == "error"
    assert "KeyboardInterrupt" in sink.result["error"]


def test_human_override_is_recorded_per_run(tmp_path):
    session, sink = _session(
        tmp_path,
        _spec(human_review=True, launched_by="user-1"),
        [_skill("s1", HYPOTHESIS)],
        curator=FakeCurator(rejected={"s1"}),
        overrides_provider=lambda result: {"s1": True},
    )

    session.run()

    roles = {r["role"] for r in sink.run_skills}
    assert {"curator_rejected", "human_override_approve", "injected"} <= roles
    [human] = [d for d in sink.decisions if d["decider"] == "human"]
    assert human["verdict"] == "approve"
    assert human["user_id"] == "user-1"
    assert human["trigger"] == "pre_run"


def test_overrides_provider_ignored_without_human_review(tmp_path):
    session, sink = _session(
        tmp_path,
        _spec(human_review=False),
        [_skill("s1", HYPOTHESIS)],
        curator=FakeCurator(rejected={"s1"}),
        overrides_provider=lambda result: {"s1": True},
    )

    session.run()

    assert not [r for r in sink.run_skills if r["role"] in ("human_override_approve", "injected")]


def test_curator_always_runs_even_without_human(tmp_path):
    curator = FakeCurator()
    session, sink = _session(tmp_path, _spec(human_review=False), [_skill("s1", "x")], curator=curator)

    session.run()

    assert len(curator.run_calls) == 1
    assert {d["decider"] for d in sink.decisions} == {"curator"}


def test_curator_skipped_when_skills_disabled(tmp_path):
    curator = FakeCurator()
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [_skill("s1", "x")], curator=curator)

    session.run()

    assert curator.run_calls == []
    assert sink.run_skills == []


def test_skills_enabled_requires_curator(tmp_path):
    with pytest.raises(ValueError, match="requires a curator"):
        RunSession(_spec(), _source(tmp_path, []), FakeSink(), None, None, None, tmp_path)


@pytest.mark.parametrize(
    ("split", "learn", "learns"),
    [("test", True, False), ("none", False, False), ("none", True, True), ("train", False, True)],
)
def test_learning_follows_split(tmp_path, split, learn, learns):
    extractor = FakeExtractor([_skill("new", "a general strategy", approved=None)])
    usage = []
    session, sink = _session(
        tmp_path,
        _spec(split=split, learn=learn),
        [_skill("s1", HYPOTHESIS)],
        extractor=extractor,
        usage_recorder=lambda skill_id, passed: usage.append(skill_id),
    )

    session.run()

    assert bool(extractor.calls) is learns
    assert bool(sink.candidates) is learns
    assert bool(usage) is (split == "train")


def test_entry_review_error_leaves_candidate_pending(tmp_path):
    extractor = FakeExtractor([_skill("new", "a general strategy", approved=None)])
    session, sink = _session(
        tmp_path, _spec(), [], curator=FakeCurator(candidate_error=RuntimeError("down")), extractor=extractor
    )

    session.run()

    assert len(sink.candidates) == 1
    assert [d for d in sink.decisions if d["trigger"] == "entry"] == []
    assert sink.result["status"] == "completed"


def test_run_without_a_statistic_completes_with_a_null_res_stat(tmp_path, monkeypatch):
    class NoStatisticPopper(FakePopper):
        def validate(self, hypothesis):
            result = super().validate(hypothesis)
            self.agent.res = False
            self.agent.res_stat = None
            return result

    monkeypatch.setattr(runner_module, "Popper", NoStatisticPopper)
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [])

    session.run()

    assert sink.result["status"] == "completed"
    assert sink.result["res"] is False
    assert sink.result["res_stat"] is None


def test_extractor_failure_does_not_lose_the_run(tmp_path):
    session, sink = _session(tmp_path, _spec(), [], extractor=_BrokenExtractor())

    session.run()

    assert sink.result["status"] == "completed"
    assert "extractor down" in sink.result["error"]


def _usage_popper(counter, usage):
    class UsagePopper(FakePopper):
        def validate(self, hypothesis):
            if usage is not None:
                counter.add(usage)
            return super().validate(hypothesis)

    return UsagePopper


def test_tokens_are_counted_per_run(tmp_path, monkeypatch):
    counter = UsageCounter()
    counter.add({"prompt_tokens": 999, "completion_tokens": 999})
    monkeypatch.setattr(runner_module, "Popper", _usage_popper(counter, {"prompt_tokens": 12, "completion_tokens": 3}))
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)

    session.run()

    assert (sink.result["tokens_in"], sink.result["tokens_out"]) == (12, 3)


def test_tokens_stay_absent_without_usage(tmp_path, monkeypatch):
    counter = UsageCounter()
    monkeypatch.setattr(runner_module, "Popper", _usage_popper(counter, None))
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)

    session.run()

    assert "tokens_in" not in sink.result
    assert "tokens_out" not in sink.result


def test_tokens_are_recorded_on_error_runs(tmp_path, monkeypatch):
    counter = UsageCounter()
    popper = _usage_popper(counter, {"prompt_tokens": 7, "completion_tokens": 1})
    monkeypatch.setattr(runner_module, "Popper", popper)
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)

    session.run()

    assert sink.result["status"] == "error"
    assert (sink.result["tokens_in"], sink.result["tokens_out"]) == (7, 1)


def test_run_cost_is_recorded_per_run_from_priced_usage(tmp_path, monkeypatch):
    counter = UsageCounter({"input": 2, "cache_read": 0.1, "cache_write": 2.5, "output": 10})
    counter.add({"prompt_tokens": 999, "completion_tokens": 999})
    usage = {
        "prompt_tokens": 1300,
        "completion_tokens": 50,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 100,
    }
    monkeypatch.setattr(runner_module, "Popper", _usage_popper(counter, usage))
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)

    session.run()

    expected = (200 * 2 + 1000 * 0.1 + 100 * 2.5 + 50 * 10) / 1e6
    assert sink.result["cost_usd"] == pytest.approx(expected)


def test_run_cost_is_recorded_on_error_runs_and_absent_without_pricing(tmp_path, monkeypatch):
    priced = UsageCounter({"input": 2, "output": 10})
    spend = {"prompt_tokens": 1000, "completion_tokens": 100}
    monkeypatch.setattr(runner_module, "Popper", _usage_popper(priced, spend))
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=priced)

    session.run()

    assert sink.result["status"] == "error"
    assert sink.result["cost_usd"] == pytest.approx(3000 / 1e6)

    plain = UsageCounter()
    monkeypatch.setattr(runner_module, "Popper", _usage_popper(plain, spend))
    monkeypatch.setattr(FakePopper, "failure", None)
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=plain)

    session.run()

    assert "cost_usd" not in sink.result


def test_on_start_receives_session_with_runner(tmp_path):
    started = []
    session, _ = _session(tmp_path, _spec(skills_enabled=False), [], on_start=lambda s: started.append(s.runner))

    session.run()

    assert started == [session.runner]
    assert session.runner is not None


def test_reviewer_without_skills(tmp_path):
    session, sink = _session(
        tmp_path, _spec(skills_enabled=False, reviewer_enabled=True), [], reviewer=_Reviewer([(True, "ok")])
    )

    session.run()

    assert [r["approved"] for r in sink.design_reviews] == [True]
    assert sink.result["reviewer_rejections"] == 0


def test_experiment_fields_are_recorded_and_the_factory_supplies_the_data(tmp_path):
    built = []

    def factory():
        built.append(1)
        return FakeLoader()

    spec = _spec(
        split="test",
        register_kwargs={},
        data_loader_factory=factory,
        answer=False,
        task="gender_bias",
        replicate=3,
        permuted=True,
        permute_seed=42,
        datasets=["bbq"],
    )
    session, sink = _session(tmp_path, spec, [_skill("s1", HYPOTHESIS)])

    session.run()

    assert built == [1, 1]
    assert (sink.config["answer"], sink.config["task"], sink.config["replicate"]) == (False, "gender_bias", 3)
    assert (sink.config["permuted"], sink.config["permute_seed"], sink.config["datasets"]) == (True, 42, ["bbq"])
    assert "control_degenerate" not in sink.result
    assert sink.candidates == []


def test_degenerate_wrong_domain_control_is_flagged_and_injects_nothing(tmp_path):
    base = _source(tmp_path, [_skill("s1", HYPOTHESIS)])
    sink = FakeSink()
    session = RunSession(
        _spec(split="test", arm="wrong_domain_skills"),
        WrongDomainSkillSource(base, "social science"),
        sink,
        FakeCurator(),
        None,
        None,
        tmp_path / "trajectories",
    )

    session.run()

    assert sink.result["control_degenerate"] is True
    assert [r for r in sink.run_skills if r["role"] == "injected"] == []


def test_degenerate_control_is_flagged_on_errored_runs_too(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    base = _source(tmp_path, [_skill("s1", HYPOTHESIS)])
    sink = FakeSink()
    session = RunSession(
        _spec(split="test", arm="wrong_domain_skills"),
        WrongDomainSkillSource(base, "social science"),
        sink,
        FakeCurator(),
        None,
        None,
        tmp_path / "trajectories",
    )

    session.run()

    assert sink.result["status"] == "error"
    assert sink.result["control_degenerate"] is True
