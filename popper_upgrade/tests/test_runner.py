import os

import pytest

import popper_upgrade.runner as runner_module
from popper_upgrade.reviewer.gate import ReviewGate
from popper_upgrade.runner import PopperRunner
from popper_upgrade.skills.proposal_agent import SkillAugmentedProposalAgent
from popper_upgrade.trajectory import Trajectory


class _FakeRealProposalAgent:
    pass


class _FakeAgent:
    def __init__(self):
        self.log = {"designer": [], "executor": [], "relevance_checker": [], "summarizer": [], "sequential_testing": []}
        self.tracked_tests = []
        self.tracked_stat = []
        self.res = True
        self.res_stat = 42.0
        self.test_proposal_agent = _FakeRealProposalAgent()


class _FakeLoader:
    table_dict = {"df_t": "frame"}


class _FakePopper:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.agent = _FakeAgent()
        self.data_loader = _FakeLoader()
        self.register_data_calls = []
        self.configure_calls = []
        self.validate_calls = []

    def register_data(self, data_path, loader_type="bio", **kwargs):
        self.register_data_calls.append((data_path, loader_type, kwargs))

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)

    def validate(self, hypothesis):
        self.validate_calls.append(hypothesis)
        return {"parsed_result": {"conclusion": True}, "last_message": "done"}


def test_init_delegates_kwargs_to_popper(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)

    runner = PopperRunner(llm="claude-sonnet-4-5", domain="social science")
    runner.prepare()

    assert runner._popper.init_kwargs == {"llm": "claude-sonnet-4-5", "domain": "social science"}


def test_init_sets_openai_api_key_when_server_port_given(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    PopperRunner(llm="llama3.1:8b", is_locally_served=True, server_port=11434, api_key="EMPTY")

    assert os.environ["OPENAI_API_KEY"] == "EMPTY"


def test_init_does_not_touch_api_key_without_server_port(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    PopperRunner(llm="claude-sonnet-4-5")

    assert "OPENAI_API_KEY" not in os.environ


def test_register_data_and_configure_delegate(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5")

    runner.register_data("data/", loader_type="custom", extra=1)
    runner.configure(plot_agent_architecture=False)
    runner.prepare()

    assert runner._popper.register_data_calls == [("data/", "custom", {"extra": 1})]
    assert runner._popper.configure_calls == [{"plot_agent_architecture": False}]


def test_validate_returns_trajectory_built_from_agent_state(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5")

    trajectory = runner.validate("some hypothesis")

    assert runner._popper.validate_calls == ["some hypothesis"]
    assert isinstance(trajectory, Trajectory)
    assert trajectory.res is True
    assert trajectory.res_stat == 42.0
    assert trajectory.parsed_result == {"conclusion": True}
    assert trajectory.last_message == "done"


def test_agent_property_exposes_underlying_popper_agent(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5")

    assert runner.agent is None
    runner.prepare()
    assert runner.agent is runner._popper.agent
    runner.validate("H")
    assert runner.agent is runner._popper.agent


def test_configure_leaves_proposal_agent_untouched_when_skills_disabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", skills_enabled=False)

    runner.configure()
    runner.prepare()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_swaps_in_skill_augmented_agent_when_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    store = object()
    runner = PopperRunner(llm="claude-sonnet-4-5", skills_enabled=True, skill_source=store)
    runner.configure()
    runner.prepare()

    wrapped = runner.agent.test_proposal_agent
    assert isinstance(wrapped, SkillAugmentedProposalAgent)
    assert isinstance(wrapped._real, _FakeRealProposalAgent)
    assert wrapped._source is store


def test_configure_leaves_proposal_agent_untouched_when_enabled_without_store(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", skills_enabled=True, skill_source=None)

    runner.configure()
    runner.prepare()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_leaves_proposal_agent_untouched_when_reviewer_disabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", reviewer_enabled=False)

    runner.configure()
    runner.prepare()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_wraps_in_review_gate_when_reviewer_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    reviewer = object()
    runner = PopperRunner(llm="claude-sonnet-4-5", reviewer_enabled=True, reviewer=reviewer, max_review_attempts=5)
    runner.configure()
    runner.prepare()

    wrapped = runner.agent.test_proposal_agent
    assert isinstance(wrapped, ReviewGate)
    assert isinstance(wrapped._real, _FakeRealProposalAgent)
    assert wrapped._reviewer is reviewer
    assert wrapped._max_attempts == 5


def test_configure_leaves_proposal_agent_untouched_when_reviewer_enabled_without_reviewer(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", reviewer_enabled=True, reviewer=None)

    runner.configure()
    runner.prepare()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_wraps_review_gate_around_skill_augmented_agent_when_both_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    store = object()
    reviewer = object()
    runner = PopperRunner(
        llm="claude-sonnet-4-5",
        skills_enabled=True,
        skill_source=store,
        reviewer_enabled=True,
        reviewer=reviewer,
    )
    runner.configure()
    runner.prepare()

    outer = runner.agent.test_proposal_agent
    assert isinstance(outer, ReviewGate)
    inner = outer._real
    assert isinstance(inner, SkillAugmentedProposalAgent)
    assert isinstance(inner._real, _FakeRealProposalAgent)
    assert inner._source is store


def test_validate_builds_fresh_popper_each_time(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="m")
    runner.register_data("data", loader_type="custom")
    runner.configure(relevance_checker=True, alpha=0.1)

    runner.validate("H1")
    first = runner._popper
    runner.validate("H2")

    assert runner._popper is not first
    assert runner._popper.register_data_calls == [("data", "custom", {})]
    assert runner._popper.configure_calls == [{"relevance_checker": True, "alpha": 0.1}]
    assert runner._popper.validate_calls == ["H2"]


def test_configure_rejects_relevance_checker_off(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)

    with pytest.raises(ValueError, match="relevance_checker must be True"):
        PopperRunner(llm="m").configure(relevance_checker=False)


def test_hooks_are_reapplied_to_each_fresh_agent(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="m", reviewer_enabled=True, reviewer=object())

    runner.validate("H1")
    first_gate = runner.agent.test_proposal_agent
    runner.validate("H2")

    assert isinstance(runner.agent.test_proposal_agent, ReviewGate)
    assert runner.agent.test_proposal_agent is not first_gate
    assert isinstance(runner.agent.test_proposal_agent._real, _FakeRealProposalAgent)


def test_runner_exposes_wrappers(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    source = object()
    runner = PopperRunner(
        llm="m",
        skills_enabled=True,
        skill_source=source,
        allowed_skill_ids={"s1"},
        reviewer_enabled=True,
        reviewer=object(),
    )
    assert runner.skill_agent is None
    assert runner.review_gate is None

    runner.prepare()

    assert runner.review_gate is runner.agent.test_proposal_agent
    assert runner.skill_agent is runner.review_gate._real
    assert runner.skill_agent._allowed_ids == {"s1"}


def test_runner_wrappers_are_none_when_flags_off(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="m")

    runner.prepare()

    assert runner.skill_agent is None
    assert runner.review_gate is None


def test_allowed_skill_ids_can_change_between_prepare_and_validate(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="m", skills_enabled=True, skill_source=object(), allowed_skill_ids=set())

    runner.prepare()
    runner.allowed_skill_ids = {"s2"}
    runner.validate("H")

    assert runner.skill_agent._allowed_ids == {"s2"}


def test_data_tables_reads_loader_after_prepare(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="m")

    assert runner.data_tables() == {}
    runner.prepare()

    assert runner.data_tables() == {"df_t": "frame"}
