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


class _FakePopper:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.agent = _FakeAgent()
        self.register_data_calls = []
        self.configure_calls = []
        self.validate_calls = []

    def register_data(self, loader_type, data_path, **kwargs):
        self.register_data_calls.append((loader_type, data_path, kwargs))

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)

    def validate(self, hypothesis):
        self.validate_calls.append(hypothesis)
        return {"parsed_result": {"conclusion": True}, "last_message": "done"}


def test_init_delegates_kwargs_to_popper(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)

    runner = PopperRunner(llm="claude-sonnet-4-5", domain="social science")

    assert runner._popper.init_kwargs == {"llm": "claude-sonnet-4-5", "domain": "social science"}


def test_register_data_and_configure_delegate(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5")

    runner.register_data("custom", "data/", extra=1)
    runner.configure(plot_agent_architecture=False)

    assert runner._popper.register_data_calls == [("custom", "data/", {"extra": 1})]
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

    assert runner.agent is runner._popper.agent


def test_configure_leaves_proposal_agent_untouched_when_skills_disabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", skills_enabled=False)

    runner.configure()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_swaps_in_skill_augmented_agent_when_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    store = object()
    runner = PopperRunner(llm="claude-sonnet-4-5", skills_enabled=True, skill_store=store)
    real_before = runner.agent.test_proposal_agent

    runner.configure()

    wrapped = runner.agent.test_proposal_agent
    assert isinstance(wrapped, SkillAugmentedProposalAgent)
    assert wrapped._real is real_before
    assert wrapped._store is store


def test_configure_leaves_proposal_agent_untouched_when_enabled_without_store(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", skills_enabled=True, skill_store=None)

    runner.configure()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_leaves_proposal_agent_untouched_when_reviewer_disabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", reviewer_enabled=False)

    runner.configure()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_wraps_in_review_gate_when_reviewer_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    reviewer = object()
    runner = PopperRunner(llm="claude-sonnet-4-5", reviewer_enabled=True, reviewer=reviewer, max_review_attempts=5)
    real_before = runner.agent.test_proposal_agent

    runner.configure()

    wrapped = runner.agent.test_proposal_agent
    assert isinstance(wrapped, ReviewGate)
    assert wrapped._real is real_before
    assert wrapped._reviewer is reviewer
    assert wrapped._max_attempts == 5


def test_configure_leaves_proposal_agent_untouched_when_reviewer_enabled_without_reviewer(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    runner = PopperRunner(llm="claude-sonnet-4-5", reviewer_enabled=True, reviewer=None)

    runner.configure()

    assert isinstance(runner.agent.test_proposal_agent, _FakeRealProposalAgent)


def test_configure_wraps_review_gate_around_skill_augmented_agent_when_both_enabled(monkeypatch):
    monkeypatch.setattr(runner_module, "Popper", _FakePopper)
    store = object()
    reviewer = object()
    runner = PopperRunner(
        llm="claude-sonnet-4-5",
        skills_enabled=True,
        skill_store=store,
        reviewer_enabled=True,
        reviewer=reviewer,
    )
    real_before = runner.agent.test_proposal_agent

    runner.configure()

    outer = runner.agent.test_proposal_agent
    assert isinstance(outer, ReviewGate)
    inner = outer._real
    assert isinstance(inner, SkillAugmentedProposalAgent)
    assert inner._real is real_before
    assert inner._store is store
