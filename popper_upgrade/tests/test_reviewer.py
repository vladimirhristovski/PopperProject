import os

import popper_upgrade.reviewer.reviewer as reviewer_module
from popper_upgrade.reviewer.gate import ReviewGate
from popper_upgrade.reviewer.reviewer import Reviewer, ReviewVerdict


class _FakeChain:
    def __init__(self, output):
        self._output = output
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        return self._output


class _FakeModel:
    def __init__(self, verdict):
        self._verdict = verdict
        self.chain = None

    def with_structured_output(self, schema):
        self.chain = _FakeChain(self._verdict)
        return self.chain


class _FakePromptTemplate:
    def __or__(self, other):
        return other


def _patch_reviewer_dependencies(monkeypatch, model):
    monkeypatch.setattr(reviewer_module, "get_llm", lambda llm, port=None, api_key="EMPTY": model)
    monkeypatch.setattr(
        reviewer_module.ChatPromptTemplate,
        "from_messages",
        staticmethod(lambda messages: _FakePromptTemplate()),
    )


def test_review_returns_approved_and_reasoning(monkeypatch):
    verdict = ReviewVerdict(approved=True, reasoning="satisfies all four criteria")
    model = _FakeModel(verdict)
    _patch_reviewer_dependencies(monkeypatch, model)

    reviewer = Reviewer(llm="claude-sonnet-4-5")
    approved, reasoning = reviewer.review("main hypothesis", "proposed test", ["prior test A"])

    assert approved is True
    assert reasoning == "satisfies all four criteria"
    call = model.chain.calls[0]
    assert "main hypothesis" in call["input"]
    assert "proposed test" in call["input"]
    assert "prior test A" in call["input"]


def test_review_defaults_test_results_text_when_none(monkeypatch):
    verdict = ReviewVerdict(approved=False, reasoning="redundant")
    model = _FakeModel(verdict)
    _patch_reviewer_dependencies(monkeypatch, model)

    reviewer = Reviewer(llm="claude-sonnet-4-5")
    approved, _ = reviewer.review("main hypothesis", "proposed test", [])

    assert approved is False
    assert "None yet." in model.chain.calls[0]["input"]


def test_init_sets_openai_api_key_when_port_given(monkeypatch):
    model = _FakeModel(ReviewVerdict(approved=True, reasoning="ok"))
    _patch_reviewer_dependencies(monkeypatch, model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    Reviewer(llm="llama3.1:8b", port=11434, api_key="EMPTY")

    assert os.environ["OPENAI_API_KEY"] == "EMPTY"


def test_init_does_not_touch_api_key_without_port(monkeypatch):
    model = _FakeModel(ReviewVerdict(approved=True, reasoning="ok"))
    _patch_reviewer_dependencies(monkeypatch, model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    Reviewer(llm="claude-sonnet-4-5")

    assert "OPENAI_API_KEY" not in os.environ


class _FakeRealAgent:
    def __init__(self, proposals):
        self._proposals = iter(proposals)
        self.existing_tests = []
        self.failed_tests = []

    def go(self, main_hypothesis, test_results=None, log=None):
        return next(self._proposals)


def test_reviewer_never_receives_test_statistics(monkeypatch):
    model = _FakeModel(ReviewVerdict(approved=True, reasoning="ok"))
    _patch_reviewer_dependencies(monkeypatch, model)
    gate = ReviewGate(_FakeRealAgent(["P"]), Reviewer("claude-sonnet-4-5"))

    gate.go("H", "------- Round 1 ------- \n Falsification Test: T \n test statistics: 0.003", {})

    sent = model.chain.calls[0]["input"]
    assert "test statistics" not in sent
    assert "0.003" not in sent
    assert "Falsification Test: T" not in sent


def test_review_lists_prior_designs_under_session_heading(monkeypatch):
    model = _FakeModel(ReviewVerdict(approved=True, reasoning="ok"))
    _patch_reviewer_dependencies(monkeypatch, model)

    Reviewer("claude-sonnet-4-5").review("H", "P", ["D1", "D2"])

    sent = model.chain.calls[0]["input"]
    assert "Designs already proposed in this session" in sent
    assert "- D1\n- D2" in sent
