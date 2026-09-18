import popper_upgrade.reviewer.reviewer as reviewer_module
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
    approved, reasoning = reviewer.review("main hypothesis", "proposed test", "prior test A")

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
    approved, reasoning = reviewer.review("main hypothesis", "proposed test")

    assert approved is False
    assert "None yet." in model.chain.calls[0]["input"]
