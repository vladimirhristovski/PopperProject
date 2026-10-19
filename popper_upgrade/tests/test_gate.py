import pytest

from popper_upgrade.reviewer.gate import ReviewGate


class _FakeRealAgent:
    def __init__(self, proposals):
        self._proposals = iter(proposals)
        self.calls = []
        self.failed_tests = []
        self.existing_tests = []
        self.received_test_results = []

    def go(self, main_hypothesis, test_results=None, log=None):
        proposal = next(self._proposals)
        self.calls.append((main_hypothesis, test_results, log))
        self.received_test_results.append(test_results)
        if isinstance(proposal, Exception):
            raise proposal
        return proposal

    def add_to_existing_tests(self, test):
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test):
        self.failed_tests.append(test)


class _FakeReviewer:
    def __init__(self, verdicts):
        self._verdicts = iter(verdicts)
        self.calls = []

    def review(self, main_hypothesis, proposal, prior_designs):
        verdict = next(self._verdicts)
        self.calls.append((main_hypothesis, proposal, prior_designs))
        return verdict


def test_returns_immediately_on_first_approval():
    real = _FakeRealAgent(["proposal 1"])
    reviewer = _FakeReviewer([(True, "looks good")])
    gate = ReviewGate(real, reviewer, max_attempts=3)

    result = gate.go("hypothesis", "prior tests", log={})

    assert result == "proposal 1"
    assert len(real.calls) == 1
    assert real.failed_tests == []


def test_retries_on_rejection_then_returns_approved_proposal():
    real = _FakeRealAgent(["proposal 1", "proposal 2"])
    reviewer = _FakeReviewer([(False, "redundant"), (True, "approved now")])
    gate = ReviewGate(real, reviewer, max_attempts=3)

    result = gate.go("hypothesis", "prior tests", log={})

    assert result == "proposal 2"
    assert real.failed_tests == []
    assert gate.rejected == ["proposal 1"]
    assert len(real.calls) == 2


def test_returns_last_proposal_when_attempts_exhausted():
    real = _FakeRealAgent(["proposal 1", "proposal 2"])
    reviewer = _FakeReviewer([(False, "no"), (False, "still no")])
    gate = ReviewGate(real, reviewer, max_attempts=2)

    result = gate.go("hypothesis", "prior tests", log={})

    assert result == "proposal 2"
    assert real.failed_tests == []
    assert gate.rejected == ["proposal 1", "proposal 2"]


def test_logs_one_reviewer_entry_per_attempt():
    real = _FakeRealAgent(["proposal 1", "proposal 2"])
    reviewer = _FakeReviewer([(False, "no"), (True, "yes")])
    gate = ReviewGate(real, reviewer, max_attempts=3)
    log = {}

    gate.go("hypothesis", "prior tests", log=log)

    assert len(log["reviewer"]) == 2
    assert "rejected" in log["reviewer"][0]
    assert "approved" in log["reviewer"][1]


class _FlakyReviewer:
    def __init__(self, results):
        self._results = iter(results)
        self.calls = []

    def review(self, main_hypothesis, proposal, prior_designs):
        self.calls.append((main_hypothesis, proposal, prior_designs))
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


def test_treats_reviewer_exception_as_rejection_and_retries():
    real = _FakeRealAgent(["proposal 1", "proposal 2"])
    reviewer = _FlakyReviewer([ValueError("malformed JSON from model"), (True, "approved now")])
    gate = ReviewGate(real, reviewer, max_attempts=3)
    log = {}

    result = gate.go("hypothesis", "prior tests", log=log)

    assert result == "proposal 2"
    assert gate.rejected == ["proposal 1"]
    assert "Reviewer call failed" in log["reviewer"][0]
    assert "malformed JSON from model" in log["reviewer"][0]


def test_treats_proposal_generation_exception_as_retryable():
    real = _FakeRealAgent([ValueError("malformed structured output"), "proposal 2"])
    reviewer = _FakeReviewer([(True, "approved")])
    gate = ReviewGate(real, reviewer, max_attempts=3)
    log = {}

    result = gate.go("hypothesis", "prior tests", log=log)

    assert result == "proposal 2"
    assert len(reviewer.calls) == 1
    assert "Proposal generation failed" in log["reviewer"][0]
    assert "malformed structured output" in log["reviewer"][0]


def test_reraises_last_error_when_proposal_generation_never_succeeds():
    real = _FakeRealAgent([ValueError("first failure"), ValueError("second failure")])
    gate = ReviewGate(real, _FakeReviewer([]), max_attempts=2)

    with pytest.raises(ValueError, match="second failure"):
        gate.go("hypothesis", "prior tests", log={})


def test_forwards_add_to_existing_and_failed_tests():
    real = _FakeRealAgent([])
    gate = ReviewGate(real, _FakeReviewer([]), max_attempts=3)

    gate.add_to_existing_tests("test A")
    gate.add_to_failed_tests("test B")

    assert real.existing_tests == ["test A"]
    assert real.failed_tests == ["test B"]


def test_gate_exposes_real_failed_tests():
    real = _FakeRealAgent([])

    assert ReviewGate(real, _FakeReviewer([])).failed_tests is real.failed_tests


def test_rejection_does_not_consume_popper_failed_tests():
    real = _FakeRealAgent(["P1", "P2"])
    gate = ReviewGate(real, _FakeReviewer([(False, "no"), (True, "yes")]))

    assert gate.go("H", "No Implemented Falsification Test Yet.", {}) == "P2"
    assert real.failed_tests == []
    assert gate.rejected == ["P1"]


def test_rejections_are_fed_back_into_prompt():
    real = _FakeRealAgent(["P1", "P2"])

    ReviewGate(real, _FakeReviewer([(False, "no"), (True, "yes")])).go("H", "base", {})

    assert real.received_test_results[0] == "base"
    assert "- P1" in real.received_test_results[1]
    assert real.received_test_results[1].startswith("base")


def test_reviewer_receives_existing_failed_and_rejected_designs():
    real = _FakeRealAgent(["P1", "P2"])
    real.existing_tests = ["E"]
    real.failed_tests = ["F"]
    reviewer = _FakeReviewer([(False, "no"), (True, "yes")])

    ReviewGate(real, reviewer).go("H", "base", {})

    assert reviewer.calls[0][2] == ["E", "F"]
    assert reviewer.calls[1][2] == ["E", "F", "P1"]


def test_all_rejected_fails_open_and_counts_override():
    real = _FakeRealAgent(["P1", "P2", "P3"])
    gate = ReviewGate(real, _FakeReviewer([(False, "no")] * 3))

    assert gate.go("H", "base", {}) == "P3"
    assert gate.overrides == 1
    assert gate.reviews[-1]["overridden"] is True
    assert [r["attempt"] for r in gate.reviews] == [1, 2, 3]
    assert real.failed_tests == []


def test_review_round_increments_per_go():
    real = _FakeRealAgent(["P1", "P2"])
    gate = ReviewGate(real, _FakeReviewer([(True, "yes"), (True, "yes")]))

    gate.go("H", "base", {})
    gate.go("H", "base", {})

    assert [(r["round"], r["attempt"], r["approved"], r["overridden"]) for r in gate.reviews] == [
        (1, 1, True, False),
        (2, 1, True, False),
    ]
