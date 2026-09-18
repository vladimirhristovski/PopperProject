from popper_upgrade.reviewer.gate import ReviewGate


class _FakeRealAgent:
    def __init__(self, proposals):
        self._proposals = iter(proposals)
        self.calls = []
        self.failed_tests = []
        self.existing_tests = []

    def go(self, main_hypothesis, test_results=None, log=None):
        proposal = next(self._proposals)
        self.calls.append((main_hypothesis, test_results, log))
        return proposal

    def add_to_existing_tests(self, test):
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test):
        self.failed_tests.append(test)


class _FakeReviewer:
    def __init__(self, verdicts):
        self._verdicts = iter(verdicts)
        self.calls = []

    def review(self, main_hypothesis, proposal, test_results=None):
        verdict = next(self._verdicts)
        self.calls.append((main_hypothesis, proposal, test_results))
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
    assert real.failed_tests == ["proposal 1"]
    assert len(real.calls) == 2


def test_returns_last_proposal_when_attempts_exhausted():
    real = _FakeRealAgent(["proposal 1", "proposal 2"])
    reviewer = _FakeReviewer([(False, "no"), (False, "still no")])
    gate = ReviewGate(real, reviewer, max_attempts=2)

    result = gate.go("hypothesis", "prior tests", log={})

    assert result == "proposal 2"
    assert real.failed_tests == ["proposal 1", "proposal 2"]


def test_logs_one_reviewer_entry_per_attempt():
    real = _FakeRealAgent(["proposal 1", "proposal 2"])
    reviewer = _FakeReviewer([(False, "no"), (True, "yes")])
    gate = ReviewGate(real, reviewer, max_attempts=3)
    log = {}

    gate.go("hypothesis", "prior tests", log=log)

    assert len(log["reviewer"]) == 2
    assert "rejected" in log["reviewer"][0]
    assert "approved" in log["reviewer"][1]


def test_forwards_add_to_existing_and_failed_tests():
    real = _FakeRealAgent([])
    gate = ReviewGate(real, _FakeReviewer([]), max_attempts=3)

    gate.add_to_existing_tests("test A")
    gate.add_to_failed_tests("test B")

    assert real.existing_tests == ["test A"]
    assert real.failed_tests == ["test B"]
