from popper_upgrade.skills.proposal_agent import SkillAugmentedProposalAgent
from popper_upgrade.skills.schema import Skill


class _FakeRealAgent:
    def __init__(self):
        self.calls = []
        self.existing_tests = []
        self.failed_tests = []

    def go(self, main_hypothesis, test_results=None, log=None):
        self.calls.append((main_hypothesis, test_results, log))
        return "proposed test text"

    def add_to_existing_tests(self, test):
        self.existing_tests.append(test)

    def add_to_failed_tests(self, test):
        self.failed_tests.append(test)


class _FakeStore:
    def __init__(self, skills):
        self._skills = skills
        self.queries = []

    def retrieve(self, query, k=3):
        self.queries.append((query, k))
        return self._skills


def _skill():
    return Skill(
        id="s1",
        applicability="gender bias in occupation associations",
        design_template="pronoun-occupation co-occurrence test",
        preconditions="social science",
        domain="social science",
        source_hypothesis="gender bias in occupation associations",
        times_used=1,
        times_passed=1,
        created_at="2026-09-18T00:00:00+00:00",
    )


def test_injects_retrieved_skills_into_test_results():
    real = _FakeRealAgent()
    store = _FakeStore([_skill()])
    wrapper = SkillAugmentedProposalAgent(real, store)

    result = wrapper.go("some hypothesis", "No Implemented Falsification Test Yet.", log={})

    assert result == "proposed test text"
    called_hypothesis, called_test_results, _ = real.calls[0]
    assert called_hypothesis == "some hypothesis"
    assert "pronoun-occupation co-occurrence test" in called_test_results
    assert "not run in this session" in called_test_results
    assert "No Implemented Falsification Test Yet." in called_test_results


def test_passes_through_unchanged_when_store_returns_nothing():
    real = _FakeRealAgent()
    store = _FakeStore([])
    wrapper = SkillAugmentedProposalAgent(real, store)

    wrapper.go("some hypothesis", "round 1 text", log={})

    assert real.calls[0] == ("some hypothesis", "round 1 text", {})


def test_forwards_add_to_existing_and_failed_tests():
    real = _FakeRealAgent()
    wrapper = SkillAugmentedProposalAgent(real, _FakeStore([]))

    wrapper.add_to_existing_tests("test A")
    wrapper.add_to_failed_tests("test B")

    assert real.existing_tests == ["test A"]
    assert real.failed_tests == ["test B"]
