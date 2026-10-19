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
    def __init__(self, skills, similarities=None):
        self._skills = skills
        self._similarities = similarities or {}
        self.queries = []

    def retrieve_scored(self, query, k, allowed_ids=None, min_similarity=0.0):
        self.queries.append((query, k, allowed_ids, min_similarity))
        scored = [(skill, self._similarities.get(skill.id, 0.9)) for skill in self._skills]
        scored = [
            (skill, similarity)
            for skill, similarity in scored
            if (allowed_ids is None or skill.id in allowed_ids) and similarity >= min_similarity
        ]
        return scored[:k]


def _skill(skill_id="s1", design_template="pronoun-occupation co-occurrence test"):
    return Skill(
        id=skill_id,
        applicability="gender bias in occupation associations",
        design_template=design_template,
        preconditions="social science",
        domain="social science",
        source_hypothesis="gender bias in occupation associations",
    )


def test_injects_retrieved_skills_into_test_results():
    real = _FakeRealAgent()
    store = _FakeStore([_skill()])
    wrapper = SkillAugmentedProposalAgent(real, store, None)

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
    wrapper = SkillAugmentedProposalAgent(real, store, None)

    wrapper.go("some hypothesis", "round 1 text", log={})

    assert real.calls[0] == ("some hypothesis", "round 1 text", {})


def test_forwards_add_to_existing_and_failed_tests():
    real = _FakeRealAgent()
    wrapper = SkillAugmentedProposalAgent(real, _FakeStore([]), None)

    wrapper.add_to_existing_tests("test A")
    wrapper.add_to_failed_tests("test B")

    assert real.existing_tests == ["test A"]
    assert real.failed_tests == ["test B"]


def test_skill_agent_exposes_real_failed_tests():
    real = _FakeRealAgent()
    real.failed_tests = ["a"]
    real.existing_tests = ["b"]
    agent = SkillAugmentedProposalAgent(real, _FakeStore([]), None)

    assert agent.failed_tests is real.failed_tests
    assert agent.existing_tests is real.existing_tests


def test_skill_agent_records_injected_ids():
    agent = SkillAugmentedProposalAgent(_FakeRealAgent(), _FakeStore([_skill("s1"), _skill("s2")]), None)

    agent.go("H", "No Implemented Falsification Test Yet.", {})

    assert agent.last_retrieved_ids == ["s1", "s2"]


def test_skill_agent_resets_injected_ids_when_nothing_retrieved():
    store = _FakeStore([_skill("s1")])
    agent = SkillAugmentedProposalAgent(_FakeRealAgent(), store, None)
    agent.go("H", "base", {})
    store._skills = []

    agent.go("H", "base", {})

    assert agent.last_retrieved_ids == []


def test_skill_agent_records_similarities():
    agent = SkillAugmentedProposalAgent(_FakeRealAgent(), _FakeStore([_skill("s1"), _skill("s2")]), {"s1", "s2"})

    agent.go("H", "No Implemented Falsification Test Yet.", {})

    assert agent.last_retrieved == [("s1", 0.9), ("s2", 0.9)]
    assert agent._source.queries == [("H", 2, {"s1", "s2"}, 0.35)]


def test_only_allowed_ids_injected():
    real = _FakeRealAgent()
    agent = SkillAugmentedProposalAgent(real, _FakeStore([_skill("s1"), _skill("s2")]), {"s2"})

    agent.go("H", "base", {})

    assert agent.last_retrieved_ids == ["s2"]
    assert "Suggested test" in real.calls[0][1]


def test_empty_allowed_ids_injects_nothing():
    real = _FakeRealAgent()
    agent = SkillAugmentedProposalAgent(real, _FakeStore([_skill("s1")]), set())

    agent.go("H", "base", {})

    assert agent.last_retrieved == []
    assert real.calls[0][1] == "base"


def test_below_threshold_not_injected():
    store = _FakeStore([_skill("s1"), _skill("s2")], {"s1": 0.2, "s2": 0.5})
    agent = SkillAugmentedProposalAgent(_FakeRealAgent(), store, {"s1", "s2"})

    agent.go("H", "base", {})

    assert agent.last_retrieved == [("s2", 0.5)]


def test_template_truncated_to_800():
    real = _FakeRealAgent()
    agent = SkillAugmentedProposalAgent(real, _FakeStore([_skill("s1", "x" * 2000)]), {"s1"})

    agent.go("H", "base", {})

    sent = real.calls[0][1]
    assert "x" * 800 in sent
    assert "x" * 801 not in sent


def test_injection_log_rounds():
    store = _FakeStore([_skill("s1")], {"s1": 0.7})
    agent = SkillAugmentedProposalAgent(_FakeRealAgent(), store, {"s1"})

    agent.go("H", "base", {})
    agent.go("H", "base", {})

    assert agent.injection_log == [
        {"round": 1, "skill_id": "s1", "similarity": 0.7},
        {"round": 2, "skill_id": "s1", "similarity": 0.7},
    ]
