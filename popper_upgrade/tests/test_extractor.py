import popper_upgrade.skills.extractor as extractor_module
from popper_upgrade.skills.extractor import ExtractedSkill, ExtractedSkills, SkillExtractor
from popper_upgrade.skills.schema import DecisionExample, Skill
from popper_upgrade.tests.fakes import patch_chain


def _extracted(**overrides):
    fields = {"applicability": "a", "design_template": "t", "preconditions": "p", "parent_skill_id": None}
    fields.update(overrides)
    return ExtractedSkill(**fields)


def test_no_designs_no_llm_call(monkeypatch):
    chain = patch_chain(monkeypatch, extractor_module, ExtractedSkills(skills=[]))

    assert SkillExtractor("claude-x").extract([], "H", "social science", [], None) == []
    assert chain.calls == []


def test_extract_fills_provenance(monkeypatch):
    patch_chain(monkeypatch, extractor_module, ExtractedSkills(skills=[_extracted()]))

    [skill] = SkillExtractor("claude-x").extract(["design"], "H", "social science", [], "run-1")

    assert skill.id
    assert (skill.approved, skill.source_hypothesis, skill.source_run_id) == (None, "H", "run-1")
    assert skill.domain == "social science"


def test_rejected_examples_reach_prompt(monkeypatch):
    chain = patch_chain(monkeypatch, extractor_module, ExtractedSkills(skills=[_extracted(parent_skill_id="old")]))
    rejected = Skill(
        id="old",
        applicability="compare means",
        design_template="t-test on raw counts",
        preconditions="",
        domain="d",
        source_hypothesis="h",
    )

    [skill] = SkillExtractor("claude-x").extract(
        ["design"], "H", "d", [DecisionExample(rejected, "reject", "ignores confounding", "curator")], "run-1"
    )

    sent = chain.calls[0]["input"]
    assert "id: old" in sent
    assert "ignores confounding" in sent
    assert "t-test on raw counts" in sent
    assert skill.parent_skill_id == "old"


def test_caps_at_three(monkeypatch):
    patch_chain(monkeypatch, extractor_module, ExtractedSkills(skills=[_extracted() for _ in range(5)]))

    assert len(SkillExtractor("claude-x").extract(["design"], "H", "d", [], None)) == 3


def test_designs_and_hypothesis_are_the_only_run_data_sent(monkeypatch):
    chain = patch_chain(monkeypatch, extractor_module, ExtractedSkills(skills=[]))

    SkillExtractor("claude-x").extract(["design one", "design two"], "H", "d", [], None)

    sent = chain.calls[0]["input"]
    assert "design one" in sent
    assert "design two" in sent
    assert "Hypothesis: H" in sent
