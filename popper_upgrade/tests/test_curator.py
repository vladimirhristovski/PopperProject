import pandas as pd

import popper_upgrade.skills.curator as curator_module
from popper_upgrade.skills.context import RunContext, describe_datasets
from popper_upgrade.skills.curator import CRITERIA, LEAKAGE_REASON, CuratorVerdict, SkillCurator
from popper_upgrade.skills.schema import DecisionExample, Skill
from popper_upgrade.tests.fakes import patch_chain
from popper_upgrade.trajectory import Trajectory


def _skill(skill_id="s1", source_hypothesis="other hypothesis"):
    return Skill(
        id=skill_id,
        applicability=f"applicability {skill_id}",
        design_template="compare group means",
        preconditions="two groups",
        domain="social science",
        source_hypothesis=source_hypothesis,
    )


def _verdict(approved=True, **criteria):
    fields = dict.fromkeys(CRITERIA, True)
    fields.update(criteria)
    return CuratorVerdict(approved=approved, reason="because", **fields)


def _context(hypotheses=("H",), datasets=()):
    return RunContext(list(hypotheses), "social science", list(datasets))


def test_leakage_rejected_without_llm(monkeypatch):
    chain = patch_chain(monkeypatch, curator_module, _verdict())
    curator = SkillCurator("claude-x")

    leaking = _skill(source_hypothesis="  The  Hypothesis ")

    [decision] = curator.curate_for_run([leaking], _context(["the hypothesis"]), {})

    assert decision.approved is False
    assert decision.reason == LEAKAGE_REASON
    assert chain.calls == []


def test_all_criteria_required_for_approval(monkeypatch):
    patch_chain(monkeypatch, curator_module, _verdict(approved=True, non_redundant=False))

    [decision] = SkillCurator("claude-x").curate_for_run([_skill()], _context(), {})

    assert decision.approved is False
    assert decision.criteria["non_redundant"] is False
    assert decision.criteria["domain_fit"] is True


def test_approves_when_all_criteria_hold(monkeypatch):
    patch_chain(monkeypatch, curator_module, _verdict())

    [decision] = SkillCurator("claude-x").curate_for_run([_skill()], _context(), {})

    assert decision.approved is True
    assert decision.reason == "because"


def test_llm_error_becomes_rejection(monkeypatch):
    def outputs(inputs):
        if "applicability s1" in inputs["input"].split("Skill under review:")[1].split("Other skills")[0]:
            raise RuntimeError("boom")
        return _verdict()

    patch_chain(monkeypatch, curator_module, outputs)

    first, second = SkillCurator("claude-x").curate_for_run([_skill("s1"), _skill("s2")], _context(), {})

    assert first.approved is False
    assert first.reason == "curator error: boom"
    assert second.approved is True


def test_malformed_output_becomes_rejection(monkeypatch):
    patch_chain(monkeypatch, curator_module, None)

    [decision] = SkillCurator("claude-x").curate_for_run([_skill()], _context(), {})

    assert decision.approved is False
    assert decision.reason.startswith("curator error: ")


def test_precedents_and_schema_in_prompt(monkeypatch):
    chain = patch_chain(monkeypatch, curator_module, _verdict())
    datasets = describe_datasets({"df_winobias": pd.DataFrame({"gender": ["f"], "occupation": ["nurse"]})})
    precedent = DecisionExample(_skill("old"), "reject", "ignores multiple comparisons", "curator")

    SkillCurator("claude-x").curate_for_run([_skill()], _context(datasets=datasets), {"s1": [precedent]})

    sent = chain.calls[0]["input"]
    assert "gender [object]" in sent
    assert "occupation [object]" in sent
    assert "winobias (1 rows)" in sent
    assert "ignores multiple comparisons" in sent


def test_prompt_never_contains_results(monkeypatch):
    chain = patch_chain(monkeypatch, curator_module, _verdict())
    trajectory = Trajectory(
        log={},
        tracked_tests=["design"],
        tracked_stat=[0.003],
        res=True,
        res_stat=12.5,
        parsed_result={"conclusion": True},
        last_message="p-value 0.003",
    )
    context = _context(["H"], describe_datasets({"df_t": pd.DataFrame({"a": [1]})}))
    skill = _skill()
    skill.times_used, skill.times_passed = 4, 3

    curator = SkillCurator("claude-x")
    curator.curate_for_run([skill], context, {})
    curator.review_candidate(skill, [_skill("s2")], [])

    for call in chain.calls:
        sent = call["input"]
        for forbidden in ("p-value", "e-value", "res_stat", "parsed_result", str(trajectory.res_stat), "0.003"):
            assert forbidden not in sent


def test_review_candidate_lists_library_and_precedents(monkeypatch):
    chain = patch_chain(monkeypatch, curator_module, _verdict(approved=False, no_leakage=False))
    precedent = DecisionExample(_skill("old"), "approve", "sound design", "human")

    decision = SkillCurator("claude-x").review_candidate(_skill("s1"), [_skill("s2")], [precedent])

    sent = chain.calls[0]["input"]
    assert "applicability s2" in sent
    assert "sound design" in sent
    assert decision.approved is False
    assert decision.skill_id == "s1"


def test_prompt_text_and_model_name_exposed(monkeypatch):
    patch_chain(monkeypatch, curator_module, _verdict())

    curator = SkillCurator("claude-x")

    assert curator.model_name == "claude-x"
    assert "five criteria" in curator.prompt_text
