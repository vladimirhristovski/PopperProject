from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from popper.utils import get_llm
from pydantic import BaseModel, Field

from popper_upgrade.llm import ensure_local_llm_api_key

CRITERIA = ("data_compatible", "domain_fit", "statistically_valid", "non_redundant", "no_leakage")
LEAKAGE_REASON = "leakage: skill derived from a hypothesis in this run"

CURATOR_SYSTEM_PROMPT = """You are a curator of a library of reusable falsification-test strategies (skills). \
You decide whether a skill is fit to be shown to a test-design agent. You never see any p-value, e-value or \
verdict of any run, and your decision must never be based on one.

Judge the skill on these five criteria:
1. data_compatible: the data the skill requires (column types, group structure, sample size) exists in the \
described datasets. When no datasets are described, judge whether the stated preconditions are concrete and \
plausible for tabular data.
2. domain_fit: the skill fits the scientific domain and the kind of hypothesis at hand.
3. statistically_valid: the test the skill describes is well specified, answerable, and does not invite obvious \
confounding or invalid inference.
4. non_redundant: the skill is not a near duplicate of another skill in the library.
5. no_leakage: the skill text is a general strategy and does not embed the hypothesis it came from, named datasets, \
specific columns or specific numbers, and is not derived from a hypothesis being evaluated in this run.

Earlier decisions on similar skills are given as precedents, with their reasons. Stay consistent with them unless \
this skill clearly differs.

Approve only if all five criteria hold. Give a short reason that names the failed criteria."""


class CuratorVerdict(BaseModel):
    approved: bool = Field(description="whether the skill is approved")
    reason: str = Field(description="short reason naming any failed criteria")
    data_compatible: bool = Field(description="required data exists in the described datasets")
    domain_fit: bool = Field(description="skill fits the domain and hypothesis")
    statistically_valid: bool = Field(description="test is well specified and statistically sound")
    non_redundant: bool = Field(description="skill is not a near duplicate of another library skill")
    no_leakage: bool = Field(description="skill is a general strategy and not derived from the evaluated hypotheses")


@dataclass
class CuratorDecision:
    skill_id: str
    approved: bool
    reason: str
    criteria: dict[str, bool]


def normalize(text):
    return " ".join(text.lower().split())


def _format_skill(skill):
    return (
        f"id: {skill.id}\napplicability: {skill.applicability}\ndesign_template: {skill.design_template}\n"
        f"preconditions: {skill.preconditions}\ndomain: {skill.domain}\nsource_hypothesis: {skill.source_hypothesis}"
    )


def _format_precedents(precedents):
    if not precedents:
        return "None."
    return "\n\n".join(
        f"[{example.verdict} by {example.decider}] applicability: {example.skill.applicability}\n"
        f"design_template: {example.skill.design_template}\nreason: {example.reason}"
        for example in precedents
    )


def _format_datasets(datasets):
    if not datasets:
        return "None described."
    return "\n".join(
        f"- {d.name} ({d.n_rows} rows): " + ", ".join(f"{column} [{dtype}]" for column, dtype in d.columns)
        for d in datasets
    )


def _format_library(others):
    if not others:
        return "None."
    return "\n".join(f"- {skill.applicability}" for skill in others)


class SkillCurator:
    def __init__(self, llm, port=None, api_key="EMPTY"):
        ensure_local_llm_api_key(port, api_key)
        model = get_llm(llm, port=port, api_key=api_key)
        self.model_name = llm
        self.prompt_text = CURATOR_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages([("system", self.prompt_text), ("human", "{input}")])
        self._chain = prompt | model.with_structured_output(CuratorVerdict)

    def _decide(self, skill, input_text):
        verdict = self._chain.invoke({"input": input_text})
        criteria = {name: bool(getattr(verdict, name)) for name in CRITERIA}
        return CuratorDecision(skill.id, bool(verdict.approved) and all(criteria.values()), verdict.reason, criteria)

    def review_candidate(self, skill, approved_library, precedents):
        others = [other for other in approved_library if other.id != skill.id]
        input_text = (
            f"Candidate skill to admit to the library:\n{_format_skill(skill)}\n\n"
            f"Approved skills already in the library:\n{_format_library(others)}\n\n"
            f"Precedents:\n{_format_precedents(precedents)}"
        )
        return self._decide(skill, input_text)

    def curate_for_run(self, skills, context, precedents):
        hypotheses = {normalize(h) for h in context.hypotheses}
        decisions = []
        for skill in skills:
            if normalize(skill.source_hypothesis) in hypotheses:
                decisions.append(CuratorDecision(skill.id, False, LEAKAGE_REASON, dict.fromkeys(CRITERIA, False)))
                continue
            others = [other for other in skills if other.id != skill.id]
            input_text = (
                "Hypotheses of the upcoming run:\n" + "\n".join(f"- {h}" for h in context.hypotheses) + "\n\n"
                f"Domain: {context.domain}\n\n"
                f"Registered datasets:\n{_format_datasets(context.datasets)}\n\n"
                f"Skill under review:\n{_format_skill(skill)}\n\n"
                f"Other skills in the library:\n{_format_library(others)}\n\n"
                f"Precedents:\n{_format_precedents(precedents.get(skill.id, []))}"
            )
            try:
                decisions.append(self._decide(skill, input_text))
            except Exception as e:
                decisions.append(
                    CuratorDecision(skill.id, False, f"curator error: {e}", dict.fromkeys(CRITERIA, False))
                )
        return decisions
