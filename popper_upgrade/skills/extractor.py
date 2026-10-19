import uuid

from langchain_core.prompts import ChatPromptTemplate
from popper.utils import get_llm
from pydantic import BaseModel, Field

from popper_upgrade.llm import ensure_local_llm_api_key
from popper_upgrade.skills.schema import Skill

MAX_SKILLS = 3

EXTRACTOR_SYSTEM_PROMPT = """You distil reusable falsification-test strategies from the test designs that were \
proposed while validating a scientific hypothesis. You only see the designs, never any result.

For each strategy worth keeping, write:
- applicability: when this strategy applies, in general terms
- design_template: the reusable test design, as a short procedure
- preconditions: the data properties the strategy requires, stated explicitly (variable types, group structure, \
sample size, independence assumptions)

Rules:
1. Generalize to a reusable strategy. Strip dataset names, table names, column names and numbers.
2. State the data preconditions explicitly; a strategy without them is not usable.
3. Return at most three strategies and none if the designs hold nothing reusable.
4. You may be shown strategies that were rejected earlier, with the reasons. Avoid those flaws. When you improve \
one of them, set parent_skill_id to that strategy's id; otherwise leave it empty."""


class ExtractedSkill(BaseModel):
    applicability: str = Field(description="general conditions under which the strategy applies")
    design_template: str = Field(description="the reusable falsification test design")
    preconditions: str = Field(description="data properties the strategy requires")
    parent_skill_id: str | None = Field(default=None, description="id of the rejected strategy this one improves")


class ExtractedSkills(BaseModel):
    skills: list[ExtractedSkill] = Field(description="at most three reusable strategies")


class SkillExtractor:
    def __init__(self, llm, port=None, api_key="EMPTY"):
        ensure_local_llm_api_key(port, api_key)
        model = get_llm(llm, port=port, api_key=api_key)
        self.prompt_text = EXTRACTOR_SYSTEM_PROMPT
        prompt = ChatPromptTemplate.from_messages([("system", self.prompt_text), ("human", "{input}")])
        self._chain = prompt | model.with_structured_output(ExtractedSkills)

    def extract(self, designs, hypothesis, domain, rejected_examples, source_run_id):
        if not designs:
            return []
        result = self._chain.invoke({"input": self._input(designs, hypothesis, domain, rejected_examples)})
        return [
            Skill(
                id=str(uuid.uuid4()),
                applicability=item.applicability,
                design_template=item.design_template,
                preconditions=item.preconditions,
                domain=domain,
                source_hypothesis=hypothesis,
                source_run_id=source_run_id,
                parent_skill_id=item.parent_skill_id,
                approved=None,
            )
            for item in result.skills[:MAX_SKILLS]
        ]

    @staticmethod
    def _input(designs, hypothesis, domain, rejected_examples):
        designs_text = "\n\n".join(f"Design {i}:\n{d}" for i, d in enumerate(designs, 1))
        text = f"Hypothesis: {hypothesis}\nDomain: {domain}\n\nImplemented test designs:\n{designs_text}"
        if rejected_examples:
            text += "\n\nStrategies rejected earlier:\n" + "\n\n".join(
                _format_rejected(example) for example in rejected_examples
            )
        return text


def _format_rejected(example):
    skill = example.skill
    return (
        f"id: {skill.id}\napplicability: {skill.applicability}\ndesign_template: {skill.design_template}\n"
        f"rejection reason: {example.reason}"
    )
