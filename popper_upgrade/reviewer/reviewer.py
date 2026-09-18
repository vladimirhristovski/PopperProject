from langchain_core.prompts import ChatPromptTemplate
from popper.utils import get_llm
from pydantic import BaseModel, Field

from popper_upgrade.llm import ensure_local_llm_api_key

REVIEWER_SYSTEM_PROMPT = """You are a rigorous scientific reviewer evaluating a proposed falsification test's design, before it is executed. You never see any test result or p-value, and your verdict must never be based on one.

Judge the proposal strictly on these four criteria:
1. Relevance: if the main hypothesis were false, would this sub-test also be false? Is it a genuine implication of the main hypothesis?
2. Statistical validity: is this an answerable, well-specified test given the described data?
3. Redundancy: does it duplicate a test already listed as run or attempted?
4. Confounding risk: could the proposed test's outcome be explained by something other than the main hypothesis?

Approve only if all four criteria are satisfied."""


class ReviewVerdict(BaseModel):
    approved: bool = Field(description="whether the proposed test design is approved")
    reasoning: str = Field(description="reasoning behind the verdict, referencing the four criteria")


class Reviewer:
    def __init__(self, llm, port=None, api_key="EMPTY"):
        ensure_local_llm_api_key(port, api_key)
        model = get_llm(llm, port=port, api_key=api_key)
        prompt = ChatPromptTemplate.from_messages([("system", REVIEWER_SYSTEM_PROMPT), ("human", "{input}")])
        self._chain = prompt | model.with_structured_output(ReviewVerdict)

    def review(self, main_hypothesis, proposal, test_results=None):
        test_results_text = test_results if test_results else "None yet."
        input_text = (
            f"Main hypothesis: {main_hypothesis}\n\n"
            f"Proposed falsification test:\n{proposal}\n\n"
            f"Tests already run or attempted:\n{test_results_text}"
        )
        result = self._chain.invoke({"input": input_text})
        return result.approved, result.reasoning
