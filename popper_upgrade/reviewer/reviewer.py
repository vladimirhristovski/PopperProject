from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from popper.utils import get_llm

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
        model = get_llm(llm, port=port, api_key=api_key)
        prompt = ChatPromptTemplate.from_messages([("system", REVIEWER_SYSTEM_PROMPT), ("human", "{input}")])
        self._chain = prompt | model.with_structured_output(ReviewVerdict)

    def review(self, main_hypothesis, proposal, test_results=None):
        input_text = "Main hypothesis: {main_hypothesis}\n\nProposed falsification test:\n{proposal}\n\nTests already run or attempted:\n{test_results}".format(
            main_hypothesis=main_hypothesis,
            proposal=proposal,
            test_results=test_results if test_results else "None yet.",
        )
        result = self._chain.invoke({"input": input_text})
        return result.approved, result.reasoning
