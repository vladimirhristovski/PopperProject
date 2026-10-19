import hashlib
from dataclasses import dataclass


@dataclass
class Skill:
    id: str
    applicability: str
    design_template: str
    preconditions: str
    domain: str
    source_hypothesis: str
    source_run_id: str | None = None
    parent_skill_id: str | None = None
    approved: bool | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    times_used: int = 0
    times_passed: int = 0
    created_at: str = ""

    @property
    def content_hash(self):
        text = self.applicability + "\n" + self.design_template + "\n" + self.preconditions
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class DecisionExample:
    skill: Skill
    verdict: str
    reason: str
    decider: str
