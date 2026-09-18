from dataclasses import dataclass


@dataclass
class Skill:
    id: str
    applicability: str
    design_template: str
    preconditions: str
    domain: str
    source_hypothesis: str
    times_used: int
    times_passed: int
    created_at: str
