from dataclasses import dataclass, field

from popper_upgrade.skills.curator import CuratorDecision

PRECEDENT_COUNT = 5


@dataclass
class CurationResult:
    approved_ids: set[str]
    rejected_ids: set[str]
    decisions: list[CuratorDecision]
    overrides: dict[str, bool] = field(default_factory=dict)


def apply_overrides(result, overrides):
    known = {decision.skill_id for decision in result.decisions}
    applied = {skill_id: approve for skill_id, approve in overrides.items() if skill_id in known}
    approved = (result.approved_ids | {i for i, approve in applied.items() if approve}) - {
        i for i, approve in applied.items() if not approve
    }
    rejected = (result.rejected_ids | {i for i, approve in applied.items() if not approve}) - {
        i for i, approve in applied.items() if approve
    }
    return CurationResult(approved, rejected, result.decisions, applied)


def curate_before_run(source, curator, context, overrides=None):
    skills = source.list_skills(True)
    precedents = {
        skill.id: [
            example
            for example in source.similar_decisions(skill.applicability, PRECEDENT_COUNT)
            if example.skill.id != skill.id
        ]
        for skill in skills
    }
    decisions = curator.curate_for_run(skills, context, precedents)
    result = CurationResult(
        {d.skill_id for d in decisions if d.approved},
        {d.skill_id for d in decisions if not d.approved},
        decisions,
    )
    return apply_overrides(result, overrides) if overrides else result
