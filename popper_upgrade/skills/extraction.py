import uuid
from datetime import datetime, timezone

from popper_upgrade.skills.schema import Skill


def extract_skill(trajectory, main_hypothesis, domain=""):
    if not trajectory.tracked_tests:
        return None
    return Skill(
        id=str(uuid.uuid4()),
        applicability=main_hypothesis,
        design_template=trajectory.tracked_tests[-1],
        preconditions=domain,
        domain=domain,
        source_hypothesis=main_hypothesis,
        times_used=1,
        times_passed=1 if trajectory.res else 0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
