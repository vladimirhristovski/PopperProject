from popper_upgrade.skills.extraction import extract_skill
from popper_upgrade.trajectory import Trajectory


def _trajectory(tracked_tests, res):
    return Trajectory(
        log={"designer": [], "executor": [], "relevance_checker": [], "summarizer": [], "sequential_testing": []},
        tracked_tests=tracked_tests,
        tracked_stat=[0.01] * len(tracked_tests),
        res=res,
        res_stat=12.5,
        parsed_result={},
        last_message="",
    )


def test_extracts_skill_from_last_tracked_test():
    trajectory = _trajectory(["test A", "test B"], res=True)

    skill = extract_skill(trajectory, "some hypothesis", domain="social science")

    assert skill is not None
    assert skill.applicability == "some hypothesis"
    assert skill.design_template == "test B"
    assert skill.domain == "social science"
    assert skill.times_used == 1
    assert skill.times_passed == 1


def test_records_failure_when_trajectory_did_not_pass():
    trajectory = _trajectory(["test A"], res=False)

    skill = extract_skill(trajectory, "some hypothesis")

    assert skill.times_passed == 0


def test_returns_none_when_no_tests_were_run():
    trajectory = _trajectory([], res=False)

    assert extract_skill(trajectory, "some hypothesis") is None
