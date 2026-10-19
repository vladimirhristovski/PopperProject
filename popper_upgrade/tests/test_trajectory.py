from popper_upgrade.trajectory import Trajectory


class _FakeAgent:
    def __init__(self):
        self.log = {
            "designer": ["proposed test A"],
            "executor": ["ran test A"],
            "relevance_checker": [],
            "summarizer": ["summary"],
            "sequential_testing": ["List of p-values: [0.01] \nSequential test result: sufficient evidence - PASS"],
        }
        self.tracked_tests = ["test A"]
        self.tracked_stat = [0.01]
        self.res = True
        self.res_stat = 12.5


class _FakePopper:
    def __init__(self):
        self.agent = _FakeAgent()


def test_from_popper_reads_agent_state_directly():
    popper_instance = _FakePopper()
    validate_result = {
        "parsed_result": {"conclusion": True, "reasoning": "because"},
        "last_message": "final summary text",
    }

    trajectory = Trajectory.from_popper(popper_instance, validate_result)

    assert trajectory.log == popper_instance.agent.log
    assert trajectory.tracked_tests == ["test A"]
    assert trajectory.tracked_stat == [0.01]
    assert trajectory.res is True
    assert trajectory.res_stat == 12.5
    assert trajectory.parsed_result == {"conclusion": True, "reasoning": "because"}
    assert trajectory.last_message == "final summary text"


def test_from_popper_does_not_recompute_the_decision():
    popper_instance = _FakePopper()
    popper_instance.agent.res = False
    popper_instance.agent.res_stat = 0.3
    validate_result = {"parsed_result": {}, "last_message": ""}

    trajectory = Trajectory.from_popper(popper_instance, validate_result)

    assert trajectory.res is False
    assert trajectory.res_stat == 0.3


def test_trajectory_is_detached_from_agent():
    popper_instance = _FakePopper()
    traj = Trajectory.from_popper(popper_instance, {"parsed_result": {}, "last_message": ""})

    popper_instance.agent.tracked_stat.append(0.5)
    popper_instance.agent.tracked_tests.append("test B")
    popper_instance.agent.log["designer"].append("x")

    assert traj.tracked_stat == [0.01]
    assert traj.tracked_tests == ["test A"]
    assert traj.log["designer"] == ["proposed test A"]


def test_trajectory_dict_round_trip():
    traj = Trajectory.from_popper(
        _FakePopper(), {"parsed_result": {"conclusion": True}, "last_message": "done"}
    )

    restored = Trajectory.from_dict(traj.to_dict())

    assert restored == traj
    assert restored.log is not traj.log
