from popper_upgrade.reviewer.gate import ReviewGate
from popper_upgrade.skills.proposal_agent import SkillAugmentedProposalAgent


def test_full_graph_runs_with_both_hooks(real_popper_runner):
    runner = real_popper_runner(skills_enabled=True, reviewer_enabled=True)

    traj = runner.validate("H")

    assert traj.res is True
    assert len(traj.tracked_stat) >= 1
    gate = runner.agent.test_proposal_agent
    assert isinstance(gate, ReviewGate)
    assert isinstance(gate._real, SkillAugmentedProposalAgent)
    assert gate._real.last_retrieved_ids == ["s1"]
    assert len(gate.reviews) == 1
    assert gate.overrides == 0
    assert gate.existing_tests == traj.tracked_tests


def test_second_hypothesis_starts_from_clean_state(real_popper_runner):
    runner = real_popper_runner(skills_enabled=True, reviewer_enabled=True)

    first = runner.validate("H1")
    second = runner.validate("H2")

    assert first.tracked_stat == [0.001]
    assert second.tracked_stat == [0.001]
    assert len(runner.agent.test_proposal_agent.reviews) == 1


def test_flags_off_keeps_original_proposal_agent(real_popper_runner):
    runner = real_popper_runner(skills_enabled=False, reviewer_enabled=False)

    runner.prepare()

    assert type(runner.agent.test_proposal_agent).__name__ == "falsification_test_proposal_agent"
