from popper import Popper

import popper_upgrade.config as config
from popper_upgrade.reviewer.gate import ReviewGate
from popper_upgrade.skills.proposal_agent import SkillAugmentedProposalAgent
from popper_upgrade.trajectory import Trajectory


class PopperRunner:
    def __init__(
        self,
        skills_enabled=None,
        skill_store=None,
        reviewer_enabled=None,
        reviewer=None,
        max_review_attempts=3,
        **popper_kwargs
    ):
        self._popper = Popper(**popper_kwargs)
        self._skills_enabled = config.SKILLS_ENABLED if skills_enabled is None else skills_enabled
        self._skill_store = skill_store
        self._reviewer_enabled = config.REVIEWER_ENABLED if reviewer_enabled is None else reviewer_enabled
        self._reviewer = reviewer
        self._max_review_attempts = max_review_attempts

    @property
    def agent(self):
        return self._popper.agent

    def register_data(self, loader_type, data_path, **kwargs):
        self._popper.register_data(loader_type, data_path, **kwargs)

    def configure(self, **kwargs):
        self._popper.configure(**kwargs)
        self._apply_hooks()

    def _apply_hooks(self):
        agent = self._popper.agent
        if self._skills_enabled and self._skill_store is not None:
            agent.test_proposal_agent = SkillAugmentedProposalAgent(agent.test_proposal_agent, self._skill_store)
        if self._reviewer_enabled and self._reviewer is not None:
            agent.test_proposal_agent = ReviewGate(agent.test_proposal_agent, self._reviewer, self._max_review_attempts)

    def validate(self, hypothesis) -> Trajectory:
        result = self._popper.validate(hypothesis)
        return Trajectory.from_popper(self._popper, result)
