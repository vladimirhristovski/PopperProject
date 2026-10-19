from popper import Popper

from popper_upgrade import config
from popper_upgrade.llm import ensure_local_llm_api_key
from popper_upgrade.reviewer.gate import ReviewGate
from popper_upgrade.skills.proposal_agent import SkillAugmentedProposalAgent
from popper_upgrade.trajectory import Trajectory


class PopperRunner:
    def __init__(
        self,
        skills_enabled=None,
        skill_source=None,
        allowed_skill_ids=None,
        reviewer_enabled=None,
        reviewer=None,
        max_review_attempts=3,
        data_loader_factory=None,
        **popper_kwargs
    ):
        ensure_local_llm_api_key(popper_kwargs.get("server_port"), popper_kwargs.get("api_key", "EMPTY"))
        self._popper_kwargs = popper_kwargs
        self._popper = None
        self._data_call = None
        self._configure_kwargs = None
        self._skills_enabled = config.SKILLS_ENABLED if skills_enabled is None else skills_enabled
        self._skill_source = skill_source
        self.allowed_skill_ids = allowed_skill_ids
        self.skill_agent = None
        self.review_gate = None
        self._reviewer_enabled = config.REVIEWER_ENABLED if reviewer_enabled is None else reviewer_enabled
        self._reviewer = reviewer
        self._max_review_attempts = max_review_attempts
        self._data_loader_factory = data_loader_factory

    @property
    def agent(self):
        return self._popper.agent if self._popper is not None else None

    def register_data(self, data_path, loader_type="bio", **kwargs):
        self._data_call = (data_path, loader_type, kwargs)

    def configure(self, **kwargs):
        if kwargs.get("relevance_checker", True) is not True:
            raise ValueError("relevance_checker must be True")
        self._configure_kwargs = kwargs

    def prepare(self):
        popper = Popper(**self._popper_kwargs)
        if self._data_loader_factory is not None:
            popper.data_loader = self._data_loader_factory()
        elif self._data_call is not None:
            data_path, loader_type, kwargs = self._data_call
            popper.register_data(data_path, loader_type=loader_type, **kwargs)
        if self._configure_kwargs is not None:
            popper.configure(**self._configure_kwargs)
        self._popper = popper
        self._apply_hooks()

    def data_tables(self):
        loader = getattr(self._popper, "data_loader", None)
        return dict(loader.table_dict) if loader is not None else {}

    def _apply_hooks(self):
        self.skill_agent = None
        self.review_gate = None
        agent = self._popper.agent
        if agent is None:
            return
        if self._skills_enabled and self._skill_source is not None:
            self.skill_agent = SkillAugmentedProposalAgent(
                agent.test_proposal_agent, self._skill_source, self.allowed_skill_ids
            )
            agent.test_proposal_agent = self.skill_agent
        if self._reviewer_enabled and self._reviewer is not None:
            self.review_gate = ReviewGate(agent.test_proposal_agent, self._reviewer, self._max_review_attempts)
            agent.test_proposal_agent = self.review_gate

    def validate(self, hypothesis) -> Trajectory:
        self.prepare()
        result = self._popper.validate(hypothesis)
        return Trajectory.from_popper(self._popper, result)
