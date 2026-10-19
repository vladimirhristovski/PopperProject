def _format_skills_block(skills, max_template_chars):
    entries = []
    for skill in skills:
        template = skill.design_template[:max_template_chars]
        entries.append(f"Applicability: {skill.applicability}\nSuggested test: {template}")
    return (
        "Suggested strategies from prior experiments (not run in this session — "
        "for inspiration, adapt as needed, avoid pure duplication):\n\n"
        + "\n\n".join(entries)
        + "\n\nFalsification tests already run in this session:\n\n"
    )


class SkillAugmentedProposalAgent:
    def __init__(
        self, real_proposal_agent, source, allowed_ids=None, k=2, min_similarity=0.35, max_template_chars=800
    ):
        self._real = real_proposal_agent
        self._source = source
        self._allowed_ids = allowed_ids
        self._k = k
        self._min_similarity = min_similarity
        self._max_template_chars = max_template_chars
        self._round = 0
        self.last_retrieved_ids = []
        self.last_retrieved = []
        self.injection_log = []

    def __getattr__(self, name):
        if name == "_real":
            raise AttributeError(name)
        return getattr(self._real, name)

    def go(self, main_hypothesis, test_results=None, log=None):
        self._round += 1
        scored = self._source.retrieve_scored(
            main_hypothesis, self._k, allowed_ids=self._allowed_ids, min_similarity=self._min_similarity
        )
        skills = [skill for skill, _ in scored]
        self.last_retrieved_ids = [skill.id for skill in skills]
        self.last_retrieved = [(skill.id, similarity) for skill, similarity in scored]
        self.injection_log.extend(
            {"round": self._round, "skill_id": skill.id, "similarity": similarity} for skill, similarity in scored
        )
        if not skills:
            return self._real.go(main_hypothesis, test_results, log)
        body = test_results if test_results else "No Implemented Falsification Test Yet."
        augmented = _format_skills_block(skills, self._max_template_chars) + body
        return self._real.go(main_hypothesis, augmented, log)

    def add_to_existing_tests(self, test):
        self._real.add_to_existing_tests(test)

    def add_to_failed_tests(self, test):
        self._real.add_to_failed_tests(test)
