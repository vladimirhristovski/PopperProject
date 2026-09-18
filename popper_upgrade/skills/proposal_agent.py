def _format_skills_block(skills):
    entries = []
    for skill in skills:
        entries.append(
            "Applicability: {applicability}\nSuggested test: {design_template}".format(
                applicability=skill.applicability, design_template=skill.design_template
            )
        )
    return (
        "Suggested strategies from prior experiments (not run in this session — "
        "for inspiration, adapt as needed, avoid pure duplication):\n\n"
        + "\n\n".join(entries)
        + "\n\nFalsification tests already run in this session:\n\n"
    )


class SkillAugmentedProposalAgent:
    def __init__(self, real_proposal_agent, skill_store, k=3):
        self._real = real_proposal_agent
        self._store = skill_store
        self._k = k

    def go(self, main_hypothesis, test_results=None, log=None):
        skills = self._store.retrieve(main_hypothesis, k=self._k) if self._store else []
        if not skills:
            return self._real.go(main_hypothesis, test_results, log)
        body = test_results if test_results else "No Implemented Falsification Test Yet."
        augmented = _format_skills_block(skills) + body
        return self._real.go(main_hypothesis, augmented, log)

    def add_to_existing_tests(self, test):
        self._real.add_to_existing_tests(test)

    def add_to_failed_tests(self, test):
        self._real.add_to_failed_tests(test)
