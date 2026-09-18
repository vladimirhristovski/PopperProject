class ReviewGate:
    def __init__(self, real_proposal_agent, reviewer, max_attempts=3):
        self._real = real_proposal_agent
        self._reviewer = reviewer
        self._max_attempts = max_attempts

    def go(self, main_hypothesis, test_results=None, log=None):
        proposal = None
        for _ in range(self._max_attempts):
            proposal = self._real.go(main_hypothesis, test_results, log)
            approved, reasoning = self._reviewer.review(main_hypothesis, proposal, test_results)
            if log is not None:
                log.setdefault("reviewer", []).append(
                    "Reviewer verdict: {verdict}\nReasoning: {reasoning}\nProposal:\n{proposal}".format(
                        verdict="approved" if approved else "rejected",
                        reasoning=reasoning,
                        proposal=proposal,
                    )
                )
            if approved:
                return proposal
            self._real.add_to_failed_tests(proposal)
        return proposal

    def add_to_existing_tests(self, test):
        self._real.add_to_existing_tests(test)

    def add_to_failed_tests(self, test):
        self._real.add_to_failed_tests(test)
