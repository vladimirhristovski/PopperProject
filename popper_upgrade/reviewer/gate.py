class ReviewGate:
    def __init__(self, real_proposal_agent, reviewer, max_attempts=3):
        self._real = real_proposal_agent
        self._reviewer = reviewer
        self._max_attempts = max_attempts

    def go(self, main_hypothesis, test_results=None, log=None):
        proposal = None
        last_error = None
        for _ in range(self._max_attempts):
            try:
                proposal = self._real.go(main_hypothesis, test_results, log)
            except Exception as e:
                last_error = e
                if log is not None:
                    log.setdefault("reviewer", []).append(f"Proposal generation failed: {e}")
                continue
            try:
                approved, reasoning = self._reviewer.review(main_hypothesis, proposal, test_results)
            except Exception as e:
                approved, reasoning = False, f"Reviewer call failed: {e}"
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
        if proposal is None and last_error is not None:
            raise last_error
        return proposal

    def add_to_existing_tests(self, test):
        self._real.add_to_existing_tests(test)

    def add_to_failed_tests(self, test):
        self._real.add_to_failed_tests(test)
