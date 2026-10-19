class ReviewGate:
    def __init__(self, real_proposal_agent, reviewer, max_attempts=3):
        self._real = real_proposal_agent
        self._reviewer = reviewer
        self._max_attempts = max_attempts
        self._round = 0
        self.rejected = []
        self.reviews = []
        self.overrides = 0

    def __getattr__(self, name):
        if name == "_real":
            raise AttributeError(name)
        return getattr(self._real, name)

    def _prior_designs(self):
        existing = getattr(self._real, "existing_tests", [])
        failed = getattr(self._real, "failed_tests", [])
        return list(existing) + list(failed) + self.rejected

    def _prompt_with_rejections(self, test_results):
        if not self.rejected:
            return test_results
        block = "\n\nDesigns rejected by the reviewer in this session (do not propose these again):\n" + "\n".join(
            f"- {r}" for r in self.rejected
        )
        return (test_results or "") + block

    def go(self, main_hypothesis, test_results=None, log=None):
        self._round += 1
        proposal = None
        last_error = None
        approved = False
        last_review = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                proposal = self._real.go(main_hypothesis, self._prompt_with_rejections(test_results), log)
            except Exception as e:
                last_error = e
                if log is not None:
                    log.setdefault("reviewer", []).append(f"Proposal generation failed: {e}")
                continue
            try:
                approved, reasoning = self._reviewer.review(main_hypothesis, proposal, self._prior_designs())
            except Exception as e:
                approved, reasoning = False, f"Reviewer call failed: {e}"
            last_review = {
                "round": self._round,
                "attempt": attempt,
                "proposal": proposal,
                "approved": approved,
                "reason": reasoning,
                "overridden": False,
            }
            self.reviews.append(last_review)
            if log is not None:
                verdict = "approved" if approved else "rejected"
                log.setdefault("reviewer", []).append(
                    f"Reviewer verdict: {verdict}\nReasoning: {reasoning}\nProposal:\n{proposal}"
                )
            if approved:
                return proposal
            self.rejected.append(proposal)
        if proposal is None and last_error is not None:
            raise last_error
        if last_review is not None and not last_review["approved"]:
            last_review["overridden"] = True
            self.overrides += 1
        return proposal

    def add_to_existing_tests(self, test):
        self._real.add_to_existing_tests(test)

    def add_to_failed_tests(self, test):
        self._real.add_to_failed_tests(test)
