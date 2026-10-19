import random

ARM_FLAGS = {
    "baseline": (False, False),
    "skills": (True, False),
    "skills_reviewer": (True, True),
    "random_skills": (True, False),
    "wrong_domain_skills": (True, False),
}


def flags_for_arm(arm):
    return ARM_FLAGS[arm]


def _domain_key(domain):
    return (domain or "").strip().lower()


class _Wrapper:
    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        if name == "_base":
            raise AttributeError(name)
        return getattr(self._base, name)

    def list_skills(self, approved=True):
        return self._base.list_skills(approved)

    def similar_decisions(self, text, k=5):
        return self._base.similar_decisions(text, k)


class RandomSkillSource(_Wrapper):
    def __init__(self, base, seed):
        super().__init__(base)
        self._seed = seed

    def retrieve_scored(self, query, k, allowed_ids=None, min_similarity=0.0):
        if allowed_ids is not None and not allowed_ids:
            return []
        pool = sorted(
            (skill for skill in self._base.list_skills(True) if allowed_ids is None or skill.id in allowed_ids),
            key=lambda skill: skill.id,
        )
        rng = random.Random(f"{self._seed}:{query}")
        return [(skill, None) for skill in rng.sample(pool, min(k, len(pool)))]


class WrongDomainSkillSource(_Wrapper):
    def __init__(self, base, run_domain):
        super().__init__(base)
        self._run_domain = _domain_key(run_domain)

    def _other_domain(self, approved=True):
        return [skill for skill in self._base.list_skills(approved) if _domain_key(skill.domain) != self._run_domain]

    @property
    def degenerate(self):
        return not self._other_domain(True)

    def list_skills(self, approved=True):
        return self._other_domain(approved)

    def retrieve_scored(self, query, k, allowed_ids=None, min_similarity=0.0):
        ids = {skill.id for skill in self._other_domain(True)}
        if allowed_ids is not None:
            ids &= set(allowed_ids)
        return self._base.retrieve_scored(query, k, allowed_ids=ids, min_similarity=min_similarity)


def source_for_arm(arm, base, seed, domain):
    if arm == "random_skills":
        return RandomSkillSource(base, seed)
    if arm == "wrong_domain_skills":
        return WrongDomainSkillSource(base, domain)
    return base
