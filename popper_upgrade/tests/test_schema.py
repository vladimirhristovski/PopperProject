from popper_upgrade.skills.schema import Skill


def test_skill_construction_and_equality():
    kwargs = dict(
        id="s1",
        applicability="gender bias in occupation associations",
        design_template="compare pronoun-occupation co-occurrence rates",
        preconditions="social science",
        domain="social science",
        source_hypothesis="LLMs associate female pronouns with lower-authority occupations",
        times_used=0,
        times_passed=0,
        created_at="2026-09-18T00:00:00+00:00",
    )

    a = Skill(**kwargs)
    b = Skill(**kwargs)

    assert a == b
    assert a.applicability == "gender bias in occupation associations"
