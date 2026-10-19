from popper_upgrade.skills.schema import Skill


def test_skill_construction_and_equality():
    kwargs = {
        "id": "s1",
        "applicability": "gender bias in occupation associations",
        "design_template": "compare pronoun-occupation co-occurrence rates",
        "preconditions": "social science",
        "domain": "social science",
        "source_hypothesis": "LLMs associate female pronouns with lower-authority occupations",
        "times_used": 0,
        "times_passed": 0,
        "created_at": "2026-09-18T00:00:00+00:00",
    }

    a = Skill(**kwargs)
    b = Skill(**kwargs)

    assert a == b
    assert a.applicability == "gender bias in occupation associations"


def test_content_hash_ignores_metadata():
    base = {
        "applicability": "a",
        "design_template": "d",
        "preconditions": "p",
        "domain": "x",
        "source_hypothesis": "h",
    }

    one = Skill(id="s1", times_used=5, **base)
    two = Skill(id="s2", approved=False, domain="y", **{k: v for k, v in base.items() if k != "domain"})

    assert one.content_hash == two.content_hash


def test_content_hash_changes_with_content():
    one = Skill(id="s1", applicability="a", design_template="d", preconditions="p", domain="", source_hypothesis="")
    two = Skill(id="s1", applicability="a", design_template="d2", preconditions="p", domain="", source_hypothesis="")

    assert one.content_hash != two.content_hash
