import tempfile

import pytest

from popper_upgrade.skills.schema import Skill
from popper_upgrade.skills.store import SkillStore


def _skill(id_, applicability, design_template):
    return Skill(
        id=id_,
        applicability=applicability,
        design_template=design_template,
        preconditions="social science",
        domain="social science",
        source_hypothesis=applicability,
        times_used=0,
        times_passed=0,
        created_at="2026-09-18T00:00:00+00:00",
    )


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield SkillStore(persist_directory=tmp)


def test_retrieve_ranks_closest_skill_first(store):
    store.add(_skill("s1", "gender bias in occupation associations for pronouns", "pronoun-occupation co-occurrence test"))
    store.add(_skill("s2", "vLLM server GPU memory allocation tuning", "vram threshold branching test"))

    results = store.retrieve("do LLMs associate women with lower-status jobs", k=1)

    assert len(results) == 1
    assert results[0].id == "s1"


def test_add_skips_near_duplicate(store):
    added_first = store.add(_skill("s1", "gender bias in occupation associations for pronouns", "pronoun-occupation co-occurrence test"))
    added_second = store.add(_skill("s2", "gender bias in occupation associations for pronouns", "pronoun-occupation co-occurrence test"))

    assert added_first is True
    assert added_second is False
    assert len(store.retrieve("gender bias in occupation associations for pronouns", k=5)) == 1


def test_record_outcome_updates_counters(store):
    store.add(_skill("s1", "gender bias in occupation associations", "pronoun-occupation co-occurrence test"))

    store.record_outcome("s1", passed=True)
    store.record_outcome("s1", passed=False)

    result = store.retrieve("gender bias in occupation associations", k=1)[0]
    assert result.times_used == 2
    assert result.times_passed == 1
