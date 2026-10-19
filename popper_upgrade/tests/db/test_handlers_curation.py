import uuid

import pytest

from popper_upgrade.app.context import REGISTRY_DIR
from popper_upgrade.app.handlers import curation as handlers
from popper_upgrade.skills.schema import Skill
from popper_upgrade.tests.fakes import FakeCurator

pytestmark = pytest.mark.db


@pytest.fixture
def curation_ctx(make_ctx, db):
    curator = FakeCurator()
    ctx = make_ctx(curator_factory=lambda name: curator)
    ctx.registry(db).seed_from(REGISTRY_DIR)
    return ctx, curator


def _approved_skill(ctx, db, applicability):
    repo = ctx.skills(db)
    skill_id = repo.add_candidate_skill(
        Skill(
            id=str(uuid.uuid4()),
            applicability=applicability,
            design_template="compare group means",
            preconditions="",
            domain="social science",
            source_hypothesis="",
        )
    )
    repo.set_approval(skill_id, True, "curator", "seed", "entry")
    return skill_id


def test_preview_returns_decisions_and_context(curation_ctx, db, plain_user):
    ctx, curator = curation_ctx
    skill_id = _approved_skill(ctx, db, "compare group means")
    result = handlers.preview_curation(ctx, plain_user, "claude-sonnet-4-5", ["H one", "H two"], ["winobias"])
    assert [d["skill_id"] for d in result["decisions"]] == [skill_id]
    assert result["decisions"][0]["approved"] is True
    _, context, _ = curator.run_calls[0]
    assert context.hypotheses == ["H one", "H two"]
    assert [(d.name, d.n_rows) for d in context.datasets] == [("winobias", 1584)]


def test_preview_rejects_cluster_model_and_empty_hypotheses(curation_ctx, db, plain_user):
    ctx, _ = curation_ctx
    with pytest.raises(ValueError):
        handlers.preview_curation(ctx, plain_user, "casperhansen/llama-3.3-70b-instruct-awq", ["H"], [])
    with pytest.raises(ValueError):
        handlers.preview_curation(ctx, plain_user, "claude-sonnet-4-5", ["  "], [])


def test_overrides_stored_per_token(curation_ctx, db, plain_user):
    ctx, _ = curation_ctx
    first = _approved_skill(ctx, db, "first skill")
    second = _approved_skill(ctx, db, "second skill")
    a = handlers.preview_curation(ctx, plain_user, "claude-sonnet-4-5", ["H"], [])["token"]
    b = handlers.preview_curation(ctx, plain_user, "claude-sonnet-4-5", ["H"], [])["token"]
    assert a != b
    handlers.apply_overrides(a, {first: False, "not-a-skill": True})
    handlers.apply_overrides(b, {second: False})
    assert handlers.overrides_for(a) == {first: False}
    assert handlers.overrides_for(b) == {second: False}
    handlers.apply_overrides(a, {})
    assert handlers.overrides_for(a) == {}
    with pytest.raises(KeyError):
        handlers.apply_overrides("missing", {})
