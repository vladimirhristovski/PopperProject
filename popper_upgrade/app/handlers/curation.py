import threading
import uuid

from popper_upgrade.models import check_environment, serving_for
from popper_upgrade.skills.context import DatasetSchema, RunContext
from popper_upgrade.skills.curation import curate_before_run

DOMAIN = "social science"

_lock = threading.Lock()
_pending = {}


def _dataset_schema(row):
    schema = row["schema"]
    return DatasetSchema(row["name"], list(schema["columns"].items()), schema["n_rows"])


def preview_curation(ctx, user, model_name, hypotheses, dataset_names):
    hypotheses = [h.strip() for h in hypotheses if h.strip()]
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    with ctx.conn() as conn:
        registry = ctx.registry(conn)
        model = registry.model(model_name)
        check_environment(model, "laptop")
        datasets = [_dataset_schema(registry.dataset(name)) for name in dataset_names]
        source = ctx.skills(conn)
        with serving_for(model):
            curator = ctx.curator_factory(model_name)
            result = curate_before_run(source, curator, RunContext(hypotheses, DOMAIN, datasets))
        applications = {skill.id: skill.applicability for skill in source.list_skills(True)}
    decisions = [
        {
            "skill_id": decision.skill_id,
            "applicability": applications.get(decision.skill_id, ""),
            "approved": decision.approved,
            "reason": decision.reason,
        }
        for decision in result.decisions
    ]
    token = str(uuid.uuid4())
    with _lock:
        _pending[token] = {"decisions": decisions, "overrides": {}}
    return {"token": token, "decisions": decisions}


def apply_overrides(token, overrides):
    with _lock:
        if token not in _pending:
            raise KeyError(token)
        known = {decision["skill_id"] for decision in _pending[token]["decisions"]}
        _pending[token]["overrides"] = {
            skill_id: bool(approve) for skill_id, approve in overrides.items() if skill_id in known
        }


def overrides_for(token):
    with _lock:
        if token not in _pending:
            raise KeyError(token)
        return dict(_pending[token]["overrides"])


def picker_choices(ctx):
    with ctx.conn() as conn:
        registry = ctx.registry(conn)
        models = [model["name"] for model in registry.models("laptop")]
        sets = [row["name"] for row in conn.execute("select name from registry.hypothesis_sets order by name")]
        datasets = [row["name"] for row in conn.execute("select name from registry.datasets order by name")]
    return {"models": models, "hypothesis_sets": sets, "datasets": datasets}


def set_hypotheses(ctx, name):
    with ctx.conn() as conn:
        return [row["text"] for row in ctx.registry(conn).hypothesis_set(name)]
