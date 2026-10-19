import json
from dataclasses import asdict, dataclass
from pathlib import Path

from popper_upgrade.eval.controls import flags_for_arm
from popper_upgrade.session import RunSpec

ARMS = ("baseline", "skills", "skills_reviewer", "random_skills", "wrong_domain_skills")
SOURCE_KINDS = ("gender_custom", "discoverybench")
PROMPT_ROLES = ("curator", "reviewer", "extractor")
POPPER_DEFAULTS = {"alpha": 0.1, "max_num_of_tests": 5, "max_retry": 3, "time_limit": 60}


@dataclass
class ExperimentSpec:
    name: str
    models: list
    arms: list
    source: dict
    train_tasks: list
    test_tasks: list
    replicates: int
    permutation_seeds: list
    popper: dict
    prompt_versions: dict

    def to_dict(self):
        return asdict(self)


def _validate(spec):
    unknown = set(spec.arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    if not spec.arms or not spec.models:
        raise ValueError("a spec needs at least one model and one arm")
    if spec.source.get("kind") not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {spec.source.get('kind')}")
    if spec.replicates < 1:
        raise ValueError("replicates must be at least 1")
    if len(spec.permutation_seeds) < spec.replicates:
        raise ValueError("permutation_seeds must cover every replicate")
    if len(set(spec.permutation_seeds)) != len(spec.permutation_seeds):
        raise ValueError("permutation_seeds must be distinct")
    if set(spec.train_tasks) & set(spec.test_tasks):
        raise ValueError("a task cannot be in both train and test")
    if not spec.test_tasks:
        raise ValueError("a spec needs at least one test task")
    if set(spec.prompt_versions) != set(PROMPT_ROLES):
        raise ValueError(f"prompt_versions must name {list(PROMPT_ROLES)}")


def parse_spec(data):
    spec = ExperimentSpec(
        name=data["name"],
        models=list(data["models"]),
        arms=list(data["arms"]),
        source=dict(data["source"]),
        train_tasks=list(data.get("train_tasks", [])),
        test_tasks=list(data["test_tasks"]),
        replicates=int(data["replicates"]),
        permutation_seeds=list(data["permutation_seeds"]),
        popper={**POPPER_DEFAULTS, **data.get("popper", {})},
        prompt_versions=dict(data["prompt_versions"]),
    )
    _validate(spec)
    return spec


def load_spec(path):
    return parse_spec(json.loads(Path(path).read_text(encoding="utf-8")))


def configure_kwargs(spec):
    return {**spec.popper, "aggregate_test": "E-value", "relevance_checker": True, "use_react_agent": True}


def run_identity(run):
    return (run.model["name"], run.arm, run.task, run.hypothesis, run.replicate, run.permuted)


def expand(spec, model, examples, popper_kwargs, environment, experiment_id=None, snapshot_id=None):
    configure = configure_kwargs(spec)
    runs = []
    for example in examples:
        for replicate in range(example.get("replicates", spec.replicates)):
            seed = spec.permutation_seeds[replicate]
            permuted = example["permuted"]
            for arm in spec.arms:
                skills_enabled, reviewer_enabled = flags_for_arm(arm)
                runs.append(
                    RunSpec(
                        hypothesis=example["hypothesis"],
                        hypothesis_id=example.get("hypothesis_id"),
                        split="test",
                        arm=arm,
                        model=model,
                        popper_kwargs={**popper_kwargs, "domain": example["domain"]},
                        configure_kwargs=dict(configure),
                        skills_enabled=skills_enabled,
                        reviewer_enabled=reviewer_enabled,
                        learn=False,
                        environment=environment,
                        experiment_id=experiment_id,
                        permuted=permuted,
                        permute_seed=seed if permuted else None,
                        library_snapshot_id=snapshot_id,
                        data_loader_factory=example["factory"](seed if permuted else None),
                        answer=example["answer"],
                        task=example["task"],
                        replicate=replicate,
                        datasets=list(example.get("datasets", [])),
                    )
                )
    return runs


def _normalise(text):
    return " ".join(text.split()).lower()


def check_snapshot_isolation(snapshot, test_hypotheses):
    forbidden = {_normalise(text) for text in test_hypotheses}
    for skill in [*snapshot.list_skills(True), *snapshot.list_skills(False)]:
        if _normalise(skill.source_hypothesis) in forbidden:
            raise ValueError(f"skill {skill.id} derived from test hypothesis")
