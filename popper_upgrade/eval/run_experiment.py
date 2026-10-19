import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path

from popper_upgrade.db.registry_repo import prompt_hash
from popper_upgrade.eval.benchmark import discoverybench_examples, gender_examples
from popper_upgrade.eval.controls import flags_for_arm, source_for_arm
from popper_upgrade.eval.spec import check_snapshot_isolation, expand, load_spec, run_identity
from popper_upgrade.models import ServingError, check_environment, popper_kwargs_for, serving_for
from popper_upgrade.reviewer.reviewer import REVIEWER_SYSTEM_PROMPT
from popper_upgrade.run import build_agents, default_data_dir, default_trajectory_dir, session_wiring
from popper_upgrade.session import RunSession
from popper_upgrade.skills.curator import CURATOR_SYSTEM_PROMPT
from popper_upgrade.skills.extractor import EXTRACTOR_SYSTEM_PROMPT

EXPERIMENT_NAMESPACE = uuid.UUID("5d3c1c53-7a55-4c4e-9a64-3c2f0b4f6b0a")
PROMPTS = {
    "curator": CURATOR_SYSTEM_PROMPT,
    "reviewer": REVIEWER_SYSTEM_PROMPT,
    "extractor": EXTRACTOR_SYSTEM_PROMPT,
}


def current_prompt_hashes():
    return {role: prompt_hash(role, text) for role, text in PROMPTS.items()}


def check_prompt_versions(spec):
    current = current_prompt_hashes()
    stale = sorted(role for role, digest in spec.prompt_versions.items() if current[role] != digest)
    if stale:
        raise ValueError(f"prompt versions differ from the spec for: {', '.join(stale)}")


def apply_overrides(spec, tasks=None, replicates=None):
    changes = {}
    if tasks:
        unknown = set(tasks) - set(spec.test_tasks)
        if unknown:
            raise ValueError(f"not test tasks of {spec.name}: {sorted(unknown)}")
        changes["test_tasks"] = list(tasks)
    if replicates is not None:
        changes["replicates"] = replicates
    return replace(spec, **changes) if changes else spec


def build_examples(spec, source, data_dir, per_task=None):
    kind = spec.source["kind"]
    if kind == "discoverybench":
        path = Path(data_dir) / spec.source["path"]
        return discoverybench_examples(str(path), set(spec.test_tasks), per_task)
    sets = source.registry["hypothesis_sets"]
    name = spec.source["hypothesis_set"]
    if name not in sets:
        raise ValueError(f"hypothesis set {name} is not in the snapshot registry")
    permutable = {row["name"]: row["permutable_columns"] for row in source.registry["datasets"]}
    if not permutable:
        raise ValueError("the snapshot registry has no datasets")
    task = spec.source.get("task", "gender_bias")
    if task not in spec.test_tasks:
        return []
    return gender_examples(sets[name], data_dir, permutable, task, spec.source.get("domain", "social science"))


def experiment_id_for(spec, snapshot_id, per_task=None):
    body = {key: value for key, value in spec.to_dict().items() if key != "models"}
    body["per_task"] = per_task
    canonical = json.dumps(body, sort_keys=True)
    return str(uuid.uuid5(EXPERIMENT_NAMESPACE, f"{snapshot_id}:{canonical}"))


def _model_entry(source, name):
    model = next((m for m in source.registry["models"] if m["name"] == name), None)
    if model is None:
        raise ValueError(f"model {name} is not in the snapshot registry")
    return model


def run_experiment(
    spec_path,
    model,
    source,
    sink,
    environment,
    tasks=None,
    replicates=None,
    per_task=None,
    data_dir=None,
    trajectory_dir=None,
    stop_requested=None,
    on_serving=None,
):
    if not hasattr(source, "snapshot_id"):
        raise ValueError("an experiment needs a frozen library snapshot")
    spec = apply_overrides(load_spec(spec_path), tasks, replicates)
    if model not in spec.models:
        raise ValueError(f"model {model} is not part of {spec.name}")
    check_prompt_versions(spec)
    model_row = _model_entry(source, model)
    check_environment(model_row, environment)
    data_dir = data_dir or default_data_dir()
    examples = build_examples(spec, source, data_dir, per_task)
    if not examples:
        raise ValueError(f"{spec.name} has no examples for the selected test tasks")
    check_snapshot_isolation(source, {example["hypothesis"] for example in examples})
    experiment_id = experiment_id_for(spec, source.snapshot_id, per_task)
    sink.create_experiment(
        {
            "id": experiment_id,
            "name": spec.name,
            "spec": {**spec.to_dict(), "per_task": per_task},
            "status": "open",
        }
    )
    done = sink.completed_runs(experiment_id)
    trajectory_dir = trajectory_dir or default_trajectory_dir()
    resolver, _ = session_wiring(source)
    with serving_for(model_row) as counter:
        if on_serving is not None:
            on_serving(counter, model_row)
        popper_kwargs = {**popper_kwargs_for(model_row, counter.port), "plot_agent_architecture": False}
        runs = expand(spec, model_row, examples, popper_kwargs, environment, experiment_id, source.snapshot_id)
        agents = {}
        for run in runs:
            if run_identity(run) in done:
                continue
            if stop_requested is not None and stop_requested():
                break
            flags = flags_for_arm(run.arm)
            if flags not in agents:
                agents[flags] = build_agents(popper_kwargs, flags[0], flags[1], False)
            curator, reviewer, extractor = agents[flags]
            arm_source = source_for_arm(
                run.arm, source, spec.permutation_seeds[run.replicate], run.popper_kwargs["domain"]
            )
            RunSession(
                run,
                arm_source,
                sink,
                curator,
                reviewer,
                extractor,
                trajectory_dir,
                prompt_version_resolver=resolver,
                usage=counter,
            ).run()
    return experiment_id


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.eval.run_experiment")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="db")
    parser.add_argument("--sink", default="db")
    parser.add_argument("--environment", choices=("laptop", "cluster"), default="laptop")
    parser.add_argument("--tasks")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--examples-per-task", type=int)
    parser.add_argument("--trajectory-dir")
    parser.add_argument("--snapshot-out")
    return parser.parse_args(argv)


def _freeze_source(spec, conn, embedder, snapshot_out):
    from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot

    if spec == "db":
        path = Path(snapshot_out or Path.home() / ".popper_upgrade" / "snapshots" / f"{uuid.uuid4()}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        export_snapshot(conn, path)
        return SnapshotSkillSource(path, embedder)
    if spec.startswith("snapshot:"):
        return SnapshotSkillSource(spec.removeprefix("snapshot:"), embedder)
    raise ValueError(f"unknown source {spec}")


def _build_sink(spec, conn, source, embedder):
    from popper_upgrade.db.bundle import BundleSink
    from popper_upgrade.db.runs_repo import RunRepository
    from popper_upgrade.db.skills_repo import SkillRepository

    if spec == "db":
        return RunRepository(conn, SkillRepository(conn, embedder))
    if spec.startswith("bundle:"):
        return BundleSink(spec.removeprefix("bundle:"), source)
    raise ValueError(f"unknown sink {spec}")


def main(argv=None):
    args = parse_args(argv)
    from popper_upgrade.skills.embedding import Embedder

    conn = None
    if "db" in (args.source, args.sink):
        from popper_upgrade.db.connection import connect

        conn = connect()
        conn.autocommit = True
    try:
        embedder = Embedder()
        source = _freeze_source(args.source, conn, embedder, args.snapshot_out)
        sink = _build_sink(args.sink, conn, source, embedder)
        experiment_id = run_experiment(
            args.spec,
            args.model,
            source,
            sink,
            args.environment,
            tasks=args.tasks.split(",") if args.tasks else None,
            replicates=args.replicates,
            per_task=args.examples_per_task,
            trajectory_dir=args.trajectory_dir,
        )
    except (ValueError, KeyError, ServingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        if conn is not None:
            conn.close()
    print(f"experiment {experiment_id} snapshot {source.snapshot_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
