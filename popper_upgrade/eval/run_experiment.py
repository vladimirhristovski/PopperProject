import argparse
import json
import sys
import time
import urllib.request
import uuid
from dataclasses import replace
from pathlib import Path

from popper_upgrade.db.registry_repo import prompt_hash
from popper_upgrade.eval.benchmark import discoverybench_examples, gender_examples
from popper_upgrade.eval.controls import flags_for_arm, source_for_arm
from popper_upgrade.eval.learning import db_learning
from popper_upgrade.eval.spec import (
    LEARN_ARM,
    check_snapshot_isolation,
    expand,
    expand_learning,
    load_spec,
    planned_counts,
    run_identity,
    source_parts,
)
from popper_upgrade.models import ServingError, check_environment, popper_kwargs_for, serving_for
from popper_upgrade.reviewer.reviewer import REVIEWER_SYSTEM_PROMPT
from popper_upgrade.run import build_agents, default_data_dir, default_trajectory_dir, session_wiring
from popper_upgrade.session import RunSession
from popper_upgrade.skills.curator import CURATOR_SYSTEM_PROMPT
from popper_upgrade.skills.extractor import EXTRACTOR_SYSTEM_PROMPT

EXIT_SERVER_EXITED = 4
HEALTH_PATH = "/v1/models"
HTTP_OK = 200
EXPERIMENT_NAMESPACE = uuid.UUID("5d3c1c53-7a55-4c4e-9a64-3c2f0b4f6b0a")
PROMPTS = {
    "curator": CURATOR_SYSTEM_PROMPT,
    "reviewer": REVIEWER_SYSTEM_PROMPT,
    "extractor": EXTRACTOR_SYSTEM_PROMPT,
}


class ServerExited(ServingError):
    pass


def server_responding(port, attempts=2, pause=1.0):
    for attempt in range(attempts):
        if attempt:
            time.sleep(pause)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{HEALTH_PATH}", timeout=15) as response:
                if response.status == HTTP_OK:
                    return True
        except Exception:
            continue
    return False


def _liveness(counter, model_row):
    alive = getattr(counter, "alive", None)
    if alive is None and model_row["backend"] == "vllm":
        return lambda: server_responding(counter.port)
    return alive or (lambda: True)


def _require_alive(context):
    if context["alive"]():
        context["started"] += 1
        return
    detail = context["counter"].upstream_error() if hasattr(context["counter"], "upstream_error") else None
    message = (
        f"the vLLM server is no longer responding; stopped before starting the next run "
        f"after {context['started']} runs this session"
    )
    raise ServerExited(f"{message} ({detail})" if detail else message)


def default_snapshot_dir():
    return Path.home() / ".popper_upgrade" / "snapshots"


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
        stripped = [{k: v for k, v in part.items() if k != "replicates"} for part in source_parts(spec.source)]
        changes["source"] = {**spec.source, "parts": stripped} if "parts" in spec.source else stripped[0]
    return replace(spec, **changes) if changes else spec


def _part_examples(part, source, data_dir, per_task, tasks, sampled):
    if part["kind"] == "discoverybench":
        path = Path(data_dir) / part["path"]
        return discoverybench_examples(str(path), set(tasks), per_task, part.get("sample") if sampled else None)
    sets = source.registry["hypothesis_sets"]
    name = part["hypothesis_set"]
    if name not in sets:
        raise ValueError(f"hypothesis set {name} is not in the snapshot registry")
    permutable = {row["name"]: row["permutable_columns"] for row in source.registry["datasets"]}
    if not permutable:
        raise ValueError("the snapshot registry has no datasets")
    domain = part.get("domain", "social science")
    examples = []
    task = part.get("task", "gender_bias")
    if task in tasks:
        examples += gender_examples(sets[name], data_dir, permutable, task, domain)
    train_task = part.get("train_task")
    if train_task and train_task in tasks:
        train = [{"id": None, "text": text} for text in part.get("train_hypotheses", [])]
        examples += gender_examples(train, data_dir, permutable, train_task, domain)
    return examples


def build_examples(spec, source, data_dir, per_task=None, tasks=None, sampled=True):
    tasks = spec.test_tasks if tasks is None else tasks
    examples = []
    for part in source_parts(spec.source):
        built = _part_examples(part, source, data_dir, per_task, tasks, sampled)
        counts = part.get("replicates")
        if counts:
            for example in built:
                example["replicates"] = counts["permuted" if example["permuted"] else "original"]
        examples += built
    return examples


def experiment_id_for(spec, snapshot_id, per_task=None, phase=None):
    body = {key: value for key, value in spec.to_dict().items() if key != "models"}
    body["per_task"] = per_task
    canonical = json.dumps(body, sort_keys=True)
    prefix = f"{phase}:" if phase else ""
    return str(uuid.uuid5(EXPERIMENT_NAMESPACE, f"{prefix}{snapshot_id}:{canonical}"))


def _model_entry(source, name):
    model = next((m for m in source.registry["models"] if m["name"] == name), None)
    if model is None:
        raise ValueError(f"model {name} is not in the snapshot registry")
    return model


def _open_experiment(sink, spec, experiment_id, per_task, **extra):
    sink.create_experiment(
        {
            "id": experiment_id,
            "name": spec.name,
            "spec": {**spec.to_dict(), "per_task": per_task, **extra},
            "status": "open",
        }
    )


def _learn_phase(spec, learning, model_row, examples, popper_kwargs, environment, base_id, per_task, counter, context):
    learn_id = experiment_id_for(spec, base_id, per_task, "learn")
    _open_experiment(
        learning.sink,
        spec,
        learn_id,
        per_task,
        phase="learn",
        base_snapshot_id=base_id,
        planned_runs={LEARN_ARM: len(examples)},
    )
    done = learning.sink.completed_runs(learn_id)
    resolver, recorder = session_wiring(learning.source)
    agents = None
    for run in expand_learning(spec, model_row, examples, popper_kwargs, environment, learn_id, base_id):
        if run_identity(run) in done:
            continue
        if context["stop_requested"] is not None and context["stop_requested"]():
            break
        if context["budget"] is not None and not context["budget"].may_start("learn", run.arm):
            break
        _require_alive(context)
        if agents is None:
            agents = build_agents(popper_kwargs, True, True, True)
        curator, reviewer, extractor = agents
        RunSession(
            run,
            learning.source,
            learning.sink,
            curator,
            reviewer,
            extractor,
            context["trajectory_dir"],
            prompt_version_resolver=resolver,
            usage_recorder=recorder,
            usage=counter,
        ).run()
    return learn_id


def _evaluate_phase(
    spec, source, sink, model_row, examples, popper_kwargs, environment, experiment_id, counter, context
):
    done = sink.completed_runs(experiment_id)
    resolver, _ = session_wiring(source)
    runs = expand(spec, model_row, examples, popper_kwargs, environment, experiment_id, source.snapshot_id)
    agents = {}
    stop_requested = context["stop_requested"]
    for run in runs:
        if run_identity(run) in done:
            continue
        if stop_requested is not None and stop_requested():
            break
        if context["budget"] is not None and not context["budget"].may_start("evaluate", run.arm):
            break
        _require_alive(context)
        flags = flags_for_arm(run.arm)
        if flags not in agents:
            agents[flags] = build_agents(popper_kwargs, flags[0], flags[1], False)
        curator, reviewer, extractor = agents[flags]
        arm_source = source_for_arm(run.arm, source, spec.permutation_seeds[run.replicate], run.popper_kwargs["domain"])
        RunSession(
            run,
            arm_source,
            sink,
            curator,
            reviewer,
            extractor,
            context["trajectory_dir"],
            prompt_version_resolver=resolver,
            usage=counter,
        ).run()


def _use_library(library, source, learning):
    if library is None:
        return source, learning
    if not hasattr(library, "snapshot_id"):
        raise ValueError("a supplied library must be a frozen snapshot")
    return library, None


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
    learning=None,
    budget=None,
    library=None,
    log_dir=None,
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
    source, learning = _use_library(library, source, learning)
    check_snapshot_isolation(source, {example["hypothesis"] for example in examples})
    learn_examples = []
    if learning is not None and spec.train_tasks:
        train = build_examples(spec, source, data_dir, per_task, spec.train_tasks, sampled=False)
        learn_examples = [example for example in train if not example["permuted"]]
        if not learn_examples:
            raise ValueError(f"{spec.name} has no examples for its train tasks")
    base_id = source.snapshot_id
    planned = planned_counts(spec, examples)
    if budget is not None:
        plan = {"evaluate": planned}
        if learn_examples:
            plan = {"learn": {LEARN_ARM: len(learn_examples)}, **plan}
        budget.begin(plan, spec_path, model_row)
    experiment_id = None
    if not learn_examples:
        experiment_id = experiment_id_for(spec, base_id, per_task)
        _open_experiment(sink, spec, experiment_id, per_task, planned_runs=planned)
    trajectory_dir = trajectory_dir or default_trajectory_dir()
    with serving_for(model_row, log_dir) as counter:
        if on_serving is not None:
            on_serving(counter, model_row)
        popper_kwargs = {**popper_kwargs_for(model_row, counter.port), "plot_agent_architecture": False}
        context = {
            "stop_requested": stop_requested,
            "trajectory_dir": trajectory_dir,
            "budget": budget,
            "counter": counter,
            "alive": _liveness(counter, model_row),
            "started": 0,
        }
        if learn_examples:
            learn_id = _learn_phase(
                spec, learning, model_row, learn_examples, popper_kwargs, environment, base_id, per_task, counter,
                context,
            )
            if (stop_requested is not None and stop_requested()) or (budget is not None and budget.exhausted):
                return learn_id
            source = learning.freeze(learn_id)
            check_snapshot_isolation(source, {example["hypothesis"] for example in examples})
            experiment_id = experiment_id_for(spec, source.snapshot_id, per_task)
            extra = {
                "phase": "evaluate",
                "base_snapshot_id": base_id,
                "learn_experiment_id": learn_id,
                "planned_runs": planned,
            }
            _open_experiment(sink, spec, experiment_id, per_task, **extra)
        _evaluate_phase(
            spec, source, sink, model_row, examples, popper_kwargs, environment, experiment_id, counter, context
        )
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
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--library-from")
    return parser.parse_args(argv)


def _freeze_source(spec, conn, embedder, snapshot_out):
    from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot

    if spec == "db":
        path = Path(snapshot_out or default_snapshot_dir() / f"{uuid.uuid4()}.json")
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
    from popper_upgrade.db.snapshot import SnapshotSkillSource
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
        learning = None
        library = None
        if args.library_from:
            if args.learn:
                raise ValueError("--library-from and --learn cannot be used together")
            library = SnapshotSkillSource(args.library_from, embedder)
        if args.learn:
            if conn is None or args.sink != "db":
                raise ValueError("--learn needs the database as source and sink")
            snapshot_dir = Path(args.snapshot_out).parent if args.snapshot_out else default_snapshot_dir()
            learning = db_learning(conn, embedder, sink, snapshot_dir)
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
            learning=learning,
            library=library,
        )
    except ServerExited as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_SERVER_EXITED
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
