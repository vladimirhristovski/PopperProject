import argparse
import os
import sys
from pathlib import Path

from popper_upgrade.db.registry_repo import RegistryRepository, prompt_hash
from popper_upgrade.db.runs_repo import RunRepository
from popper_upgrade.db.skills_repo import SkillRepository
from popper_upgrade.db.snapshot import SnapshotSkillSource
from popper_upgrade.models import ServingError, check_environment, popper_kwargs_for, serving_for
from popper_upgrade.reviewer.reviewer import Reviewer
from popper_upgrade.session import RunSession, RunSpec
from popper_upgrade.skills.curator import SkillCurator
from popper_upgrade.skills.embedding import Embedder
from popper_upgrade.skills.extractor import SkillExtractor

ARM_FLAGS = {
    "baseline": (False, False),
    "skills": (True, False),
    "skills_reviewer": (True, True),
}
DOMAIN = "social science"
CONFIGURE_KWARGS = {
    "alpha": 0.1,
    "max_num_of_tests": 5,
    "max_retry": 3,
    "time_limit": 60,
    "aggregate_test": "E-value",
    "relevance_checker": True,
    "use_react_agent": True,
}
REPO_ROOT = Path(__file__).resolve().parent.parent


def default_data_dir():
    return os.environ.get("POPPER_DATA_DIR", str(REPO_ROOT / "data"))


def default_trajectory_dir():
    return os.environ.get("POPPER_TRAJECTORY_DIR", str(Path.home() / ".popper_upgrade" / "trajectories"))


def snapshot_prompt_resolver(snapshot):
    versions = {(row["role"], row["hash"]): row["id"] for row in snapshot.registry["prompt_versions"]}

    def resolve(role, text):
        return versions.get((role, prompt_hash(role, text)))

    return resolve


def session_wiring(source):
    if isinstance(source, SnapshotSkillSource):
        return snapshot_prompt_resolver(source), None
    if isinstance(source, SkillRepository):
        return RegistryRepository(source.conn).prompt_version, source.record_usage
    return None, None


def build_agents(popper_kwargs, skills_enabled, reviewer_enabled, needs_extractor):
    agent_kwargs = {"port": popper_kwargs.get("server_port"), "api_key": popper_kwargs.get("api_key", "EMPTY")}
    llm = popper_kwargs["llm"]
    curator = SkillCurator(llm, **agent_kwargs) if skills_enabled else None
    reviewer = Reviewer(llm, **agent_kwargs) if reviewer_enabled else None
    extractor = SkillExtractor(llm, **agent_kwargs) if skills_enabled and needs_extractor else None
    return curator, reviewer, extractor


def run_hypotheses(
    model,
    hypotheses,
    arm,
    source,
    sink,
    environment,
    human_review=False,
    learn=False,
    seed=None,
    launched_by=None,
    overrides_provider=None,
    on_session_start=None,
    usage=None,
    trajectory_dir=None,
    stop_requested=None,
    configure_kwargs=None,
    data_dir=None,
):
    check_environment(model, environment)
    if arm not in ARM_FLAGS:
        raise ValueError(f"unknown arm {arm}")
    skills_enabled, reviewer_enabled = ARM_FLAGS[arm]
    data_dir = data_dir or default_data_dir()
    (Path(data_dir) / "bio_database").mkdir(parents=True, exist_ok=True)
    popper_kwargs = {
        **popper_kwargs_for(model, getattr(usage, "port", None)),
        "plot_agent_architecture": False,
        "domain": DOMAIN,
    }
    needs_extractor = learn or any(h.get("split") == "train" for h in hypotheses)
    curator, reviewer, extractor = build_agents(popper_kwargs, skills_enabled, reviewer_enabled, needs_extractor)
    resolver, recorder = session_wiring(source)
    run_ids = []
    for hypothesis in hypotheses:
        if stop_requested is not None and stop_requested():
            break
        spec = RunSpec(
            hypothesis=hypothesis["text"],
            hypothesis_id=hypothesis.get("id"),
            split=hypothesis.get("split", "none"),
            arm=arm,
            model=model,
            popper_kwargs=popper_kwargs,
            register_kwargs={"data_path": str(data_dir), "loader_type": "custom"},
            configure_kwargs=dict(configure_kwargs or CONFIGURE_KWARGS),
            skills_enabled=skills_enabled,
            reviewer_enabled=reviewer_enabled,
            human_review=human_review,
            learn=learn,
            environment=environment,
            launched_by=launched_by,
            seed=seed,
            library_snapshot_id=getattr(source, "snapshot_id", None),
        )
        session = RunSession(
            spec,
            source,
            sink,
            curator,
            reviewer,
            extractor,
            trajectory_dir or default_trajectory_dir(),
            overrides_provider=overrides_provider,
            prompt_version_resolver=resolver,
            usage_recorder=recorder,
            usage=usage,
            on_start=on_session_start,
        )
        run_ids.append(session.run())
    return run_ids


class RecordingSink:
    def __init__(self, sink):
        self._sink = sink
        self.runs = {}

    def create_run(self, config):
        run_id = self._sink.create_run(config)
        self.runs[run_id] = dict(config)
        return run_id

    def finish_run(self, run_id, result):
        self._sink.finish_run(run_id, result)
        self.runs.setdefault(run_id, {}).update(result)

    def __getattr__(self, name):
        return getattr(self._sink, name)


def format_table(run_ids, runs):
    header = ("hypothesis", "status", "res_stat", "n_tests", "tokens", "time_s")
    rows = [header]
    for run_id in run_ids:
        run = runs.get(run_id, {})
        res_stat = run.get("res_stat")
        wall = run.get("wall_time_s")
        rows.append(
            (
                str(run.get("hypothesis", run_id))[:50],
                str(run.get("status", "?")),
                "-" if res_stat is None else f"{res_stat:.4g}",
                str(run.get("n_tests", "-")),
                f"{run.get('tokens_in', '-')}/{run.get('tokens_out', '-')}",
                "-" if wall is None else f"{wall:.1f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hypothesis-set", required=True)
    parser.add_argument("--hypothesis-index", type=int, action="append")
    parser.add_argument("--arm", choices=sorted(ARM_FLAGS), required=True)
    parser.add_argument("--human-review", action="store_true")
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source", default="db")
    parser.add_argument("--sink", default="db")
    parser.add_argument("--environment", choices=("laptop", "cluster"), default="laptop")
    parser.add_argument("--trajectory-dir")
    return parser.parse_args(argv)


def _open_connection():
    from popper_upgrade.db.connection import connect

    conn = connect()
    conn.autocommit = True
    return conn


def _lookup(args, source, conn):
    if isinstance(source, SnapshotSkillSource):
        models = {m["name"]: m for m in source.registry["models"]}
        sets = source.registry["hypothesis_sets"]
        if args.model not in models:
            raise KeyError(args.model)
        if args.hypothesis_set not in sets:
            raise KeyError(args.hypothesis_set)
        return models[args.model], sets[args.hypothesis_set]
    registry = RegistryRepository(conn)
    return registry.model(args.model), registry.hypothesis_set(args.hypothesis_set)


def _build_source(spec, conn):
    if spec == "db":
        return SkillRepository(conn, Embedder())
    if spec.startswith("snapshot:"):
        return SnapshotSkillSource(spec.removeprefix("snapshot:"), Embedder())
    raise ValueError(f"unknown source {spec}")


def _build_sink(spec, conn, source):
    if spec == "db":
        skills = source if isinstance(source, SkillRepository) else SkillRepository(conn, Embedder())
        return RunRepository(conn, skills)
    if spec.startswith("bundle:"):
        if not isinstance(source, SnapshotSkillSource):
            raise ValueError("a bundle sink needs a snapshot source")
        from popper_upgrade.db.bundle import BundleSink

        return BundleSink(spec.removeprefix("bundle:"), source)
    raise ValueError(f"unknown sink {spec}")


def main(argv=None):
    args = parse_args(argv)
    conn = _open_connection() if "db" in (args.source, args.sink) else None
    try:
        source = _build_source(args.source, conn)
        sink = _build_sink(args.sink, conn, source)
        model, hypotheses = _lookup(args, source, conn)
        check_environment(model, args.environment)
        if args.hypothesis_index:
            hypotheses = [hypotheses[i] for i in args.hypothesis_index]
    except (ValueError, KeyError, IndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    recorder = RecordingSink(sink)
    try:
        with serving_for(model) as counter:
            run_ids = run_hypotheses(
                model,
                hypotheses,
                args.arm,
                source,
                recorder,
                args.environment,
                human_review=args.human_review,
                learn=args.learn,
                seed=args.seed,
                usage=counter,
                trajectory_dir=args.trajectory_dir,
            )
    except ServingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(format_table(run_ids, recorder.runs))
    return 0 if all(recorder.runs.get(run_id, {}).get("status") == "completed" for run_id in run_ids) else 1


if __name__ == "__main__":
    sys.exit(main())
