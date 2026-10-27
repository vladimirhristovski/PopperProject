import argparse
import os
import sys
from pathlib import Path

from popper_upgrade.models import ServingError

SPECS_DIR = Path(__file__).resolve().parent.parent / "popper_upgrade" / "eval" / "specs"
SPEC_PATH = SPECS_DIR / "discoverybench_v1.json"
QWEN_SPEC_PATH = SPECS_DIR / "qwen_round.json"
KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}
ENVIRONMENT = "laptop"


def build_parser(prog, models, spec=SPEC_PATH, library_required=False):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--model", choices=models, required=True)
    parser.add_argument("--spec", default=str(spec))
    parser.add_argument("--library-from", required=library_required)
    parser.add_argument("--tasks")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--examples-per-task", type=int)
    parser.add_argument("--report-dir")
    return parser


def missing_key(provider):
    name = KEY_ENVS[provider]
    if os.environ.get(name):
        return None
    return f"error: environment variable {name} is not set; set it and run again. Nothing was run."


def load_library(path, embedder):
    if not path:
        return None
    if not Path(path).is_file():
        raise ValueError(f"library snapshot not found: {path}")
    from popper_upgrade.db.snapshot import SnapshotSkillSource

    return SnapshotSkillSource(path, embedder)


def run_paid(provider, args):
    problem = missing_key(provider)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2
    from popper_upgrade.db.connection import connect
    from popper_upgrade.db.registry_repo import RegistryRepository
    from popper_upgrade.eval.cost_guard import CostGuard
    from popper_upgrade.eval.learning import db_learning
    from popper_upgrade.eval.report import build_report, write_report
    from popper_upgrade.eval.run_experiment import (
        _build_sink,
        _freeze_source,
        default_snapshot_dir,
        run_experiment,
    )
    from popper_upgrade.models import check_environment
    from popper_upgrade.serving.model_status import check_model_status, format_status
    from popper_upgrade.skills.embedding import Embedder

    conn = connect()
    conn.autocommit = True
    try:
        model = RegistryRepository(conn).model(args.model)
        check_environment(model, ENVIRONMENT)
        print(format_status(check_model_status(model)))
        embedder = Embedder()
        library = load_library(args.library_from, embedder)
        source = _freeze_source("db", conn, embedder, None)
        sink = _build_sink("db", conn, source, embedder)
        learning = None if library is not None else db_learning(conn, embedder, sink, default_snapshot_dir())
        guard = CostGuard(sink.model_costs) if model.get("pricing") else None
        experiment_id = run_experiment(
            args.spec,
            args.model,
            source,
            sink,
            ENVIRONMENT,
            tasks=args.tasks.split(",") if args.tasks else None,
            replicates=args.replicates,
            per_task=args.examples_per_task,
            learning=learning,
            library=library,
            budget=guard,
            on_serving=guard.attach if guard is not None else None,
        )
        out_dir = Path(args.report_dir or f"popper_report_{args.model.replace('/', '_')}")
        write_report(build_report(conn, experiment_id), out_dir)
        if guard is not None:
            print(guard.summary())
    except (ValueError, KeyError, ServingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(f"experiment {experiment_id}; report in {out_dir.resolve()}")
    if guard is not None and guard.exhausted:
        print("error: the budget stopped the experiment before all runs were done", file=sys.stderr)
        return 3
    return 0
