import argparse
import sys
from pathlib import Path


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cluster.common import run_experiment_job

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--examples-per-task", type=int)
    args = parser.parse_args(argv)
    return run_experiment_job(
        args.spec,
        args.model,
        Path(args.snapshot),
        Path(args.bundle_dir),
        tasks=args.tasks.split(",") if args.tasks else None,
        replicates=args.replicates,
        per_task=args.examples_per_task,
    )


if __name__ == "__main__":
    sys.exit(main())
