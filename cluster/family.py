import argparse

from cluster.job import run_job

FAMILY_JOBS = {
    "llama": {"small": "llama-small", "large": "llama-large"},
    "qwen": {"small": "qwen-small", "large": "qwen-large"},
    "nvidia": {"small": "nvidia-small", "large": "nvidia-large"},
}
ANCHOR_JOB = "llama-anchor"
SIZES = ("small", "large")


def job_for(family, size=None, anchor=False):
    if anchor:
        return ANCHOR_JOB
    return FAMILY_JOBS[family][size]


def parse_family_args(family, argv):
    parser = argparse.ArgumentParser(prog=f"python3 -m cluster.run_{family}")
    parser.add_argument("--size", choices=SIZES)
    if family == "llama":
        parser.add_argument("--anchor", action="store_true")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--examples-per-task", type=int)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args(argv)
    anchor = getattr(args, "anchor", False)
    if anchor and args.size:
        parser.error("--anchor and --size cannot be used together")
    if not anchor and not args.size:
        parser.error("--size small or --size large is required" + (" (or --anchor)" if family == "llama" else ""))
    return args, anchor


def run_family(family, argv=None):
    args, anchor = parse_family_args(family, argv)
    return run_job(
        job_for(family, args.size, anchor),
        args.workdir,
        args.skip_preflight,
        args.replicates,
        args.examples_per_task,
        args.max_hours,
    )
