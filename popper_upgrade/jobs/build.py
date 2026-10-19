import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from popper_upgrade.db.snapshot import SnapshotSkillSource, export_snapshot
from popper_upgrade.eval.run_experiment import build_examples, check_prompt_versions
from popper_upgrade.eval.spec import check_snapshot_isolation, load_spec
from popper_upgrade.models import check_environment
from popper_upgrade.run import default_data_dir
from popper_upgrade.session import git_sha

SNAPSHOT = "snapshot.json"
SPEC = "spec.json"


def build_job(conn, name, model, spec_path, out_dir, embedder, data_dir=None):
    spec = load_spec(spec_path)
    if model not in spec.models:
        raise ValueError(f"model {model} is not part of {spec.name}")
    check_prompt_versions(spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / SNAPSHOT
    try:
        export_snapshot(conn, snapshot_path)
        source = SnapshotSkillSource(snapshot_path, embedder)
        row = next((m for m in source.registry["models"] if m["name"] == model), None)
        if row is None:
            raise ValueError(f"model {model} is not in the registry")
        check_environment(row, "cluster")
        examples = build_examples(spec, source, data_dir or default_data_dir())
        if not examples:
            raise ValueError(f"{spec.name} has no examples for its test tasks")
        check_snapshot_isolation(source, {example["hypothesis"] for example in examples})
    except BaseException:
        snapshot_path.unlink(missing_ok=True)
        raise
    shutil.copyfile(spec_path, out_dir / SPEC)
    job = {
        "name": name,
        "model": model,
        "spec": SPEC,
        "snapshot": SNAPSHOT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
    }
    (out_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m popper_upgrade.jobs.build")
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from popper_upgrade.db.connection import connect
    from popper_upgrade.skills.embedding import Embedder

    conn = connect()
    conn.autocommit = True
    try:
        job = build_job(conn, args.name, args.model, args.spec, args.out, Embedder())
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(f"job {job['name']} written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
