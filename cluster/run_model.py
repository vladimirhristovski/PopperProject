import argparse
import sys
from pathlib import Path


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cluster.common import run_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--bundle-dir", required=True)
    args = parser.parse_args(argv)
    return run_model(args.model, Path(args.snapshot), Path(args.bundle_dir))


if __name__ == "__main__":
    sys.exit(main())
