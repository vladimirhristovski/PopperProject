import sys
from pathlib import Path


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from cluster.family import run_family

    return run_family("qwen", argv)


if __name__ == "__main__":
    sys.exit(main())
