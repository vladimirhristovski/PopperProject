import argparse
import sys

from local.common import build_parser, run_paid

PROVIDER_MODELS = {
    "grok": ("grok-4.7", "grok-4.20-0309-non-reasoning"),
    "deepseek": ("deepseek-v4-pro", "deepseek-flash"),
}


def main(argv=None):
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--provider", choices=sorted(PROVIDER_MODELS), required=True)
    known, rest = first.parse_known_args(argv)
    parser = build_parser("python -m local.run_other_api --provider " + known.provider, PROVIDER_MODELS[known.provider])
    args = parser.parse_args(rest)
    return run_paid(known.provider, args)


if __name__ == "__main__":
    sys.exit(main())
