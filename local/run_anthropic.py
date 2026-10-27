import sys

from local.common import QWEN_SPEC_PATH, build_parser, run_paid

MODELS = ("claude-sonnet-5-5",)


def main(argv=None):
    args = build_parser("python -m local.run_anthropic", MODELS, QWEN_SPEC_PATH, library_required=True).parse_args(argv)
    return run_paid("anthropic", args)


if __name__ == "__main__":
    sys.exit(main())
