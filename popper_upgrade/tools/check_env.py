import argparse
import re
import sys
from pathlib import Path

POPPER_PACKAGE = "popper-agent"
TAG = re.compile(r"\+\w+$")
PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")
ENTRY = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = PIN.match(line)
        if match:
            pins[_normalize(match.group(1))] = match.group(2)
    return pins


def read_freeze(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = ENTRY.match(line.strip())
        if match and not line.startswith(("#", "-")):
            entries[_normalize(match.group(1))] = match.group(2).strip()
    return entries


def _version(text):
    return TAG.sub("", text.removeprefix("==").strip())


def env_diff(requirements, freeze, popper_sha=None):
    pins = _read_pins(requirements)
    installed = read_freeze(freeze)
    problems = []
    for name, version in sorted(pins.items()):
        if name not in installed:
            problems.append(f"{name}: pinned {version} but not installed")
        elif _version(installed[name]) != version:
            problems.append(f"{name}: pinned {version} but installed {_version(installed[name])}")
    if popper_sha is not None:
        found = installed.get(POPPER_PACKAGE, "")
        if not found.endswith(f"@{popper_sha}"):
            problems.append(f"{POPPER_PACKAGE}: expected commit {popper_sha} but found {found or 'nothing'}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("requirements")
    parser.add_argument("freeze")
    parser.add_argument("--popper-sha")
    args = parser.parse_args(argv)
    problems = env_diff(Path(args.requirements), Path(args.freeze), args.popper_sha)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
