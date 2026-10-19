import re
import sys
from pathlib import Path

_ENTRY = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_freeze(path):
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = _ENTRY.match(line.strip())
        if match and not line.startswith(("#", "-")):
            entries[_normalize(match.group(1))] = match.group(2).strip()
    return entries


def freeze_diff(base, upgrade):
    before = read_freeze(base)
    after = read_freeze(upgrade)
    added = sorted(f"{name} {after[name]}".strip() for name in after.keys() - before.keys())
    common = before.keys() & after.keys()
    changed = sorted(f"{name}: {before[name]} -> {after[name]}" for name in common if before[name] != after[name])
    removed = sorted(before.keys() - after.keys())
    return added, changed, removed


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    added, changed, removed = freeze_diff(args[0], args[1])
    for line in added:
        print(f"added   {line}")
    for line in changed:
        print(f"CHANGED {line}")
    for line in removed:
        print(f"REMOVED {line}")
    return 1 if changed or removed else 0


if __name__ == "__main__":
    sys.exit(main())
