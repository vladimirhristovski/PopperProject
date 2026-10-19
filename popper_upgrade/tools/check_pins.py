import re
import sys
from pathlib import Path

_PIN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def _normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def _read_pins(path):
    pins = {}
    for line in Path(path).read_text().splitlines():
        match = _PIN.match(line)
        if match:
            pins[_normalize(match.group(1))] = match.group(2)
    return pins


def check_pins(a, b):
    pins_a = _read_pins(a)
    pins_b = _read_pins(b)
    return [
        f"{name}: {pins_a[name]} != {pins_b[name]}"
        for name in sorted(pins_a.keys() & pins_b.keys())
        if pins_a[name] != pins_b[name]
    ]


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    mismatches = check_pins(Path(args[0]), Path(args[1]))
    for line in mismatches:
        print(line)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
