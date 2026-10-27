import argparse
import hashlib
import json
import sys
from pathlib import Path

SCOPE = ("discoverybench/real/test/", "answer_key/")
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "registry" / "discoverybench_manifest.json"


def hash_tree(root, scope=SCOPE):
    root = Path(root)
    hashes = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_file() and name.startswith(scope):
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def write_manifest(root, manifest):
    payload = {"scope": list(SCOPE), "files": hash_tree(root)}
    Path(manifest).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return len(payload["files"])


def compare(manifest, root):
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    expected = payload["files"]
    actual = hash_tree(root, tuple(payload["scope"]))
    common = expected.keys() & actual.keys()
    return {
        "match": sorted(name for name in common if expected[name] == actual[name]),
        "mismatch": sorted(name for name in common if expected[name] != actual[name]),
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="check_discoverybench.py")
    parser.add_argument("root")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.write:
        print(f"wrote {write_manifest(args.root, args.manifest)} hashes to {args.manifest}")
        return 0
    result = compare(args.manifest, args.root)
    for key in ("mismatch", "missing", "extra"):
        for name in result[key]:
            print(f"{key.upper()} {name}")
    print(", ".join(f"{key}={len(names)}" for key, names in result.items()))
    return 0 if not (result["mismatch"] or result["missing"] or result["extra"]) else 1


if __name__ == "__main__":
    sys.exit(main())
