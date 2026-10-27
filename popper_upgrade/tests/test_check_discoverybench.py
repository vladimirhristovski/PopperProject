import hashlib
import json

from popper_upgrade.tools.check_discoverybench import compare, hash_tree, main, write_manifest


def _write(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


FILES = {
    "discoverybench/real/test/a/x.csv": b"1",
    "discoverybench/real/test/a/y.csv": b"2",
    "answer_key/answer_key_real_cleaned_1.csv": b"k",
}


def test_hashes_cover_only_the_scope_and_are_sha256(tmp_path):
    _write(tmp_path, {**FILES, "discoverybench/synth/test/s.csv": b"s", "other.txt": b"o"})

    assert hash_tree(tmp_path) == {name: hashlib.sha256(content).hexdigest() for name, content in FILES.items()}


def test_manifest_round_trip_matches(tmp_path):
    root, manifest = tmp_path / "root", tmp_path / "m.json"
    _write(root, FILES)

    assert write_manifest(root, manifest) == 3
    assert json.loads(manifest.read_text(encoding="utf-8"))["scope"] == ["discoverybench/real/test/", "answer_key/"]
    result = compare(manifest, root)
    assert (len(result["match"]), result["mismatch"], result["missing"], result["extra"]) == (3, [], [], [])


def test_compare_reports_mismatch_missing_and_extra(tmp_path):
    root, manifest = tmp_path / "root", tmp_path / "m.json"
    _write(root, FILES)
    write_manifest(root, manifest)
    (root / "discoverybench/real/test/a/x.csv").write_bytes(b"changed")
    (root / "discoverybench/real/test/a/y.csv").unlink()
    _write(root, {"answer_key/new.csv": b"n"})

    result = compare(manifest, root)

    assert result["mismatch"] == ["discoverybench/real/test/a/x.csv"]
    assert result["missing"] == ["discoverybench/real/test/a/y.csv"]
    assert result["extra"] == ["answer_key/new.csv"]


def test_main_exit_code_and_summary(tmp_path, capsys):
    root, manifest = tmp_path / "root", tmp_path / "m.json"
    _write(root, FILES)
    assert main([str(root), "--manifest", str(manifest), "--write"]) == 0
    capsys.readouterr()

    assert main([str(root), "--manifest", str(manifest)]) == 0
    assert "match=3, mismatch=0, missing=0, extra=0" in capsys.readouterr().out
    (root / "answer_key/answer_key_real_cleaned_1.csv").write_bytes(b"z")
    assert main([str(root), "--manifest", str(manifest)]) == 1
    assert "MISMATCH answer_key/answer_key_real_cleaned_1.csv" in capsys.readouterr().out


def test_shipped_manifest_matches_the_local_layout_when_data_is_staged():
    from popper_upgrade.tools.check_discoverybench import DEFAULT_MANIFEST

    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["scope"] == ["discoverybench/real/test/", "answer_key/"]
    assert "answer_key/answer_key_real_cleaned_1.csv" in manifest["files"]
    assert all(len(digest) == 64 for digest in manifest["files"].values())
