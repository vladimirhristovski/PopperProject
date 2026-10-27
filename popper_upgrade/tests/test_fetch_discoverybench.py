import hashlib
import io
import tarfile
from pathlib import PurePosixPath

import pytest

from popper_upgrade.tools import fetch_discoverybench as fetch
from popper_upgrade.tools.check_discoverybench import hash_tree


def _archive(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


MEMBERS = {
    "popper_data_processed/benchmark/discoverybench/discoverybench/real/test/t/m.json": b"{}",
    "popper_data_processed/benchmark/discoverybench/answer_key/answer_key_real_cleaned_1.csv": b"k",
    "popper_data_processed/bio_database/big.bin": b"x",
    "popper_data_processed/benchmark/targetval/gt.csv": b"y",
}


def test_relative_name_strips_any_prefix_and_rejects_traversal():
    assert fetch.relative_name("a/b/benchmark/discoverybench/answer_key/k.csv") == PurePosixPath("answer_key/k.csv")
    assert fetch.relative_name("benchmark/discoverybench/x.csv") == PurePosixPath("x.csv")
    assert fetch.relative_name("benchmark/discoverybench") is None
    assert fetch.relative_name("benchmark/other/x.csv") is None
    assert fetch.relative_name("benchmark/discoverybench/../x.csv") is None


def test_extract_keeps_only_discoverybench_files(tmp_path):
    archive = tmp_path / "a.tar.gz"
    _archive(archive, MEMBERS)

    assert fetch.extract(archive, tmp_path / "out") == 2
    assert sorted(hash_tree(tmp_path / "out")) == [
        "answer_key/answer_key_real_cleaned_1.csv",
        "discoverybench/real/test/t/m.json",
    ]
    assert not (tmp_path / "out" / "bio_database").exists()


def test_verify_rejects_wrong_size_or_md5():
    fetch.verify(3, "abc", 3, "abc")
    with pytest.raises(ValueError, match="expected 3 bytes md5 abc"):
        fetch.verify(4, "abc", 3, "abc")
    with pytest.raises(ValueError):
        fetch.verify(3, "abd", 3, "abc")


def test_main_downloads_verifies_extracts_and_removes_the_archive(tmp_path):
    source = tmp_path / "src.tar.gz"
    _archive(source, MEMBERS)
    data = source.read_bytes()
    args = [
        "--md5", hashlib.md5(data).hexdigest(), "--size", str(len(data)),
        "--archive", str(tmp_path / "dl.tar.gz"), "--dest", str(tmp_path / "out"), "--url", source.as_uri(),
    ]

    assert fetch.main(args) == 0
    assert not (tmp_path / "dl.tar.gz").exists()
    assert (tmp_path / "out" / "answer_key" / "answer_key_real_cleaned_1.csv").read_bytes() == b"k"
    bad = [*args[:1], "0" * 32, *args[2:]]
    with pytest.raises(ValueError):
        fetch.main(bad)
