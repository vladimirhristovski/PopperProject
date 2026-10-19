import sys
from types import SimpleNamespace

import pytest

from cluster import upload


@pytest.fixture
def archive(tmp_path, monkeypatch):
    for name in ("HF_TOKEN", "HF_USERNAME", "POPPER_RESULTS_REPO"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "popper_results_demo_20261019T000000Z.tar.gz"
    path.write_bytes(b"archive")
    return path


def _fake_hub(monkeypatch, api):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda token=None: api))


class _Api:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))
        if self.fail_on == "create_repo":
            raise OSError("network down")

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        if self.fail_on == "upload_file":
            raise OSError("network down")


def test_upload_is_skipped_without_a_token(archive, monkeypatch, capsys):
    api = _Api()
    _fake_hub(monkeypatch, api)

    assert upload.maybe_upload(archive) is None
    assert api.calls == []
    assert "no HF_TOKEN" in capsys.readouterr().out


def test_upload_needs_a_results_repo(archive, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "token")
    api = _Api()
    _fake_hub(monkeypatch, api)

    assert upload.maybe_upload(archive) is None
    assert api.calls == []
    assert "POPPER_RESULTS_REPO" in capsys.readouterr().out


def test_upload_goes_to_the_private_default_repo(archive, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("HF_USERNAME", "vlad")
    api = _Api()
    _fake_hub(monkeypatch, api)

    url = upload.maybe_upload(archive)

    assert url == f"https://huggingface.co/datasets/vlad/popper-results/blob/main/{archive.name}"
    created, uploaded = api.calls
    assert created[1] == {"repo_id": "vlad/popper-results", "repo_type": "dataset", "private": True, "exist_ok": True}
    assert uploaded[1]["path_in_repo"] == archive.name
    assert uploaded[1]["repo_type"] == "dataset"


def test_explicit_repo_overrides_the_default(archive, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("HF_USERNAME", "vlad")
    monkeypatch.setenv("POPPER_RESULTS_REPO", "lab/results")
    _fake_hub(monkeypatch, _Api())

    assert upload.maybe_upload(archive).startswith("https://huggingface.co/datasets/lab/results/")


@pytest.mark.parametrize("stage", ["create_repo", "upload_file"])
def test_upload_failure_keeps_the_archive_and_returns_none(archive, monkeypatch, capsys, stage):
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("POPPER_RESULTS_REPO", "lab/results")
    _fake_hub(monkeypatch, _Api(fail_on=stage))

    assert upload.maybe_upload(archive) is None
    assert archive.read_bytes() == b"archive"
    assert "Automatic upload failed" in capsys.readouterr().out
