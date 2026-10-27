import urllib.error
from pathlib import Path

import pytest

from cluster import fetch

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "abc123"


@pytest.fixture
def hf(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    return tmp_path / "hf"


def test_download_runs_snapshot_download_into_hf_home_and_reports_progress(hf, monkeypatch):
    calls = []
    said = []
    monkeypatch.setattr(fetch, "hub_unreachable", lambda: None)
    monkeypatch.setattr(
        fetch, "snapshot_download", lambda model, **kwargs: calls.append((model, kwargs)) or str(hf / "snap")
    )

    path = fetch.download_model(MODEL, REVISION, say=said.append)

    assert path == hf / "snap"
    assert calls == [(MODEL, {"revision": REVISION, "cache_dir": str(hf / "hub")})]
    assert MODEL in said[0]
    assert str(hf) in said[0]
    assert "ready" in said[-1]


def test_unreachable_hub_without_a_cached_copy_gives_a_clear_message(hf, monkeypatch):
    monkeypatch.setattr(fetch, "hub_unreachable", lambda: "URLError: timed out")
    monkeypatch.setattr(fetch, "snapshot_download", lambda *a, **k: pytest.fail("must not download"))

    with pytest.raises(fetch.DownloadError, match="cannot reach Hugging Face.*URLError.*internet access") as info:
        fetch.download_model(MODEL, REVISION)

    assert str(hf) in str(info.value)


def test_unreachable_hub_uses_the_cached_copy(hf, monkeypatch):
    (hf / "hub" / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / REVISION).mkdir(parents=True)
    said = []
    monkeypatch.setattr(fetch, "hub_unreachable", lambda: "URLError: timed out")
    monkeypatch.setattr(fetch, "snapshot_download", lambda *a, **k: pytest.fail("must not download"))

    path = fetch.download_model(MODEL, REVISION, say=said.append)

    assert path == hf / "hub" / "models--Qwen--Qwen3.5-0.8B"
    assert "using the copy" in said[0]


def test_a_failed_download_is_wrapped_with_the_model_name(hf, monkeypatch):
    def broken(model, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fetch, "hub_unreachable", lambda: None)
    monkeypatch.setattr(fetch, "snapshot_download", broken)

    with pytest.raises(fetch.DownloadError, match=f"downloading {MODEL} failed.*disk full"):
        fetch.download_model(MODEL)


def test_cache_detection_follows_revision_or_main_ref(hf):
    base = hf / "hub" / "models--Qwen--Qwen3.5-0.8B"
    assert not fetch.is_cached(MODEL)
    (base / "refs").mkdir(parents=True)
    (base / "refs" / "main").write_text(REVISION)
    assert fetch.is_cached(MODEL)
    assert not fetch.is_cached(MODEL, REVISION)
    (base / "snapshots" / REVISION).mkdir(parents=True)
    assert fetch.is_cached(MODEL, REVISION)


def test_hub_check_treats_http_answers_as_reachable(monkeypatch):
    def answer(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", answer)
    assert fetch.hub_unreachable() is None


def test_hub_check_reports_network_errors(monkeypatch):
    def down(request, timeout):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", down)
    assert "URLError" in fetch.hub_unreachable()


def test_folder_size_counts_files(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 1024)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * 1024)

    assert fetch.folder_size_gb(Path(tmp_path)) == pytest.approx(2048 / fetch.GB)
