import fnmatch
import json
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
    assert calls == [
        (MODEL, {"revision": REVISION, "cache_dir": str(hf / "hub"), "allow_patterns": fetch.SERVING_PATTERNS})
    ]
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
    assert not fetch.is_cached(MODEL, "def456")
    (base / "refs" / "main").unlink()
    assert fetch.is_cached(MODEL, REVISION)
    assert not fetch.is_cached(MODEL)


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


QWEN_27B_FILES = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "crc32.txt",
    "generation_config.json",
    "merges.txt",
    *[f"model-{i:05d}-of-00018.safetensors" for i in range(1, 19)],
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def _selected(names):
    return [n for n in names if any(fnmatch.fnmatch(n, pattern) for pattern in fetch.SERVING_PATTERNS)]


def test_download_patterns_keep_everything_vllm_loads_and_drop_the_rest():
    selected = _selected(QWEN_27B_FILES)

    assert set(QWEN_27B_FILES) - set(selected) == {".gitattributes", "LICENSE", "README.md"}
    assert _selected(["tokenizer.model", "tiktoken.tiktoken", "configuration_x.py", "special_tokens_map.json"]) == [
        "tokenizer.model",
        "tiktoken.tiktoken",
        "configuration_x.py",
        "special_tokens_map.json",
    ]
    assert _selected(["pytorch_model.bin", "original/consolidated.00.pth", "model.gguf"]) == []


def _cache(hf, files, index=True, revision=REVISION, ref=None):
    base = hf / "hub" / "models--Qwen--Qwen3.5-0.8B"
    snapshot = base / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    for name in files:
        (snapshot / name).write_bytes(b"w")
    if index:
        weight_map = {f"layer{i}": name for i, name in enumerate(["a.safetensors", "b.safetensors"])}
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    if ref:
        (base / "refs").mkdir(parents=True)
        (base / "refs" / "main").write_text(ref)
    return snapshot


def test_weights_cached_needs_every_shard_of_the_index(hf):
    snapshot = _cache(hf, ["a.safetensors"])

    assert not fetch.weights_cached(MODEL, REVISION)
    (snapshot / "b.safetensors").write_bytes(b"w")
    assert fetch.weights_cached(MODEL, REVISION)


def test_weights_cached_rejects_a_dangling_shard_link(hf):
    snapshot = _cache(hf, ["a.safetensors"])
    (snapshot / "b.safetensors").symlink_to(hf / "hub" / "blobs" / "missing")

    assert not fetch.weights_cached(MODEL, REVISION)


def test_weights_cached_without_an_index_needs_the_single_weight_file(hf):
    snapshot = _cache(hf, ["a.safetensors"], index=False)

    assert not fetch.weights_cached(MODEL, REVISION)
    (snapshot / "model.safetensors").write_bytes(b"w")
    assert fetch.weights_cached(MODEL, REVISION)


def test_weights_cached_needs_the_model_config(hf):
    snapshot = _cache(hf, ["a.safetensors", "b.safetensors"])
    (snapshot / "config.json").unlink()

    assert not fetch.weights_cached(MODEL, REVISION)


def test_weights_cached_rejects_an_unreadable_index(hf):
    snapshot = _cache(hf, ["a.safetensors", "b.safetensors"])
    (snapshot / "model.safetensors.index.json").write_text("{broken")

    assert not fetch.weights_cached(MODEL, REVISION)


def test_weights_cached_ignores_other_revisions(hf):
    _cache(hf, ["a.safetensors", "b.safetensors"])

    assert not fetch.weights_cached(MODEL, "def456")
    assert not fetch.weights_cached("Qwen/other", REVISION)


def test_weights_cached_follows_the_main_ref_without_a_revision(hf):
    _cache(hf, ["a.safetensors", "b.safetensors"], ref=REVISION)

    assert fetch.weights_cached(MODEL)


def test_weights_cached_without_a_revision_or_ref_is_false(hf):
    _cache(hf, ["a.safetensors", "b.safetensors"])

    assert not fetch.weights_cached(MODEL)
