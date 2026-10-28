import contextlib
import json
import os
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from cluster import canary, fetch, preflight
from cluster import job as job_module
from popper_upgrade.models import ServingError
from popper_upgrade.serving.rewrite import request_rewriter
from popper_upgrade.serving.vllm import ModelVllmServer, plan_launch

SOFTWARE = {"vllm": "0.19.1", "torch": "2.10.0", "triton": "3.6.0", "flashinfer-python": "0.6.0"}
LONG_BODY = {
    "choices": [{"message": {"content": "There are 260 records."}}],
    "usage": {"prompt_tokens": 3900, "completion_tokens": 9},
}
SHORT_BODY = {"choices": [{"message": {"content": "12 + 30 is 42."}}], "usage": {}}
KERNEL_LOG = (
    "INFO 10-27 starting\n"
    "(EngineCore pid=1) ERROR Traceback\n"
    "triton.runtime.errors.OutOfResources: ptxas fatal : Unresolved extern function in chunk_gated_delta_rule\n"
)


class Env:
    def __init__(self):
        self.rows = []
        self.posts = []
        self.bodies = [LONG_BODY, SHORT_BODY]
        self.serving_error = None
        self.log = ""
        self.post_error = None
        self.downloads = []
        self.download_error = None
        self.stall = 0


@pytest.fixture
def env(monkeypatch, tmp_path):
    state = Env()
    monkeypatch.delenv("POPPER_FAKE_BACKEND", raising=False)
    monkeypatch.delenv("POPPER_IMAGE", raising=False)
    monkeypatch.delenv(canary.GPU_UTIL_ENV, raising=False)
    monkeypatch.delenv(canary.MAX_LEN_ENV, raising=False)
    monkeypatch.delenv(canary.EXTRA_ARGS_ENV, raising=False)
    monkeypatch.setattr(preflight, "_driver_version", lambda: "570.10")
    monkeypatch.setattr(preflight, "_gpu_query", lambda: ("Test GPU", 24.0))
    monkeypatch.setattr(canary, "_smi", lambda query: "8.6")
    monkeypatch.setattr(canary, "cuda_driver_api", lambda: "12.8")
    monkeypatch.setattr(canary, "versions", lambda: dict(SOFTWARE))
    monkeypatch.setattr(canary, "_vllm_pids", lambda: set())

    def download(model_id, revision=None, say=print):
        state.downloads.append((model_id, revision))
        if state.download_error is not None:
            raise state.download_error
        path = tmp_path / "snapshot"
        path.mkdir(exist_ok=True)
        (path / "weights").write_bytes(b"x" * 10)
        return path

    monkeypatch.setattr(fetch, "download_model", download)

    @contextlib.contextmanager
    def serving(row, log_dir):
        state.rows.append(row)
        if state.log:
            (Path(log_dir) / "vllm.log").write_text(state.log)
        if state.serving_error:
            raise ServingError(state.serving_error)
        print("Starting vLLM: python -m vllm --model x")
        yield SimpleNamespace(port=8001)

    def post(port, payload):
        state.posts.append(payload)
        if state.stall:
            time.sleep(state.stall)
        if state.post_error and len(state.posts) == state.post_error[0]:
            raise state.post_error[1]
        return state.bodies[len(state.posts) - 1]

    monkeypatch.setattr(canary, "serving_for", serving)
    monkeypatch.setattr(canary, "_post", post)
    return state


def _run(tmp_path, capsys):
    code = canary.run_canary(tmp_path / "work")
    out = capsys.readouterr().out.splitlines()
    document = json.loads((tmp_path / "work" / "canary.json").read_text(encoding="utf-8"))
    return code, out, document


def _check(document, name):
    return next(check for check in document["checks"] if check["name"] == name)


def test_pass_prints_the_run_line_and_records_every_fact(env, tmp_path, capsys):
    code, out, document = _run(tmp_path, capsys)

    assert code == 0
    assert out[-1] == "PASS — run: singularity run --nv popper_qwen.sif"
    assert len(out) <= 10
    assert out[0].startswith("Machine: Test GPU, 24 GB, compute capability 8.6, driver 570.10, CUDA driver API 12.8")
    assert "vLLM 0.19.1, torch 2.10.0" in out[1]
    assert document["verdict"] == "PASS"
    assert document["facts"]["driver"] == "570.10"
    assert document["facts"]["cuda_driver_api"] == "12.8"
    assert document["facts"]["gpu"] == {"name": "Test GPU", "vram_gb": 24.0, "compute_capability": "8.6"}
    assert document["facts"]["software"] == SOFTWARE
    assert [check["name"] for check in document["checks"]] == list(canary.ORDER)
    assert all(check["ok"] for check in document["checks"])
    assert document["launch"]["command"] == "python -m vllm --model x"
    assert env.downloads == [(canary.MODEL_ID, canary.REVISION)]
    assert document["model"]["revision"] == canary.REVISION
    assert any("0.30" in line for line in document["launch"]["differences_from_qwen_large"])


def test_the_launch_row_is_qwen_large_with_only_the_listed_differences(env, tmp_path, capsys):
    _run(tmp_path, capsys)
    row = env.rows[0]
    qwen = preflight.model_row(job_module.load_job("qwen-large"))
    args = row["serving"]["vllm_args"]
    base = qwen["serving"]["vllm_args"]

    assert row["served_model_id"] == canary.MODEL_ID
    assert row["max_model_len"] == 8192
    assert args[: len(base)] == base
    assert args[len(base) :] == [
        "--revision",
        canary.REVISION,
        "--tokenizer-revision",
        canary.REVISION,
        "--max-num-batched-tokens",
        "2048",
    ]
    assert row["serving"]["gpu_util"] == 0.3
    assert args.count("--revision") == args.count("--tokenizer-revision") == 1
    assert row["revision"] is None
    assert "weights_gb" not in row["serving"]
    assert (row["dtype"], row["quantization"], row["backend"], row["port"]) == ("bfloat16", "none", "vllm", 8001)
    assert row["request_overrides"] == qwen["request_overrides"]
    assert row["strip_tools"] is True


@pytest.mark.parametrize(("util", "expected"), [(None, "0.30"), ("0.78", "0.78"), ("0.85", "0.85")])
def test_the_canary_command_carries_the_memory_flag_exactly_once(env, tmp_path, capsys, monkeypatch, util, expected):
    monkeypatch.setenv("POPPER_VLLM_PYTHON", "/opt/vllm-env/bin/python")
    monkeypatch.setenv(canary.EXTRA_ARGS_ENV, "--max-num-seqs 8")
    if util:
        monkeypatch.setenv(canary.GPU_UTIL_ENV, util)
    _run(tmp_path, capsys)
    row = env.rows[0]

    server = ModelVllmServer(row["served_model_id"], 8000, launch=row)
    command = server.command(plan_launch(row, 4.0))

    assert command.count("--gpu-memory-utilization") == 1
    assert command[command.index("--gpu-memory-utilization") + 1] == expected
    assert command.count("--max-num-batched-tokens") == 1
    assert command[-2:] == ["--max-num-seqs", "8"]


def test_environment_overrides_change_only_memory_and_context(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(canary.GPU_UTIL_ENV, "0.85")
    monkeypatch.setenv(canary.MAX_LEN_ENV, "4096")

    _, _, document = _run(tmp_path, capsys)

    assert env.rows[0]["serving"]["gpu_util"] == 0.85
    assert "--gpu-memory-utilization" not in env.rows[0]["serving"]["vllm_args"]
    assert env.rows[0]["max_model_len"] == 4096
    assert document["launch"]["effective"] == {
        "gpu_memory_utilization": "0.85",
        "max_model_len": 4096,
        "extra_args": [],
    }


def test_extra_arguments_from_the_environment_are_appended_and_listed(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(canary.EXTRA_ARGS_ENV, "--max-num-seqs 16")

    _, _, document = _run(tmp_path, capsys)

    assert env.rows[0]["serving"]["vllm_args"][-2:] == ["--max-num-seqs", "16"]
    assert document["launch"]["effective"]["extra_args"] == ["--max-num-seqs", "16"]
    assert "--max-num-seqs 16" in document["launch"]["differences_from_qwen_large"][0]


def test_the_requests_are_long_enough_and_the_proxy_applies_the_qwen_large_settings(env, tmp_path, capsys):
    _run(tmp_path, capsys)
    long_request, short_request = env.posts

    assert len(long_request["messages"][0]["content"]) > 10000
    rewritten = json.loads(request_rewriter(env.rows[0])("/v1/chat/completions", json.dumps(short_request).encode()))
    assert rewritten["chat_template_kwargs"] == {"enable_thinking": False}
    assert (rewritten["temperature"], rewritten["top_k"], rewritten["presence_penalty"]) == (0.7, 20, 1.5)
    assert rewritten["max_tokens"] == 64


def test_kernel_error_at_startup_fails_with_the_log_excerpt(env, tmp_path, capsys):
    env.serving_error = "vLLM did not start for qwen-canary"
    env.log = KERNEL_LOG

    code, out, document = _run(tmp_path, capsys)

    assert code == 1
    assert out[-1] == "FAIL — send canary.json to Vladimir; run popper_upgrade.sif llama-anchor instead"
    assert len(out) <= 10
    assert document["verdict"] == "FAIL"
    assert _check(document, "server_start")["ok"] is False
    assert document["failure"]["category"] == "kernel"
    assert "ptxas" in document["failure"]["vllm_log"]["last_error"]
    assert "chunk_gated_delta_rule" in document["failure"]["vllm_log"]["tail"]
    assert _check(document, "long_prompt")["skipped"] is True
    assert any("kernel" in line for line in out)
    assert not any(line.startswith("Long prompt") for line in out)
    assert env.posts == []


def test_kernel_error_during_the_long_prompt_fails_after_a_good_start(env, tmp_path, capsys):
    env.log = KERNEL_LOG
    env.post_error = (1, urllib.error.URLError("connection reset"))

    code, out, document = _run(tmp_path, capsys)

    assert code == 1
    assert _check(document, "server_start")["ok"] is True
    assert _check(document, "long_prompt")["ok"] is False
    assert document["failure"]["category"] == "kernel"
    assert document["failure"]["check"] == "long_prompt"
    assert _check(document, "short_chat")["skipped"] is True
    assert out[-1].startswith("FAIL")


def test_a_log_without_kernel_words_is_classified_as_other(env, tmp_path, capsys):
    env.serving_error = "vLLM did not start"
    env.log = "INFO a\nValueError: something unrelated broke\n"

    _, _, document = _run(tmp_path, capsys)

    assert document["failure"]["category"] == "other"


def test_out_of_memory_is_classified_and_explained(env, tmp_path, capsys):
    env.serving_error = "vLLM did not start"
    env.log = "ValueError: Free memory on device (1.2/4.0 GiB) is less than desired GPU memory utilization\n"

    _, out, document = _run(tmp_path, capsys)

    assert document["failure"]["category"] == "memory"
    assert any("out of GPU memory" in line for line in out)


def test_download_error_fails_before_the_gpu_is_exercised(env, tmp_path, capsys):
    env.download_error = fetch.DownloadError("cannot reach Hugging Face at https://huggingface.co (URLError)")

    code, out, document = _run(tmp_path, capsys)

    assert code == 1
    assert _check(document, "download")["ok"] is False
    assert document["failure"]["category"] == "download"
    assert env.rows == []
    assert any("download problem" in line for line in out)
    assert any("cannot reach Hugging Face" in line for line in out)
    assert out[-1].startswith("FAIL")
    assert _check(document, "server_start")["skipped"] is True


def test_unreadable_driver_fails_without_downloading_or_serving(env, tmp_path, capsys, monkeypatch):
    def unreadable():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(preflight, "_driver_version", unreadable)

    code, out, document = _run(tmp_path, capsys)

    assert code == 1
    assert _check(document, "driver")["ok"] is False
    assert "could not be read" in _check(document, "driver")["detail"]
    assert env.downloads == []
    assert env.rows == []
    assert out[0].startswith("Machine: FAILED")
    assert out[-1].startswith("FAIL")


def test_driver_below_the_image_floor_fails(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("POPPER_IMAGE", "qwen")
    monkeypatch.setattr(preflight, "_driver_version", lambda: "510.47")

    code, _, document = _run(tmp_path, capsys)

    assert code == 1
    assert "older than the 525.60" in _check(document, "driver")["detail"]
    assert env.rows == []


def test_missing_gpu_fails(env, tmp_path, capsys, monkeypatch):
    def none():
        raise OSError("no devices")

    monkeypatch.setattr(preflight, "_gpu_query", none)

    code, _, document = _run(tmp_path, capsys)

    assert code == 1
    assert _check(document, "gpu")["ok"] is False
    assert env.rows == []


def test_unreadable_software_versions_fail(env, tmp_path, capsys, monkeypatch):
    def broken():
        raise OSError("no python")

    monkeypatch.setattr(canary, "versions", broken)

    code, out, document = _run(tmp_path, capsys)

    assert code == 1
    assert _check(document, "software")["ok"] is False
    assert any(line.startswith("Software: FAILED") for line in out)


def test_thinking_text_in_the_short_reply_fails(env, tmp_path, capsys):
    env.bodies = [LONG_BODY, {"choices": [{"message": {"content": "<think>hmm</think> 42"}}], "usage": {}}]

    code, _, document = _run(tmp_path, capsys)

    assert code == 1
    assert _check(document, "short_chat")["ok"] is False
    assert document["failure"]["category"] == "thinking"


def test_reasoning_field_in_the_short_reply_fails(env, tmp_path, capsys):
    env.bodies = [LONG_BODY, {"choices": [{"message": {"content": "42", "reasoning_content": "step"}}]}]

    code, _, _ = _run(tmp_path, capsys)

    assert code == 1


def test_a_prompt_that_was_not_long_is_a_failure(env, tmp_path, capsys):
    short = {"prompt_tokens": 900, "completion_tokens": 3}
    env.bodies = [{"choices": [{"message": {"content": "ok"}}], "usage": short}]

    code, _, document = _run(tmp_path, capsys)

    assert code == 1
    assert "900 tokens" in _check(document, "long_prompt")["detail"]


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"), reason="needs SIGALRM")
def test_a_timeout_stops_the_server_and_reports_it(env, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(canary, "SERVE_TIMEOUT_S", 0.3)
    env.stall = 5

    code, _, document = _run(tmp_path, capsys)

    assert code == 1
    assert "timed out" in _check(document, "long_prompt")["detail"]
    assert _check(document, "clean_stop")["ok"] is True


def test_processes_left_behind_are_killed_and_reported_without_failing_the_verdict(env, tmp_path, capsys, monkeypatch):
    seen = iter([set(), {4242}])
    killed = []
    monkeypatch.setattr(canary, "_vllm_pids", lambda: next(seen))
    monkeypatch.setattr(canary, "_reap", killed.append)

    code, out, document = _run(tmp_path, capsys)

    assert code == 0
    assert killed == [{4242}]
    assert document["verdict"] == "PASS"
    assert _check(document, "clean_stop")["ok"] is False
    assert any(line.startswith("Clean shutdown: FAILED") for line in out)


def test_previous_canary_logs_are_cleared(env, tmp_path, capsys):
    log_dir = tmp_path / "work" / "canary"
    log_dir.mkdir(parents=True)
    (log_dir / "vllm.log").write_text("stale error: ptxas\n")

    _, _, document = _run(tmp_path, capsys)

    assert document["verdict"] == "PASS"
    assert not (log_dir / "vllm.log").exists()


def test_only_its_own_files_are_written(env, tmp_path, capsys):
    _run(tmp_path, capsys)

    assert sorted(path.name for path in (tmp_path / "work").iterdir()) == ["canary", "canary.json"]


def test_fake_backend_runs_the_whole_entrypoint_without_a_gpu(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("POPPER_FAKE_BACKEND", "1")

    code = job_module.main(["canary", "--workdir", str(tmp_path / "work")])

    out = capsys.readouterr().out.splitlines()
    document = json.loads((tmp_path / "work" / "canary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert out[-1] == canary.PASS_LINE
    assert out[0].startswith("Fake backend")
    assert document["fake_backend"] is True
    assert document["verdict"] == "PASS"
    assert _check(document, "short_chat")["ok"] is True


def test_main_dispatches_the_canary_command(monkeypatch):
    calls = []
    monkeypatch.setattr(job_module.canary_module, "run_canary", lambda workdir: calls.append(workdir) or 0)

    assert job_module.main(["canary"]) == 0
    assert job_module.main(["canary", "--workdir", "x"]) == 0
    assert calls == [".", "x"]


def test_process_scan_ignores_this_process():
    assert os.getpid() not in canary._vllm_pids()


def test_the_root_cause_is_found_above_the_wrapper_traceback(env, tmp_path, capsys):
    env.serving_error = "vLLM did not start for qwen-canary"
    env.log = (
        "(APIServer pid=71) INFO 10-27 starting Triton backend\n"
        "(EngineCore pid=156) ERROR 10-27 [core.py:1108] ValueError: Free memory on device cuda:0 (3.22/4.0 GiB)"
        " on startup is less than desired GPU memory utilization (0.85, 3.4 GiB).\n"
        + "(APIServer pid=71)   File x, line 1\n" * 40
        + "(APIServer pid=71) RuntimeError: Engine core initialization failed. See root cause above.\n"
    )

    code, out, document = _run(tmp_path, capsys)

    assert code == 1
    assert "Free memory on device" in document["failure"]["vllm_log"]["root_cause"]
    assert "Engine core initialization failed" in document["failure"]["vllm_log"]["last_error"]
    assert document["failure"]["category"] == "memory"
    assert any("Free memory on device" in line for line in out)
