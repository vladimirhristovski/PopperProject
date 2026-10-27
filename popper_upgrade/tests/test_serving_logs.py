import subprocess
import sys
from types import SimpleNamespace

import pytest

import popper_upgrade.models as models_module
import popper_upgrade.serving.vllm as vllm_module
from popper_upgrade.serving import logs
from popper_upgrade.serving.usage import UsageCounter
from popper_upgrade.tests.fakes import FakePopper
from popper_upgrade.tests.test_session import _session, _spec, fake_popper  # noqa: F401

CONTEXT_ERROR = (
    "ERROR 10-25 22:09:01 serving_chat.py:200] ValueError: This model's maximum context length is 6144 tokens. "
    "However, you requested 7000 tokens"
)
PROMPT_LINE = "INFO 10-25 22:08:32 logger.py:39] Received request: prompt: 'ValueError in the user text'"
STATUS_LINE = 'INFO:     127.0.0.1:40092 - "POST /v1/chat/completions HTTP/1.1" 400 Bad Request'
pytestmark = pytest.mark.usefixtures("fake_popper")

VLLM_MODEL = {
    "name": "m",
    "backend": "vllm",
    "served_model_id": "org/m",
    "port": 8001,
    "serving": {"weights_gb": 1.0, "kv_kb_per_token": 1},
}


def test_last_error_line_prefers_the_message_over_status_and_prompt_lines():
    text = "\n".join([CONTEXT_ERROR, STATUS_LINE, PROMPT_LINE])

    assert logs.last_error_line(text) == CONTEXT_ERROR


def test_last_error_line_falls_back_to_the_status_line():
    assert logs.last_error_line(f"{PROMPT_LINE}\n{STATUS_LINE}\n") == STATUS_LINE
    assert logs.last_error_line(PROMPT_LINE) is None


def test_last_error_line_is_capped():
    assert len(logs.last_error_line("ERROR " + "x" * 5000)) == logs.MAX_DETAIL


def test_upstream_errors_only_look_after_the_mark(tmp_path):
    path = tmp_path / "vllm.log"
    path.write_text(CONTEXT_ERROR + "\n", encoding="utf-8")
    errors = logs.UpstreamErrors(path)

    assert errors.last() == CONTEXT_ERROR
    errors.mark()
    assert errors.last() is None
    with path.open("a", encoding="utf-8") as handle:
        handle.write(STATUS_LINE + "\n")
    assert errors.last() == STATUS_LINE


def test_upstream_errors_tolerate_a_missing_log(tmp_path):
    errors = logs.UpstreamErrors(tmp_path / "absent.log")
    errors.mark()

    assert errors.last() is None


def test_copy_tail_keeps_only_the_last_bytes(tmp_path):
    source = tmp_path / "vllm.log"
    source.write_bytes(b"a" * 100 + b"b" * 10)

    assert logs.copy_tail(source, tmp_path / "out.log", limit=10)
    assert (tmp_path / "out.log").read_bytes() == b"b" * 10
    assert not logs.copy_tail(tmp_path / "absent.log", tmp_path / "none.log")
    assert not (tmp_path / "none.log").exists()


def test_capture_proxy_log_keeps_proxy_lines_and_still_prints(tmp_path, capsys):
    path = tmp_path / "logs" / "proxy.log"

    with logs.capture_proxy_log(path):
        print("  [proxy] POST /v1/chat/completions failed: HTTP Error 400: Bad Request")
        print("unrelated output")
        sys.stdout.write("  [proxy] partial")
        sys.stdout.write(" line\n")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == [
        "  [proxy] POST /v1/chat/completions failed: HTTP Error 400: Bad Request",
        "  [proxy] partial line",
    ]
    assert "unrelated output" in capsys.readouterr().out
    assert sys.stdout is not None


def test_server_failure_detection():
    assert logs.is_server_failure("InternalServerError('Error code: 502')")
    assert logs.is_server_failure("HTTPError 503")
    assert not logs.is_server_failure("RuntimeError('model down')")
    assert not logs.is_server_failure("BadRequestError('Error code: 400')")


class _Proc:
    def poll(self):
        return 1


def test_model_server_writes_vllm_output_to_the_log(monkeypatch, tmp_path):
    calls = []

    def popen(cmd, **kwargs):
        calls.append(kwargs)
        return _Proc()

    monkeypatch.setattr(vllm_module.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_module, "detect_vram_gb", lambda: 40.0)
    log_path = tmp_path / "vllm.log"
    server = vllm_module.ModelVllmServer("org/m", 8000, launch={"name": "m"}, log_path=log_path)

    assert server.start() is False

    assert calls[0]["stdout"].name == str(log_path)
    assert calls[0]["stderr"] == subprocess.STDOUT
    assert log_path.exists()


def test_model_server_without_a_log_path_inherits_output(monkeypatch):
    calls = []
    monkeypatch.setattr(vllm_module.subprocess, "Popen", lambda cmd, **kwargs: calls.append(kwargs) or _Proc())
    monkeypatch.setattr(vllm_module, "detect_vram_gb", lambda: 40.0)

    vllm_module.ModelVllmServer("org/m", 8000, launch={"name": "m"}).start()

    assert "stdout" not in calls[0]


class _RecordingServer:
    instances = []

    def __init__(self, model, port, host="localhost", proxy_port=None, launch=None, log_path=None):
        self.args = (model, port, proxy_port)
        self.launch = launch
        self.log_path = log_path
        type(self).instances.append(self)

    def start(self):
        return True

    def start_proxy(self):
        pass

    def stop_proxy(self):
        pass

    def stop(self):
        pass


class _IdleProxy:
    instances = []

    def __init__(self, listen_port, upstream_port, counter, **options):
        self.args = (listen_port, upstream_port)
        type(self).instances.append(self)

    def start(self):
        print("  [proxy] GET /v1/models failed: connection refused")

    def stop(self):
        pass


@pytest.fixture
def recording(monkeypatch):
    _RecordingServer.instances = []
    monkeypatch.setattr(models_module, "ModelVllmServer", _RecordingServer)
    monkeypatch.setattr(models_module, "UsageProxy", _IdleProxy)
    return _RecordingServer


def test_serving_with_a_log_dir_launches_the_stock_plan_and_wires_the_logs(recording, tmp_path):
    with models_module.serving_for(VLLM_MODEL, tmp_path) as counter:
        server = recording.instances[0]
        assert server.launch == {"name": "m"}
        assert server.log_path == tmp_path / logs.VLLM_LOG
        (tmp_path / logs.VLLM_LOG).write_text(CONTEXT_ERROR + "\n", encoding="utf-8")
        assert counter.upstream_error() == CONTEXT_ERROR

    proxy_lines = (tmp_path / logs.PROXY_LOG).read_text(encoding="utf-8").splitlines()
    assert proxy_lines[0].endswith("[proxy] GET /v1/models failed: connection refused")


def test_serving_with_a_log_dir_keeps_launch_overrides(recording, tmp_path):
    model = {**VLLM_MODEL, "dtype": "bfloat16"}

    with models_module.serving_for(model, tmp_path):
        assert recording.instances[0].launch == model


def test_serving_without_a_log_dir_is_unchanged(monkeypatch):
    seen = []

    class Stock:
        def __init__(self, model, port, host="localhost", proxy_port=None):
            seen.append("stock")

        def start(self):
            return True

        def start_proxy(self):
            pass

        def stop_proxy(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(models_module, "VllmServer", Stock)
    monkeypatch.setattr(models_module, "UsageProxy", _IdleProxy)
    stdout = sys.stdout

    with models_module.serving_for(VLLM_MODEL) as counter:
        assert sys.stdout is stdout
        assert counter.upstream_error() is None

    assert seen == ["stock"]


def _marked_counter(tmp_path):
    path = tmp_path / logs.VLLM_LOG
    path.write_text("old " + CONTEXT_ERROR + "\n", encoding="utf-8")
    counter = UsageCounter()
    counter.error_log = logs.UpstreamErrors(path)
    return counter, path


def _append(path, *lines):
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def test_a_502_run_error_gets_the_upstream_message_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("Error code: 502"))
    counter, path = _marked_counter(tmp_path)
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)
    original = counter.reset

    def reset_then_log():
        original()
        _append(path, CONTEXT_ERROR, STATUS_LINE)

    counter.reset = reset_then_log

    session.run()

    assert sink.result["status"] == "error"
    assert sink.result["error"] == f"RuntimeError('Error code: 502') | vLLM log: {CONTEXT_ERROR}"


def test_an_old_log_line_is_not_blamed_on_a_later_run(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("Error code: 502"))
    counter, _ = _marked_counter(tmp_path)
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)

    session.run()

    assert sink.result["error"] == "RuntimeError('Error code: 502')"


def test_other_run_errors_are_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("model down"))
    counter, path = _marked_counter(tmp_path)
    _append(path, CONTEXT_ERROR)
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=counter)

    session.run()

    assert sink.result["error"] == "RuntimeError('model down')"


def test_a_502_without_a_log_keeps_the_plain_error(tmp_path, monkeypatch):
    monkeypatch.setattr(FakePopper, "failure", RuntimeError("Error code: 502"))
    session, sink = _session(tmp_path, _spec(skills_enabled=False), [], usage=UsageCounter())

    session.run()

    assert sink.result["error"] == "RuntimeError('Error code: 502')"


def _launch_argv(monkeypatch, make_server, smi):
    started = []

    def popen(cmd, **kwargs):
        started.append((cmd, kwargs["env"]))
        return _Proc()

    def run(*args, **kwargs):
        if isinstance(smi, Exception):
            raise smi
        return SimpleNamespace(stdout=smi)

    monkeypatch.setattr(vllm_module.subprocess, "Popen", popen)
    monkeypatch.setattr(vllm_module.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "dropped")
    monkeypatch.delenv("POPPER_VLLM_PYTHON", raising=False)
    assert make_server().start() is False
    return started[0]


@pytest.mark.parametrize("smi", ["81920\n", "40960\n", "24576\n", OSError("no nvidia-smi")])
def test_logged_launch_matches_the_stock_server_argv_and_env(monkeypatch, tmp_path, smi):
    stock = _launch_argv(monkeypatch, lambda: vllm_module.VllmServer("org/m", 7999, proxy_port=8001), smi)
    logged = _launch_argv(
        monkeypatch,
        lambda: vllm_module.ModelVllmServer(
            "org/m", 7999, proxy_port=8001, launch={"name": "m"}, log_path=tmp_path / "vllm.log"
        ),
        smi,
    )

    assert logged == stock
    argv = stock[0]
    big = isinstance(smi, str) and int(smi) >= 70 * 1024
    assert argv[argv.index("--max-model-len") + 1] == ("16384" if big else "6144")
    assert argv[argv.index("--gpu-memory-utilization") + 1] == ("0.90" if big else "0.95")
    assert "PYTHONPATH" not in stock[1]


def test_logged_serving_builds_the_same_server_and_proxy_chain_as_stock(recording, tmp_path):
    with models_module.serving_for(VLLM_MODEL, tmp_path) as counter:
        assert counter.port == 8001

    assert recording.instances[0].args == ("org/m", 8000, 8002)
    assert _IdleProxy.instances[-1].args == (8001, 8002)


def test_trim_keeps_the_tail_on_a_line_boundary(tmp_path):
    path = tmp_path / "vllm.log"
    lines = [f"line {i:04d}\n" for i in range(1000)]
    path.write_text("".join(lines), encoding="utf-8")
    trimmer = logs.LogTrimmer([path], limit=5000, keep=2000)

    trimmer.trim_once()

    kept = path.read_text(encoding="utf-8")
    assert kept and len(kept) <= 2000
    assert "".join(lines).endswith(kept)
    assert kept.startswith("line ")
    assert trimmer.removed(path) == sum(map(len, lines)) - len(kept)


def test_trim_leaves_small_logs_alone_and_appends_continue_at_the_end(tmp_path):
    path = tmp_path / "vllm.log"
    path.write_text("a\n" * 10, encoding="utf-8")
    trimmer = logs.LogTrimmer([path], limit=1000, keep=500)
    trimmer.trim_once()
    assert path.read_text(encoding="utf-8") == "a\n" * 10

    with path.open("a", encoding="utf-8") as handle:
        handle.write("b\n" * 400)
        handle.flush()
        trimmer.trim_once()
        handle.write("tail\n")
    assert path.read_text(encoding="utf-8").endswith("b\ntail\n")
    assert path.stat().st_size <= 1000


def test_trim_ignores_missing_logs(tmp_path):
    logs.LogTrimmer([tmp_path / "absent.log"], limit=1, keep=1).trim_once()


def test_upstream_error_lookup_survives_a_trim(tmp_path):
    path = tmp_path / "vllm.log"
    trimmer = logs.LogTrimmer([path], limit=3000, keep=1500)
    errors = logs.UpstreamErrors(path, trimmer)
    path.write_text("ERROR old failure\n" + "INFO filler line\n" * 300, encoding="utf-8")
    errors.mark()
    with path.open("a", encoding="utf-8") as handle:
        handle.write("INFO filler line\n" * 100 + CONTEXT_ERROR + "\n" + "INFO filler line\n" * 50)

    trimmer.trim_once()

    assert trimmer.removed(path) > 0
    assert errors.last() == CONTEXT_ERROR
    errors.mark()
    assert errors.last() is None


def test_trim_started_with_serving_bounds_the_logs(recording, tmp_path, monkeypatch):
    (tmp_path / logs.VLLM_LOG).write_text("INFO filler line\n" * 100, encoding="utf-8")
    monkeypatch.setattr(models_module, "LogTrimmer", lambda paths: logs.LogTrimmer(paths, limit=100, keep=50))

    with models_module.serving_for(VLLM_MODEL, tmp_path):
        assert (tmp_path / logs.VLLM_LOG).stat().st_size <= 100
