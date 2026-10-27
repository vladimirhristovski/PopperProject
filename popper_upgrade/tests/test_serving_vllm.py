import ast
import os
import time
from copy import deepcopy
from pathlib import Path

import pytest

import popper_upgrade.serving.vllm as vllm_module

BASELINE = Path(__file__).resolve().parents[2] / "baseline_experiment" / "test3_local.py"

ATTRIBUTES = {
    "model": "LOCAL_MODEL",
    "port": "LOCAL_PORT",
    "host": "LOCAL_HOST",
    "proxy_port": "PROXY_PORT",
    "proxy_handler": "_ProxyHandler",
    "vllm_proc": "vllm_proc",
    "_proxy_server": "_proxy_server",
}
NAMES = {"host": "LOCAL_HOST", "port": "LOCAL_PORT"}
METHODS = {
    "start_proxy": "start_proxy",
    "stop_proxy": "stop_proxy",
    "start": "start_vllm",
    "stop": "stop_vllm",
}
HANDLER_METHODS = ("log_message", "do_GET", "do_POST")


class _Canonical(ast.NodeTransformer):
    def __init__(self, names, attributes):
        self._names = names
        self._attributes = attributes

    def visit_Global(self, node):
        return None

    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self._names.get(node.id, node.id), ctx=node.ctx), node)

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in self._attributes:
            return ast.copy_location(ast.Name(id=self._attributes[node.attr], ctx=node.ctx), node)
        return node


def _canonical(function, names, attributes, drop_self):
    function = _Canonical(names, attributes).visit(deepcopy(function))
    args = function.args.args[1:] if drop_self else function.args.args
    return ast.dump(ast.Module(body=function.body, type_ignores=[])), [arg.arg for arg in args]


def _functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _class(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def _methods(class_node):
    return {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}


@pytest.fixture(scope="module")
def baseline_tree():
    return ast.parse(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def copy_tree():
    return ast.parse(Path(vllm_module.__file__).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", HANDLER_METHODS)
def test_proxy_handler_methods_match_baseline(baseline_tree, copy_tree, name):
    baseline = _methods(_class(baseline_tree, "_ProxyHandler"))[name]
    copy = _methods(_class(copy_tree, "_ProxyHandler"))[name]

    assert _canonical(copy, NAMES, {}, False) == _canonical(baseline, {}, {}, False)


@pytest.mark.parametrize(("method", "function"), METHODS.items())
def test_server_methods_match_baseline(baseline_tree, copy_tree, method, function):
    baseline = _functions(baseline_tree)[function]
    copy = _methods(_class(copy_tree, "VllmServer"))[method]

    assert _canonical(copy, {}, ATTRIBUTES, True) == _canonical(baseline, {}, {}, False)


def test_default_proxy_port_follows_port():
    assert vllm_module.VllmServer("m", 8000).proxy_port == 8001
    assert vllm_module.VllmServer("m", 8000, proxy_port=9000).proxy_port == 9000


SERVING = {"weights_gb": 60.4, "kv_kb_per_token": 192}


def _model(**changes):
    return {"name": "m", "served_model_id": "org/m", **changes}


def test_plan_launch_without_settings_reproduces_the_baseline_rule():
    big = vllm_module.plan_launch(_model(), 80.0)
    small = vllm_module.plan_launch(_model(), 40.0)

    assert (big["max_model_len"], big["gpu_util"], big["quantization"], big["dtype"]) == (16384, 0.90, "awq", "float16")
    assert (small["max_model_len"], small["gpu_util"]) == (6144, 0.95)
    assert big["args"] == []


def test_plan_launch_none_omits_quantization_and_keeps_extra_args():
    plan = vllm_module.plan_launch(
        _model(quantization="none", dtype="bfloat16", serving={"vllm_args": ["--trust-remote-code"]}), 80.0
    )

    assert (plan["quantization"], plan["dtype"], plan["args"]) == (None, "bfloat16", ["--trust-remote-code"])


def test_plan_launch_shrinks_the_context_to_what_fits():
    plan = vllm_module.plan_launch(_model(serving=SERVING), 80.0)
    tight = vllm_module.plan_launch(_model(serving={"weights_gb": 67.0, "kv_kb_per_token": 192}), 80.0)

    assert plan["max_model_len"] == 16384
    assert 6144 <= tight["max_model_len"] < 16384
    assert tight["max_model_len"] % 1024 == 0


def test_plan_launch_refuses_when_nothing_fits_and_names_the_fallback():
    model = _model(serving={"weights_gb": 75.0, "kv_kb_per_token": 192, "fallback": "run the FP8 job"})

    with pytest.raises(vllm_module.LaunchError, match="80 GB.*run the FP8 job") as info:
        vllm_module.plan_launch(model, 80.0)

    assert "m needs about" in str(info.value)
    with pytest.raises(vllm_module.LaunchError, match="llama-anchor"):
        vllm_module.plan_launch(_model(serving=SERVING), 40.0)


def test_launch_overrides_are_detected():
    assert not vllm_module.has_launch_overrides(_model(serving={"weights_gb": 1, "kv_kb_per_token": 1}))
    assert vllm_module.has_launch_overrides(_model(dtype="bfloat16"))
    assert vllm_module.has_launch_overrides(_model(serving={"vllm_args": ["--x"]}))


def test_command_uses_the_image_python_and_omits_quantization_when_none(monkeypatch):
    monkeypatch.setenv("POPPER_VLLM_PYTHON", "/usr/bin/vllm-python")
    server = vllm_module.ModelVllmServer("org/m", 8000, launch=_model(quantization="none", dtype="bfloat16"))

    command = server.command(vllm_module.plan_launch(server.launch, 80.0))

    assert command[:3] == ["/usr/bin/vllm-python", "-m", "vllm.entrypoints.openai.api_server"]
    assert "--quantization" not in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--max-model-len") + 1] == "16384"
    assert command[-1] == "--enforce-eager"


def test_launch_server_does_not_start_when_memory_cannot_fit(monkeypatch):
    monkeypatch.setattr(vllm_module, "detect_vram_gb", lambda: 24.0)
    monkeypatch.setattr(vllm_module.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    server = vllm_module.ModelVllmServer("org/m", 8000, launch=_model(serving=SERVING))

    assert server.start() is False


def _wait_for_lines(path, count):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists() and len(path.read_text().split()) >= count:
            return [int(pid) for pid in path.read_text().split()]
        time.sleep(0.05)
    pytest.fail("the fake process tree did not start")


def _running(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except OSError:
        return False
    return fields[0] != "Z"


def _tree(tmp_path, script):
    server = vllm_module.ModelVllmServer("org/m", 8000, launch=_model(), log_path=tmp_path / "vllm.log")
    server.vllm_proc = server._spawn(["sh", "-c", script], os.environ.copy())
    return server


needs_proc = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")


@needs_proc
def test_stop_ends_the_whole_process_tree(tmp_path):
    marker = tmp_path / "pids"
    server = _tree(tmp_path, f"sleep 300 & echo $! >> {marker}; sleep 300 & echo $! >> {marker}; wait")
    children = _wait_for_lines(marker, 2)

    server.stop()

    assert server.vllm_proc.poll() is not None
    assert not any(_running(pid) for pid in children)


@needs_proc
def test_stop_kills_children_that_outlive_the_server_process(tmp_path):
    marker = tmp_path / "pids"
    server = _tree(tmp_path, f"sleep 300 & echo $! >> {marker}; exit 0")
    children = _wait_for_lines(marker, 1)
    server.vllm_proc.wait(timeout=10)
    assert _running(children[0])

    server.stop()

    assert not _running(children[0])


@needs_proc
def test_stop_escalates_to_kill_when_terminate_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm_module, "STOP_GRACE_S", 1)
    marker = tmp_path / "pids"
    server = _tree(tmp_path, f"trap '' TERM; sleep 300 & echo $! >> {marker}; wait")
    children = _wait_for_lines(marker, 1)
    started = time.monotonic()

    server.stop()

    assert time.monotonic() - started >= 1
    assert server.vllm_proc.poll() is not None
    assert not any(_running(pid) for pid in children)


def test_stop_without_a_process_does_nothing():
    vllm_module.ModelVllmServer("org/m", 8000, launch=_model()).stop()


def test_servers_start_in_their_own_session(tmp_path):
    server = vllm_module.ModelVllmServer("org/m", 8000, launch=_model(), log_path=tmp_path / "vllm.log")
    server.vllm_proc = server._spawn(["sleep", "300"], os.environ.copy())
    try:
        assert os.getpgid(server.vllm_proc.pid) == server.vllm_proc.pid != os.getpgid(0)
    finally:
        server.stop()
