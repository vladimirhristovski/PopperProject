import ast
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
