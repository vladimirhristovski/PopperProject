import pytest

from cluster import family, run_llama, run_nvidia, run_qwen


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(family, "run_job", lambda *args: seen.append(args) or 0)
    return seen


@pytest.mark.parametrize(
    ("module", "family_name"), [(run_llama, "llama"), (run_qwen, "qwen"), (run_nvidia, "nvidia")]
)
@pytest.mark.parametrize("size", ["small", "large"])
def test_family_and_size_map_to_the_job(calls, module, family_name, size):
    assert module.main(["--size", size]) == 0

    assert calls == [(f"{family_name}-{size}", ".", False, None, None, None)]


def test_llama_anchor_selects_the_proven_job(calls):
    assert run_llama.main(["--anchor", "--workdir", "out", "--skip-preflight"]) == 0

    assert calls == [("llama-anchor", "out", True, None, None, None)]


def test_job_options_are_passed_on(calls):
    run_qwen.main(["--size", "large", "--replicates", "2", "--examples-per-task", "3"])

    assert calls == [("qwen-large", ".", False, 2, 3, None)]


@pytest.mark.parametrize("module", [run_llama, run_qwen, run_nvidia])
def test_size_other_than_small_or_large_is_an_argument_error(calls, module, capsys):
    with pytest.raises(SystemExit) as info:
        module.main(["--size", "medium"])

    assert info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert calls == []


def test_size_is_required_unless_anchor(calls):
    with pytest.raises(SystemExit):
        run_qwen.main([])
    with pytest.raises(SystemExit):
        run_llama.main(["--anchor", "--size", "small"])
    assert calls == []


def test_max_hours_is_passed_on(calls):
    run_llama.main(["--size", "large", "--max-hours", "5.5"])

    assert calls == [("llama-large", ".", False, None, None, 5.5)]


def test_only_llama_has_an_anchor_option(calls):
    with pytest.raises(SystemExit):
        run_qwen.main(["--anchor"])
