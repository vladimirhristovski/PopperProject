from popper_upgrade.tools.check_env import env_diff, main

PINS = "numpy==1.26.4\ntorch==2.5.1\nhuggingface_hub[hf_xet]==0.30.0\n"
SHA = "59611864e09dd15d7528e240ad87c7ddbb4a701c"
FREEZE = (
    "numpy==1.26.4\ntorch==2.5.1+cpu\nhuggingface-hub==0.30.0\nextra==1\n"
    f"popper-agent @ git+https://github.com/snap-stanford/POPPER.git@{SHA}\n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_matching_environment_with_cpu_torch_tag_is_clean(tmp_path):
    assert env_diff(_write(tmp_path, "r", PINS), _write(tmp_path, "f", FREEZE), SHA) == []


def test_changed_missing_and_wrong_popper_commit_are_reported(tmp_path):
    freeze = "numpy==2.0.0\npopper-agent @ git+https://github.com/snap-stanford/POPPER.git@other\n"

    problems = env_diff(_write(tmp_path, "r", PINS), _write(tmp_path, "f", freeze), SHA)

    assert "numpy: pinned 1.26.4 but installed 2.0.0" in problems
    assert "torch: pinned 2.5.1 but not installed" in problems
    assert any(line.startswith("popper-agent: expected commit") for line in problems)


def test_main_exit_code(tmp_path):
    requirements = _write(tmp_path, "r", PINS)

    assert main([str(requirements), str(_write(tmp_path, "f", FREEZE)), "--popper-sha", SHA]) == 0
    assert main([str(requirements), str(_write(tmp_path, "g", "numpy==1.0\n"))]) == 1
