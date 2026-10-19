from popper_upgrade.tools.freeze_diff import freeze_diff, main


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


BASE = "torch==2.5.1+cu121\nnumpy==1.26.4\npopper_agent @ git+https://example.org/POPPER.git@abc\n"


def test_additions_only_is_clean(tmp_path):
    upgrade = BASE + "sentence-transformers==3.3.1\nPillow==10.0.0\n"

    added, changed, removed = freeze_diff(_write(tmp_path, "a", BASE), _write(tmp_path, "b", upgrade))

    assert added == ["pillow ==10.0.0", "sentence-transformers ==3.3.1"]
    assert (changed, removed) == ([], [])


def test_changed_and_removed_packages_fail(tmp_path):
    upgrade = "torch==2.6.0\nnumpy_extra==1\npopper-agent @ git+https://example.org/POPPER.git@abc\n"
    base_path = _write(tmp_path, "a", BASE)
    upgrade_path = _write(tmp_path, "b", upgrade)

    added, changed, removed = freeze_diff(base_path, upgrade_path)

    assert changed == ["torch: ==2.5.1+cu121 -> ==2.6.0"]
    assert removed == ["numpy"]
    assert main([str(base_path), str(upgrade_path)]) == 1


def test_main_exit_code_for_additions(tmp_path):
    assert main([str(_write(tmp_path, "a", BASE)), str(_write(tmp_path, "b", BASE + "extra==1\n"))]) == 0
