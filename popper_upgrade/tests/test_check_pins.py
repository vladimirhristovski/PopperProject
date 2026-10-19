from popper_upgrade.tools.check_pins import check_pins, main


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_reports_mismatched_shared_pins(tmp_path):
    a = _write(tmp_path, "a.txt", "torch==2.5.1\nnumpy==1.26.4\nonly_a==1.0\nloose\n")
    b = _write(tmp_path, "b.txt", "torch==2.4.0\nnumpy==1.26.4\nonly_b==2.0\nloose==1.0\n")

    assert check_pins(a, b) == ["torch: 2.5.1 != 2.4.0"]


def test_name_matching_ignores_case_separators_and_extras(tmp_path):
    a = _write(tmp_path, "a.txt", "langchain_core==0.3.22\nhuggingface_hub[hf_xet]==0.30.0\n")
    b = _write(tmp_path, "b.txt", "LangChain-Core==0.3.0\nhuggingface-hub==0.30.0\n")

    assert check_pins(a, b) == ["langchain-core: 0.3.22 != 0.3.0"]


def test_main_exit_codes(tmp_path):
    a = _write(tmp_path, "a.txt", "torch==2.5.1\n")
    same = _write(tmp_path, "same.txt", "torch==2.5.1\n")
    other = _write(tmp_path, "other.txt", "torch==2.4.0\n")

    assert main([str(a), str(same)]) == 0
    assert main([str(a), str(other)]) == 1
