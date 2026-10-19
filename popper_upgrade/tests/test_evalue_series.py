import pytest
from popper.agent import e_value_kappa_calibrator

from popper_upgrade.app.handlers.runs import evalue_series


def test_evalue_series_uses_popper_calibrator():
    stats = [0.04, 0.2, 0.01]
    series = evalue_series(stats, 0.1)
    assert [point["test"] for point in series] == [1, 2, 3]
    for i, point in enumerate(series, start=1):
        assert point["e_value"] == pytest.approx(float(e_value_kappa_calibrator(stats[:i], 0.1)[1]))
        assert point["threshold"] == pytest.approx(10)


def test_empty_series():
    assert evalue_series([], 0.1) == []


def test_zero_p_value_is_finite():
    series = evalue_series([0.0], 0.1)
    assert series[0]["e_value"] > series[0]["threshold"]
    assert series[0]["e_value"] < float("inf")
