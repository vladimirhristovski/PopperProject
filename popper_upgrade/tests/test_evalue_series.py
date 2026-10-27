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


def test_chart_frame_keeps_a_single_test_and_spans_the_threshold():
    from popper_upgrade.app.tabs.run_detail import THRESHOLD_STEPS, chart_frame

    frame = chart_frame(evalue_series([0.03], 0.1), 0.1)

    evalues = frame[frame["series"] == "e-value"]
    threshold = frame[frame["series"] == "threshold"]
    assert evalues["test"].tolist() == [1]
    assert len(threshold) == THRESHOLD_STEPS + 1
    assert threshold["value"].tolist() == [pytest.approx(10)] * len(threshold)
    assert threshold["test"].min() < 1 < threshold["test"].max()


def test_chart_frame_without_tests_still_shows_the_threshold():
    from popper_upgrade.app.tabs.run_detail import chart_frame

    frame = chart_frame([], 0.05)

    assert set(frame["series"]) == {"threshold"}
    assert frame["value"].tolist() == [pytest.approx(20)] * len(frame)
