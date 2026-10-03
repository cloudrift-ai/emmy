"""The rule that decides which nightly timings reach ``tests/durations_cpu.json``."""

from tests.conftest import _MIN_RECORDED, _refresh_durations


def test_recorded_row_holds_within_half_its_value_even_under_the_floor():
    previous = {"slow": 6.0, "steady": 40.0}
    measured = {"slow": 3.5, "steady": 59.0}

    assert _refresh_durations(previous, measured) == {"slow": 6.0, "steady": 40.0}


def test_measurement_outside_the_band_replaces_or_drops_the_row():
    previous = {"faster": 6.0, "slower": 10.0}
    measured = {"faster": 2.9, "slower": 15.0}

    assert _refresh_durations(previous, measured) == {"slower": 15.0}


def test_new_rows_enter_at_the_floor_and_absent_tests_drop_out():
    previous = {"gone": 12.0}
    measured = {"new": _MIN_RECORDED, "noise": _MIN_RECORDED - 0.01, "rounded": 7.123}

    assert _refresh_durations(previous, measured) == {"new": _MIN_RECORDED, "rounded": 7.12}
