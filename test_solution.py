import numpy as np
import pandas as pd
import pytest

from solution import EVENT_COLUMNS, find_braking_events, resample_signals


def make_resampled(brake):
    """Small hand-made DataFrame in the format returned by resample_signals."""
    times = np.round(np.arange(len(brake)) * 0.1, 1)
    speeds = np.linspace(50, 40, len(brake))
    return pd.DataFrame({"speed_kmh": speeds, "brake_pressed": brake}, index=times)


def test_short_press_is_ignored():
    # pressed 0.2-0.4 (0.2 s, too short), pressed 0.5-1.1 (0.6 s, counts)
    df = make_resampled([0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0])

    events = find_braking_events(df, min_duration_s=0.5)

    assert len(events) == 1
    assert events.loc[0, "start_s"] == pytest.approx(0.5)
    assert events.loc[0, "end_s"] == pytest.approx(1.1)
    assert events.loc[0, "duration_s"] == pytest.approx(0.6)
    assert events.loc[0, "speed_drop_kmh"] == pytest.approx(df["speed_kmh"][0.5] - df["speed_kmh"][1.1])


def test_no_events_keeps_columns():
    df = make_resampled([0, 1, 0, 0])

    events = find_braking_events(df)

    assert events.empty
    assert list(events.columns) == EVENT_COLUMNS


def test_nan_gap_is_interpolated_linearly():
    # Example from exercise.md: 50 km/h -> missing -> 40 km/h gives 45 km/h
    raw = pd.DataFrame(
        {
            "timestamp_s": [1.0, 1.1, 1.2],
            "speed_kmh": [50.0, np.nan, 40.0],
            "brake_pressed": [0, 0, 0],
        }
    )

    result = resample_signals(raw)

    assert result.loc[1.1, "speed_kmh"] == pytest.approx(45.0)
    assert not result["speed_kmh"].isna().any()


def test_grid_bounds_and_brake_uses_last_known_value():
    raw = pd.DataFrame(
        {
            "timestamp_s": [0.05, 0.15, 0.27],
            "speed_kmh": [30.0, 30.0, 30.0],
            "brake_pressed": [0, 1, 0],
        }
    )

    result = resample_signals(raw)

    # first step at/after 0.05 is 0.1, last step at/before 0.27 is 0.2
    assert list(result.index) == [0.1, 0.2]
    # 0.1 -> last known is 0.05 (0); 0.2 -> last known is 0.15 (1)
    assert list(result["brake_pressed"]) == [0, 1]


def test_drive_csv_finds_two_events():
    events = find_braking_events(resample_signals(pd.read_csv("drive.csv")))

    # two real braking events; the single-sample press at ~12.1 s is ignored
    assert len(events) == 2
    assert list(events["start_s"]) == pytest.approx([5.1, 20.1])
