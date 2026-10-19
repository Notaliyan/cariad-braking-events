import math

import numpy as np
import pandas as pd

GRID_STEP_S = 0.1
EVENT_COLUMNS = ["start_s", "end_s", "duration_s", "speed_drop_kmh"]


def _make_grid(first_s: float, last_s: float) -> np.ndarray:
    """Regular 0.1 s time grid from the first step >= first_s to the last step <= last_s."""
    # Count in whole steps (integers) to avoid float errors like 0.1 * 3 = 0.30000000000000004.
    first_step = math.ceil(round(first_s / GRID_STEP_S, 6))
    last_step = math.floor(round(last_s / GRID_STEP_S, 6))
    steps = np.arange(first_step, last_step + 1)
    return np.round(steps * GRID_STEP_S, 1)


def resample_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return df on a regular 10 Hz grid (see exercise.md)."""
    raw = df.sort_values("timestamp_s").reset_index(drop=True)
    grid = _make_grid(raw["timestamp_s"].iloc[0], raw["timestamp_s"].iloc[-1])

    # Speed: linear interpolation over time, using only the real measurements.
    # This fills the sensor dropout (NaN) and the new grid points in one step.
    measured = raw.dropna(subset=["speed_kmh"])
    speed = np.interp(grid, measured["timestamp_s"], measured["speed_kmh"])

    # Brake: last known value at or before each grid time (no interpolation, stays 0/1).
    brake = pd.merge_asof(
        pd.DataFrame({"timestamp_s": grid}),
        raw[["timestamp_s", "brake_pressed"]],
        on="timestamp_s",
        direction="backward",
    )["brake_pressed"]

    return pd.DataFrame(
        {"speed_kmh": speed, "brake_pressed": brake.astype(int).to_numpy()},
        index=pd.Index(grid, name="timestamp_s"),
    )


def find_braking_events(df: pd.DataFrame, min_duration_s: float = 0.5) -> pd.DataFrame:
    """Return one row per braking event (see exercise.md)."""
    times = df.index.to_numpy()
    pressed = df["brake_pressed"].to_numpy() == 1
    speeds = df["speed_kmh"].to_numpy()

    # Find each run of pressed samples: (index of first pressed, index of first released).
    runs = []
    start = None
    for i, is_pressed in enumerate(pressed):
        if is_pressed and start is None:
            start = i
        elif not is_pressed and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:  # log ends while the brake is still pressed
        runs.append((start, len(times) - 1))

    events = []
    for s, e in runs:
        duration = round(times[e] - times[s], 6)
        if duration < min_duration_s:
            continue
        events.append(
            {
                "start_s": times[s],
                "end_s": times[e],
                "duration_s": duration,
                "speed_drop_kmh": speeds[s] - speeds[e],
            }
        )

    # Passing columns keeps them even when there are no events.
    return pd.DataFrame(events, columns=EVENT_COLUMNS)


if __name__ == "__main__":
    raw = pd.read_csv("drive.csv")
    print(find_braking_events(resample_signals(raw)))
