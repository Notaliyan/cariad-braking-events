# Notes

**Assumptions**
- A braking event starts at the first grid sample with `brake_pressed == 1` and ends at the first sample where it is 0 again, so `duration_s` = number of pressed samples × 0.1 s. If the log ends while braking, the event ends at the last sample.
- `brake_pressed` on the grid is the last raw value at or before each grid time; a press shorter than one raw sample may therefore appear as 1–2 grid samples (the press at ~12.1 s), which `min_duration_s` filters out.
- Speed at the very start/end of the log is not extrapolated: if it were NaN there, the nearest known value is used (`np.interp` behaviour).

**With more time**
- Validate input (required columns, empty or unsorted data, a 0/1 check on `brake_pressed`), make the grid step a parameter, and flag events whose speed values come from interpolation rather than real measurements.
