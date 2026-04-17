import numpy as np


def resolve_allowed_frequency_minutes(streaming_freq_mode, supported_frequency_minutes=None, frequency_minutes=None):
    if streaming_freq_mode == "dynamic":
        if not supported_frequency_minutes:
            raise ValueError("Dynamic streaming mode requires supported_frequency_minutes.")
        return [int(freq) for freq in supported_frequency_minutes]
    if frequency_minutes is None:
        raise ValueError("Fixed streaming mode requires frequency_minutes.")
    return [int(frequency_minutes)]


def constant_frequency_segments(timestamps, allowed_frequency_minutes):
    timestamps = np.asarray(timestamps)
    if len(timestamps) < 2:
        return []

    diffs = np.diff(timestamps).astype("timedelta64[m]").astype(np.int64)
    allowed = {int(freq) for freq in allowed_frequency_minutes}
    segments = []
    i = 0
    while i < len(diffs):
        diff_minutes = int(diffs[i])
        if diff_minutes not in allowed:
            i += 1
            continue

        run_start = i
        i += 1
        while i < len(diffs) and int(diffs[i]) == diff_minutes:
            i += 1
        segments.append((run_start, i + 1, diff_minutes))
    return segments


def build_streaming_plan(timestamps, x_len, y_len, allowed_frequency_minutes):
    plan = []
    min_window_len = x_len + y_len
    for segment_start, segment_end, freq_minutes in constant_frequency_segments(timestamps, allowed_frequency_minutes):
        if segment_end - segment_start < min_window_len:
            continue
        max_start = segment_end - min_window_len
        for start_idx in range(segment_start, max_start + 1, y_len):
            plan.append((start_idx, freq_minutes))
    return plan


def select_expansion_offsets(valid_stream_starts, num_events):
    if num_events == 0:
        return []

    if not valid_stream_starts:
        raise ValueError("The test split has no valid streaming prediction starts.")

    num_valid = len(valid_stream_starts)
    if num_valid < num_events:
        raise ValueError(
            f"Not enough valid streaming prediction starts for expansion scheduling. "
            f"num_valid_starts={num_valid}, num_events={num_events}."
        )

    offsets = []
    for event_idx in range(num_events):
        frac = (event_idx + 1) / (num_events + 1)
        start_pos = int(np.floor(frac * num_valid))
        start_pos = min(start_pos, num_valid - 1)
        offsets.append(int(valid_stream_starts[start_pos]))
    return offsets
