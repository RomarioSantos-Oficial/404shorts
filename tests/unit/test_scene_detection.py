import pytest

from cortaflow.domain.analysis import TimeRange
from cortaflow.services.scene_detection import parse_silencedetect_output


def test_parses_silence_intervals() -> None:
    output = "silence_start: 1.25\nsilence_end: 2.75 | silence_duration: 1.5"
    result = parse_silencedetect_output(output)
    assert [(item.start_ms, item.end_ms) for item in result] == [(1250, 2750)]


def test_parses_multiple_ordered_silence_intervals() -> None:
    output = "\n".join(
        (
            "silence_start: -0.02",
            "silence_end: 1.0 | silence_duration: 1.02",
            "silence_start: 3.25",
            "silence_end: 4.75 | silence_duration: 1.5",
        )
    )
    result = parse_silencedetect_output(output)
    assert [(item.start_ms, item.end_ms) for item in result] == [(0, 1000), (3250, 4750)]


def test_time_range_rejects_reverse_interval() -> None:
    with pytest.raises(ValueError, match="posterior"):
        TimeRange(start_ms=1000, end_ms=500)
