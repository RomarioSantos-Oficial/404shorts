from pathlib import Path

import pytest

from cortaflow.services.media_probe import metadata_from_probe, parse_frame_rate


def test_parses_ffprobe_json() -> None:
    payload = {
        "format": {"duration": "12.5"},
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
            }
        ],
    }
    result = metadata_from_probe(Path("C:/Vídeos/meu vídeo.mp4"), payload)
    assert result.duration_seconds == 12.5
    assert (result.width, result.height) == (1920, 1080)
    assert result.fps == pytest.approx(29.97, rel=1e-3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("25/1", 25.0), ("60", 60.0), ("0/0", None), ("invalid", None), (None, None)],
)
def test_parse_frame_rate_handles_ffprobe_values(value, expected) -> None:
    assert parse_frame_rate(value) == expected
