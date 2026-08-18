import pytest
from pydantic import ValidationError
from cortaflow.domain.clip import ClipRange, format_timestamp


def test_formats_timestamp() -> None:
    assert format_timestamp(3_723_045, True) == "01:02:03.045"


def test_clip_requires_forward_range() -> None:
    with pytest.raises(ValidationError):
        ClipRange(start_ms=1000, end_ms=1000)

