from cortaflow.domain.project import ReframeKeyframe, resolve_reframe_at
from cortaflow.domain.tracking import CropFrame


def test_manual_keyframe_overrides_automatic_at_same_time() -> None:
    auto = ReframeKeyframe(timestamp_ms=1000, crop=CropFrame(x=0, y=0, width=600, height=1000))
    manual = ReframeKeyframe(timestamp_ms=1000, crop=CropFrame(x=100, y=0, width=600, height=1000), manual=True)
    assert resolve_reframe_at([manual, auto], 1000) == manual.crop

