from pathlib import Path

from cortaflow.domain.analysis import TimeRange
from cortaflow.domain.tracking import FaceBox, FaceObservation
from cortaflow.services.face_analysis import analyze_faces


class FakeDetector:
    def __init__(self, model_path: Path) -> None:
        self.closed = False

    def detect(self, rgb_frame, timestamp_ms: int):
        assert rgb_frame.shape[2] == 3
        x = .1 if timestamp_ms < 1000 else .65
        return [
            FaceObservation(
                box=FaceBox(x=x, y=.2, width=.2, height=.25),
                mouth_openness=.4,
            )
        ]

    def close(self) -> None:
        self.closed = True


class EmptyDetector(FakeDetector):
    def detect(self, rgb_frame, timestamp_ms: int):
        return []


def test_video_pipeline_tracks_faces_and_resets_at_scene(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    updates: list[dict] = []
    tracks, keyframes = analyze_faces(
        source,
        tmp_path / "fake.task",
        scenes=[TimeRange(start_ms=0, end_ms=1000), TimeRange(start_ms=1000, end_ms=2000)],
        sample_interval_ms=500,
        progress=updates.append,
        detector_factory=FakeDetector,
    )
    assert tracks
    assert len({point.track_id for point in tracks}) == 2
    assert all(point.mouth_openness == .4 for point in tracks)
    assert len(keyframes) == len(tracks)
    assert updates and updates[-1]["status"] == "faces"


def test_video_without_faces_returns_empty_tracking(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    tracks, keyframes = analyze_faces(
        source,
        tmp_path / "fake.task",
        sample_interval_ms=500,
        detector_factory=EmptyDetector,
    )

    assert tracks == []
    assert keyframes == []
