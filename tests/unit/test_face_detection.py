from pathlib import Path
import pytest
from cortaflow.domain.tracking import FaceBox, FaceObservation
from cortaflow.services.face_detection import (
    MediaPipeFaceDetector,
    _merge_observations,
    _wide_shot_tiles,
)


def test_missing_official_model_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MediaPipeFaceDetector(tmp_path / "face_detector.task")


def test_wide_shot_tiles_cover_landscape_frame_with_overlap() -> None:
    tiles = _wide_shot_tiles(1920, 1080)
    assert tiles[0] == (0, 150, 600, 600)
    assert tiles[-1] == (1320, 150, 600, 600)
    assert all(left + width <= 1920 and top + height <= 1080 for left, top, width, height in tiles)


def test_overlapping_tile_face_is_not_duplicated() -> None:
    original = FaceObservation(box=FaceBox(x=.65, y=.3, width=.08, height=.14))
    duplicate = FaceObservation(box=FaceBox(x=.655, y=.305, width=.08, height=.14))
    second_person = FaceObservation(box=FaceBox(x=.88, y=.4, width=.07, height=.13))

    merged = _merge_observations([original], [duplicate, second_person])

    assert merged == [original, second_person]
