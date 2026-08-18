from cortaflow.domain.tracking import FaceBox, FaceTrackPoint
from cortaflow.services.rendered_framing import validate_rendered_face_tracks


def _point(timestamp: int, x: float, width: float = 0.22) -> FaceTrackPoint:
    return FaceTrackPoint(
        track_id=1,
        timestamp_ms=timestamp,
        box=FaceBox(x=x, y=0.18, width=width, height=0.3),
    )


def test_finished_preview_requires_complete_centered_face() -> None:
    validation = validate_rendered_face_tracks(
        [_point(0, 0.39), _point(400, 0.76, 0.24)],
        expected_face=True,
    )

    assert validation.status == "needs_review"
    assert validation.safe_samples == 1
    assert validation.unsafe_samples == 1
    assert "MP4 pronto" in validation.message


def test_finished_preview_fails_if_expected_face_disappears() -> None:
    validation = validate_rendered_face_tracks([], expected_face=True)

    assert validation.status == "needs_review"
    assert validation.score == 0


def test_finished_preview_without_expected_or_detected_face_uses_center_fallback() -> None:
    validation = validate_rendered_face_tracks([], expected_face=False)

    assert validation.status == "no_face"
    assert validation.score == 1
