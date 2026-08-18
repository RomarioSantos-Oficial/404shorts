from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.editing import ReframeSettings
from cortaflow.domain.project import ReframeKeyframe
from cortaflow.domain.tracking import CropFrame, FaceBox, FaceTrackPoint, SpeakerKeyframe
from cortaflow.services.reframe_validation import ensure_safe_reframe, validate_suggestion_reframe


def _suggestion() -> ClipSuggestion:
    return ClipSuggestion(
        start_ms=0,
        end_ms=2_000,
        title="Troca de falante",
        transcript_excerpt="Uma ideia.",
        quality_score=.8,
        reason="Completo.",
    )


def test_reframe_is_repaired_and_validated_at_each_face_switch() -> None:
    tracks = [
        FaceTrackPoint(
            track_id=1,
            timestamp_ms=0,
            box=FaceBox(x=.1, y=.2, width=.14, height=.28),
        ),
        FaceTrackPoint(
            track_id=2,
            timestamp_ms=1_000,
            box=FaceBox(x=.72, y=.2, width=.14, height=.28),
        ),
    ]
    speakers = [
        SpeakerKeyframe(timestamp_ms=0, track_id=1, confidence=.9, uncertain=False),
        SpeakerKeyframe(timestamp_ms=1_000, track_id=2, confidence=.9, uncertain=False),
    ]
    lagging = [
        ReframeKeyframe(timestamp_ms=0, crop=CropFrame(x=0, y=0, width=608, height=1080)),
        ReframeKeyframe(timestamp_ms=1_000, crop=CropFrame(x=0, y=0, width=608, height=1080)),
    ]

    assert validate_suggestion_reframe(
        _suggestion(), tracks, speakers, lagging, ReframeSettings(), 1920, 1080
    ).status == "needs_review"

    repaired = ensure_safe_reframe(tracks, speakers, lagging, 1920, 1080)
    validation = validate_suggestion_reframe(
        _suggestion(), tracks, speakers, repaired, ReframeSettings(), 1920, 1080
    )

    assert repaired[1].crop.x > 900
    assert validation.status == "validated"
    assert validation.score == 1
    assert validation.speaker_changes == 1
    assert validation.unsafe_samples == 0


def test_clip_without_faces_uses_explainable_central_fallback() -> None:
    validation = validate_suggestion_reframe(
        _suggestion(), [], [], [], ReframeSettings(), 1920, 1080
    )

    assert validation.status == "no_face"
    assert validation.score == 1
    assert "central" in validation.message
