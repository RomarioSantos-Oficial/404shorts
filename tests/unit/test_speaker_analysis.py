from pathlib import Path

from cortaflow.domain.tracking import AudioEvidence, FaceBox, FaceTrackPoint, SpeakerKeyframe, SpeakerOverride
from cortaflow.services import speaker_analysis
from cortaflow.services.speaker_analysis import analyze_active_speaker, apply_speaker_overrides


def face(track_id: int, timestamp: int, mouth: float, x: float) -> FaceTrackPoint:
    return FaceTrackPoint(
        track_id=track_id,
        timestamp_ms=timestamp,
        box=FaceBox(x=x, y=.2, width=.2, height=.3),
        mouth_openness=mouth,
    )


def test_pipeline_switches_focus_and_generates_crops(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "vídeo.mp4"
    source.write_bytes(b"fixture")
    timestamps = list(range(0, 3000, 300))
    tracks = []
    for timestamp in timestamps:
        tracks.extend(
            [
                face(1, timestamp, .9 if timestamp < 600 else .05, .1),
                face(2, timestamp, .05 if timestamp < 600 else 1.0, .65),
            ]
        )
    monkeypatch.setattr(
        speaker_analysis,
        "extract_audio_evidence",
        lambda *args, **kwargs: [
            AudioEvidence(timestamp_ms=value, energy=.9, voice_active=True)
            for value in timestamps
        ],
    )
    decisions, crops = analyze_active_speaker(source, tracks, 1920, 1080)
    assert decisions[0].track_id == 1
    assert decisions[-1].track_id == 2
    assert len(crops) == len(timestamps)
    assert crops[-1].crop.x > crops[0].crop.x


def test_latest_manual_override_always_wins() -> None:
    decisions = [
        SpeakerKeyframe(timestamp_ms=500, track_id=1, confidence=.8, uncertain=False),
        SpeakerKeyframe(timestamp_ms=1000, track_id=1, confidence=.8, uncertain=False),
    ]
    overrides = [
        SpeakerOverride(start_ms=0, end_ms=1500, track_id=2),
        SpeakerOverride(start_ms=900, end_ms=1200, track_id=3),
    ]
    corrected = apply_speaker_overrides(decisions, overrides)
    assert corrected[0].track_id == 2
    assert corrected[1].track_id == 3
    assert all(item.manual and item.confidence == 1 for item in corrected)
