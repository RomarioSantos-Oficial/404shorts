from cortaflow.domain.tracking import FaceBox, FaceTrackPoint
from cortaflow.services.active_speaker import EvidenceActiveSpeakerDetector


def face(track_id: int, mouth: float) -> FaceTrackPoint:
    return FaceTrackPoint(track_id=track_id, timestamp_ms=0, box=FaceBox(x=.1 * track_id, y=.1, width=.2, height=.2), mouth_openness=mouth)


def test_switch_requires_persistence_and_margin() -> None:
    detector = EvidenceActiveSpeakerDetector(persistence_ms=500, history_size=3)
    assert detector.update(0, [face(1, .9), face(2, .1)], True, .8).track_id == 1
    assert detector.update(200, [face(1, 0), face(2, 1)], True, .8).track_id == 1
    detector.update(500, [face(1, 0), face(2, 1)], True, .8)
    assert detector.update(1100, [face(1, 0), face(2, 1)], True, .8).track_id == 2


def test_uncertain_or_silent_keeps_current_focus() -> None:
    detector = EvidenceActiveSpeakerDetector()
    detector.update(0, [face(1, .8)], True, .7)
    decision = detector.update(100, [face(1, .2), face(2, .2)], False, 0)
    assert decision.track_id == 1
    assert decision.uncertain


def test_video_without_faces_returns_uncertain_none() -> None:
    decision = EvidenceActiveSpeakerDetector().update(0, [], True, .8)
    assert decision.track_id is None and decision.uncertain

