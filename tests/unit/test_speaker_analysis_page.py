from pathlib import Path

from cortaflow.domain.project import ReframeKeyframe
from cortaflow.domain.tracking import CropFrame, FaceBox, FaceTrackPoint, SpeakerKeyframe
from cortaflow.ui.pages import analysis_page
from cortaflow.ui.pages.analysis_page import AnalysisPage


def test_speaker_worker_and_manual_override(qtbot, monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "vídeo.mp4"
    source.write_bytes(b"fixture")
    tracks = [
        FaceTrackPoint(track_id=1, timestamp_ms=0, box=FaceBox(x=.1, y=.2, width=.2, height=.3)),
        FaceTrackPoint(track_id=2, timestamp_ms=0, box=FaceBox(x=.65, y=.2, width=.2, height=.3)),
    ]
    decisions = [SpeakerKeyframe(timestamp_ms=0, track_id=1, confidence=.8, uncertain=False)]
    crops = [ReframeKeyframe(timestamp_ms=0, crop=CropFrame(x=0, y=0, width=608, height=1080))]

    def fake_analysis(*args):
        progress, cancelled = args[-2:]
        progress({"status": "speaker", "track_id": 1})
        return decisions, crops

    monkeypatch.setattr(analysis_page, "analyze_active_speaker", fake_analysis)
    page = AnalysisPage()
    qtbot.addWidget(page)
    page.set_context(source, None, 1000, 1920, 1080)
    page.restore_faces(tracks, crops, None)
    with qtbot.waitSignal(page.speaker_analysis_finished, timeout=5_000):
        page.start_speaker_analysis()
    assert page.speaker_keyframes[0].track_id == 1

    page.override_start.setValue(0)
    page.override_end.setValue(1)
    page.override_face.setCurrentIndex(page.override_face.findData(2))
    with qtbot.waitSignal(page.speaker_overrides_changed, timeout=1_000) as emitted:
        page.add_speaker_override()
    overrides, corrected, reframe = emitted.args[0]
    assert overrides[0].track_id == 2
    assert corrected[0].track_id == 2 and corrected[0].manual
    assert reframe[0].manual
