from pathlib import Path
from types import SimpleNamespace

from cortaflow.domain.project import ReframeKeyframe
from cortaflow.domain.tracking import CropFrame, FaceBox, FaceTrackPoint
from cortaflow.services import face_detection
from cortaflow.ui.pages import analysis_page
from cortaflow.ui.pages.analysis_page import AnalysisPage


def test_local_face_model_is_found_in_the_actual_app_data_audit_folder(
    monkeypatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "CortaFlowAI"
    model = data_dir / "Audit" / "face_landmarker.task"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"existing local model")
    monkeypatch.setattr(
        face_detection,
        "AppConfig",
        lambda: SimpleNamespace(data_dir=data_dir, cache_dir=tmp_path / "Cache"),
    )

    assert face_detection.find_local_face_landmarker() == model.resolve()


def test_existing_face_model_is_selected_automatically(qtbot, monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "face_landmarker.task"
    model.write_bytes(b"official model fixture")
    monkeypatch.setattr(analysis_page, "find_local_face_landmarker", lambda: model.resolve())

    page = AnalysisPage()
    qtbot.addWidget(page)

    assert Path(page.face_model_path.text()) == model.resolve()
    assert "criação automática detectará os rostos" in page.face_status.text()


def test_face_analysis_and_manual_selection(qtbot, monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "vídeo.mp4"
    model = tmp_path / "face_landmarker.task"
    source.write_bytes(b"fixture")
    model.write_bytes(b"official model fixture")
    tracks = [
        FaceTrackPoint(track_id=1, timestamp_ms=0, box=FaceBox(x=.1, y=.2, width=.2, height=.3)),
        FaceTrackPoint(track_id=2, timestamp_ms=0, box=FaceBox(x=.65, y=.2, width=.2, height=.3)),
    ]
    keyframes = [ReframeKeyframe(timestamp_ms=0, crop=CropFrame(x=0, y=0, width=608, height=1080))]

    def fake_analysis(path, model_path, scenes, selected, interval, progress, cancelled):
        assert path == source
        assert model_path == model
        progress({"position_ms": 500, "duration_ms": 1000, "faces": 2})
        return tracks, keyframes

    monkeypatch.setattr(analysis_page, "analyze_faces", fake_analysis)
    page = AnalysisPage()
    qtbot.addWidget(page)
    page.set_context(source, None, 1000, 1920, 1080)
    page.face_model_path.setText(str(model))
    with qtbot.waitSignal(page.face_analysis_finished, timeout=5_000):
        page.start_face_analysis()
    assert page.face_selector.count() == 3
    page.face_selector.setCurrentIndex(page.face_selector.findData(2))
    with qtbot.waitSignal(page.face_selection_changed, timeout=1_000) as emitted:
        page.apply_face_selection()
    assert emitted.args[0][0] == 2
    assert emitted.args[0][1][0].crop.x > 800
