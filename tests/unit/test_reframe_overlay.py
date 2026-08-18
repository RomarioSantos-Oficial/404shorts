from cortaflow.domain.project import ReframeKeyframe
from cortaflow.domain.tracking import CropFrame, FaceBox, FaceTrackPoint
from cortaflow.ui.widgets.reframe_overlay import ReframeOverlay


def test_overlay_renders_crop_and_selected_anonymous_face(qtbot) -> None:
    overlay = ReframeOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(640, 360)
    overlay.set_source_size(1920, 1080)
    overlay.set_data(
        [FaceTrackPoint(track_id=2, timestamp_ms=0, box=FaceBox(x=.6, y=.2, width=.2, height=.3))],
        [ReframeKeyframe(timestamp_ms=0, crop=CropFrame(x=900, y=0, width=608, height=1080))],
        2,
    )
    overlay.show()
    pixmap = overlay.grab()
    assert not pixmap.isNull()
    assert overlay._visible_faces()[0].track_id == 2
