from PySide6.QtCore import QPoint, Qt

from cortaflow.ui.widgets.timeline import TimelineWidget
from cortaflow.services.editor_operations import create_initial_clips


def test_timeline_maps_clicks_and_updates_markers(qtbot) -> None:
    timeline = TimelineWidget()
    qtbot.addWidget(timeline)
    timeline.set_duration(10_000)
    timeline.set_markers(1_000, 8_000)

    assert timeline.canvas.x_to_position(timeline.canvas.position_to_x(4_000)) == 4_000
    assert "00:00:01" in timeline.markers.text()
    assert "00:00:08" in timeline.markers.text()

    x = round(timeline.canvas.position_to_x(5_000))
    with qtbot.waitSignal(timeline.seek_requested, timeout=1_000) as emitted:
        qtbot.mouseClick(timeline.canvas, Qt.MouseButton.LeftButton, pos=QPoint(x, 50))
    assert emitted.args == [5_000]


def test_timeline_zoom_changes_content_width(qtbot) -> None:
    timeline = TimelineWidget()
    qtbot.addWidget(timeline)
    timeline.set_duration(60_000)
    initial_width = timeline.canvas.minimumWidth()
    timeline.zoom.setValue(120)
    assert timeline.canvas.minimumWidth() > initial_width


def test_timeline_exposes_seven_tracks_and_selects_clips(qtbot) -> None:
    timeline = TimelineWidget()
    qtbot.addWidget(timeline)
    timeline.set_duration(10_000)
    clips = create_initial_clips(10_000)
    timeline.set_track_data(clips)
    assert len(timeline.canvas.track_items) == 7
    x = round(timeline.canvas.position_to_x(2_000))
    with qtbot.waitSignal(timeline.clip_selected, timeout=1_000) as emitted:
        qtbot.mousePress(timeline.canvas, Qt.MouseButton.LeftButton, pos=QPoint(x, 45))
    assert emitted.args[0] == next(item.clip_id for item in clips if item.track == "video")
