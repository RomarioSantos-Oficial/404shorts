from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage

from cortaflow.ui.widgets.watermark_overlay import WatermarkOverlay


def _valid_image(path: Path) -> Path:
    image = QImage(100, 50, QImage.Format.Format_ARGB32)
    image.fill(QColor("red"))
    assert image.save(str(path))
    return path


def test_valid_watermark_can_be_dragged_and_resized(qtbot, tmp_path: Path) -> None:
    overlay = WatermarkOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(500, 500)
    overlay.show()
    assert overlay.set_image(_valid_image(tmp_path / "logo.png"))
    overlay.set_placement(20, 20, 20, .75)
    before = overlay.watermark_rect()

    center = before.center().toPoint()
    with qtbot.waitSignal(overlay.placement_changed, timeout=1_000) as moved:
        qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=center)
        qtbot.mouseMove(overlay, center + QPoint(50, 30))
        qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=center + QPoint(50, 30))
    assert moved.args[0] > 20
    assert moved.args[1] > 20

    handle = (overlay.watermark_rect().bottomRight().toPoint() - QPoint(3, 3))
    with qtbot.waitSignal(overlay.placement_changed, timeout=1_000) as resized:
        qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=handle)
        qtbot.mouseMove(overlay, handle + QPoint(50, 0))
        qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=handle + QPoint(50, 0))
    assert resized.args[2] > 20


def test_invalid_watermark_is_rejected(qtbot, tmp_path: Path) -> None:
    overlay = WatermarkOverlay()
    qtbot.addWidget(overlay)
    invalid = tmp_path / "not-image.png"
    invalid.write_text("not an image", encoding="utf-8")

    assert not overlay.set_image(invalid)
    assert overlay.watermark_rect().isEmpty()


def test_vertical_watermark_stays_inside_the_visible_letterboxed_video(
    qtbot, tmp_path: Path
) -> None:
    overlay = WatermarkOverlay()
    qtbot.addWidget(overlay)
    overlay.resize(800, 400)
    overlay.set_content_aspect_ratio(540, 960)
    assert overlay.set_image(_valid_image(tmp_path / "logo-vertical.png"))
    overlay.set_placement(100, 100, 18, .75)

    content = overlay.content_rect()
    watermark = overlay.watermark_rect()

    assert content.width() == 225
    assert content.left() == 287.5
    assert watermark.right() <= content.right()
    assert watermark.bottom() <= content.bottom()
    assert watermark.left() > 287.5
