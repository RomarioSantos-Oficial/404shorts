"""Interactive watermark placement over a video preview."""

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


class WatermarkOverlay(QWidget):
    """Draw, drag and resize a valid image using normalized output coordinates."""

    placement_changed = Signal(float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._pixmap = QPixmap()
        self._x_percent = 50.0
        self._y_percent = 50.0
        self._width_percent = 18.0
        self._opacity = 0.75
        self._content_aspect_ratio: float | None = None
        self._drag_mode: str | None = None
        self._press_position = QPointF()
        self._press_values = (0.0, 0.0, 0.0)

    def set_image(self, path: Path | str | None) -> bool:
        pixmap = QPixmap(str(path)) if path else QPixmap()
        self._pixmap = pixmap if not pixmap.isNull() else QPixmap()
        self.setVisible(not self._pixmap.isNull())
        self.update()
        return not self._pixmap.isNull()

    def set_placement(
        self,
        x_percent: float,
        y_percent: float,
        width_percent: float,
        opacity: float,
    ) -> None:
        self._x_percent = min(100.0, max(0.0, float(x_percent)))
        self._y_percent = min(100.0, max(0.0, float(y_percent)))
        self._width_percent = min(80.0, max(2.0, float(width_percent)))
        self._opacity = min(1.0, max(0.05, float(opacity)))
        self.update()

    def set_content_aspect_ratio(self, width: int, height: int) -> None:
        """Match placement to the visible video area inside a letterboxed widget."""
        self._content_aspect_ratio = width / height if min(width, height) > 0 else None
        self.update()

    def content_rect(self) -> QRectF:
        """Return the video rectangle, excluding the player's black side/top bars."""
        bounds = QRectF(self.rect())
        ratio = self._content_aspect_ratio
        if not ratio or bounds.width() <= 0 or bounds.height() <= 0:
            return bounds
        widget_ratio = bounds.width() / bounds.height()
        if widget_ratio > ratio:
            width = bounds.height() * ratio
            return QRectF((bounds.width() - width) / 2, 0, width, bounds.height())
        height = bounds.width() / ratio
        return QRectF(0, (bounds.height() - height) / 2, bounds.width(), height)

    def watermark_rect(self) -> QRectF:
        if self._pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return QRectF()
        content = self.content_rect()
        width = content.width() * self._width_percent / 100.0
        height = width * self._pixmap.height() / max(1, self._pixmap.width())
        height = min(height, content.height())
        available_x = max(0.0, content.width() - width)
        available_y = max(0.0, content.height() - height)
        left = content.left() + available_x * self._x_percent / 100.0
        top = content.top() + available_y * self._y_percent / 100.0
        return QRectF(left, top, width, height)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt virtual method
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.watermark_rect()
        painter.setOpacity(self._opacity)
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.setOpacity(1.0)
        painter.setPen(QPen(QColor("#d56cff"), 2, Qt.PenStyle.DashLine))
        painter.drawRect(rect)
        painter.fillRect(self._resize_handle(rect), QColor("#d56cff"))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        rect = self.watermark_rect()
        position = event.position()
        if event.button() != Qt.MouseButton.LeftButton or not rect.contains(position):
            event.ignore()
            return
        self._drag_mode = "resize" if self._resize_handle(rect).contains(position) else "move"
        self._press_position = position
        self._press_values = (self._x_percent, self._y_percent, self._width_percent)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._drag_mode:
            event.ignore()
            return
        delta = event.position() - self._press_position
        old_x, old_y, old_width = self._press_values
        content = self.content_rect()
        if self._drag_mode == "resize":
            width = old_width + delta.x() * 100.0 / max(1.0, content.width())
            self._width_percent = min(80.0, max(2.0, width))
        else:
            rect = self.watermark_rect()
            available_x = max(1.0, content.width() - rect.width())
            available_y = max(1.0, content.height() - rect.height())
            self._x_percent = min(100.0, max(0.0, old_x + delta.x() * 100.0 / available_x))
            self._y_percent = min(100.0, max(0.0, old_y + delta.y() * 100.0 / available_y))
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._drag_mode:
            self._drag_mode = None
            self.placement_changed.emit(
                self._x_percent,
                self._y_percent,
                self._width_percent,
            )
            event.accept()
            return
        event.ignore()

    @staticmethod
    def _resize_handle(rect: QRectF) -> QRectF:
        size = 16.0
        return QRectF(rect.right() - size, rect.bottom() - size, size, size)
