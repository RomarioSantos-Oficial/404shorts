from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkReply
from PySide6.QtWidgets import QMessageBox

from cortaflow.domain.media import MediaFormat, MediaMetadata
from cortaflow.ui.pages.import_page import ImportPage


def test_inspected_metadata_populates_quality_selectors(qtbot) -> None:
    page = ImportPage()
    qtbot.addWidget(page)
    metadata = MediaMetadata(
        source="https://example.com/video",
        title="Vídeo autorizado",
        duration_seconds=3661,
        width=1920,
        height=1080,
        platform="Example",
        formats=[
            MediaFormat(
                format_id="137",
                selector="137+ba/137",
                label="1080p · mp4 · vídeo + melhor áudio",
                width=1920,
                height=1080,
                extension="mp4",
            )
        ],
    )
    page._inspection_finished(metadata)
    assert page.quality.count() == 2
    assert page.quality.itemData(1) == "137+ba/137"
    assert page.duration_value.text() == "01:01:01"
    assert page.download_button.isEnabled()


def test_progress_displays_speed_eta_and_postprocessing(qtbot) -> None:
    page = ImportPage()
    qtbot.addWidget(page)
    page._download_progress({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 100, "speed": 2_000_000, "eta": 4})
    assert page.progress.value() == 50
    assert "2.0 MB/s" in page.transfer_status.text()
    assert "ETA 4 s" in page.transfer_status.text()
    page._download_progress({"status": "postprocessing"})
    assert "FFmpeg" in page.transfer_status.text()


def test_thumbnail_bytes_are_rendered_without_external_network(qtbot) -> None:
    page = ImportPage()
    qtbot.addWidget(page)
    image = QImage(4, 4, QImage.Format.Format_RGB32)
    image.fill(0x6257D9)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert image.save(buffer, "PNG")

    class FakeReply:
        def readAll(self):
            return data

        def error(self):
            return QNetworkReply.NetworkError.NoError

        def deleteLater(self):
            pass

    reply = FakeReply()
    page.thumbnail_reply = reply
    page._thumbnail_finished(reply)
    assert page.thumbnail.pixmap() is not None
    assert not page.thumbnail.pixmap().isNull()


def test_authorization_requires_explicit_yes(qtbot, monkeypatch) -> None:
    page = ImportPage()
    qtbot.addWidget(page)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.No)
    assert not page.confirm_authorization()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    assert page.confirm_authorization()
