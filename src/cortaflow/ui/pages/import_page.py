"""Import page."""

from pathlib import Path
from typing import Any

from PySide6.QtCore import QThreadPool, Qt, QUrl, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cortaflow.domain.media import MediaMetadata
from cortaflow.services.downloader import download_media, inspect_url, validate_public_url
from cortaflow.services.media_probe import probe_media
from cortaflow.workers.base_worker import FunctionWorker


class ImportPage(QWidget):
    media_selected = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        title = QLabel("Importar mídia")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.thumbnail = QLabel("Miniatura")
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setFixedSize(320, 180)
        self.thumbnail.setStyleSheet("border: 1px solid #444b5c; background: #111319; color: #777d8b;")
        layout.addWidget(self.thumbnail, alignment=Qt.AlignmentFlag.AlignHCenter)

        row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Cole uma URL de vídeo autorizado")
        self.inspect_button = QPushButton("Analisar link")
        self.inspect_button.clicked.connect(self.inspect_link)
        row.addWidget(self.url_input, 1)
        row.addWidget(self.inspect_button)
        layout.addLayout(row)
        details = QFormLayout()
        self.title_value, self.duration_value, self.resolution_value = QLabel("—"), QLabel("—"), QLabel("—")
        details.addRow("Título", self.title_value)
        details.addRow("Duração", self.duration_value)
        details.addRow("Resolução", self.resolution_value)
        self.platform_value = QLabel("—")
        details.addRow("Plataforma", self.platform_value)
        self.quality = QComboBox()
        details.addRow("Qualidade", self.quality)
        layout.addLayout(details)
        folder_row = QHBoxLayout()
        self.folder_input = QLineEdit(str(Path.home() / "Videos" / "CortaFlow AI"))
        folder_button = QPushButton("Escolher pasta")
        folder_button.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.folder_input, 1); folder_row.addWidget(folder_button)
        layout.addLayout(folder_row)
        download_row = QHBoxLayout()
        self.download_button = QPushButton("Baixar")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.start_download)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_current)
        download_row.addWidget(self.download_button); download_row.addWidget(self.cancel_button); download_row.addStretch()
        layout.addLayout(download_row)
        self.local_button = QPushButton("Importar arquivo local")
        self.local_button.clicked.connect(self.choose_local_file)
        layout.addWidget(self.local_button)
        self.progress = QProgressBar()
        self.progress.hide()
        layout.addWidget(self.progress)
        self.transfer_status = QLabel("")
        layout.addWidget(self.transfer_status)
        layout.addStretch()
        self.thread_pool = QThreadPool.globalInstance()
        self.network = QNetworkAccessManager(self)
        self.thumbnail_reply: QNetworkReply | None = None
        self.current_worker: FunctionWorker | None = None
        self.current_metadata: MediaMetadata | None = None

    def inspect_link(self) -> None:
        url = self.url_input.text().strip()
        self.current_metadata = None
        self.quality.clear()
        self._load_thumbnail(None)
        self._set_busy(True, indeterminate=True)

        def operation(progress: Any, cancelled: Any) -> MediaMetadata:
            if cancelled.is_set():
                raise RuntimeError("Operação cancelada.")
            metadata = inspect_url(url)
            if cancelled.is_set():
                raise RuntimeError("Operação cancelada.")
            return metadata

        worker = FunctionWorker(operation)
        worker.signals.finished.connect(self._inspection_finished)
        worker.signals.failed.connect(self._operation_failed)
        self.current_worker = worker; self.thread_pool.start(worker)

    def _inspection_finished(self, metadata: MediaMetadata) -> None:
        self.current_worker = None
        self.current_metadata = metadata
        self.show_metadata(metadata)
        self.quality.clear()
        self.quality.addItem("Melhor qualidade disponível", None)
        for media_format in sorted(metadata.formats, key=lambda item: item.height or 0, reverse=True):
            self.quality.addItem(media_format.label, media_format.selector)
        self._set_busy(False)
        self.download_button.setEnabled(True)
        self._load_thumbnail(metadata.thumbnail_url)

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Pasta para downloads", self.folder_input.text())
        if folder: self.folder_input.setText(folder)

    def start_download(self) -> None:
        if not self.current_metadata or not self.confirm_authorization():
            return
        self._set_busy(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.transfer_status.setText("Preparando download…")
        worker = FunctionWorker(
            download_media,
            self.current_metadata.source,
            Path(self.folder_input.text()),
            self.quality.currentData(),
            self.current_metadata.title,
        )
        worker.signals.progress.connect(self._download_progress)
        worker.signals.finished.connect(self._download_finished)
        worker.signals.failed.connect(self._operation_failed)
        self.current_worker = worker; self.thread_pool.start(worker)

    def _download_progress(self, status: dict[str, Any]) -> None:
        if status.get("status") == "postprocessing":
            self.transfer_status.setText("Unindo vídeo e áudio com FFmpeg…")
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        downloaded = status.get("downloaded_bytes") or 0
        if total:
            self.progress.setValue(round(downloaded * 100 / total))
        speed = status.get("speed") or 0
        eta = status.get("eta")
        self.transfer_status.setText(f"{speed / 1_000_000:.1f} MB/s · ETA {eta if eta is not None else '—'} s")

    def _download_finished(self, path: Path) -> None:
        self.current_worker = None
        try:
            metadata = probe_media(path)
        except Exception as exc:
            self._operation_failed(str(exc))
            return
        self.current_metadata = None
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.show()
        self.transfer_status.setText(f"Download concluído: {path}")
        self.show_metadata(metadata)
        self.media_selected.emit(metadata)

    def _operation_failed(self, message: str) -> None:
        self.current_worker = None
        self._set_busy(False)
        if "cancelad" in message.lower():
            self.transfer_status.setText("Operação cancelada.")
            return
        self.transfer_status.setText("Operação não concluída.")
        QMessageBox.critical(self, "Operação não concluída", message)

    def cancel_current(self) -> None:
        if self.current_worker:
            self.current_worker.cancel()
            self.cancel_button.setEnabled(False)
            self.transfer_status.setText("Cancelando…")

    def _set_busy(self, busy: bool, indeterminate: bool = False) -> None:
        self.inspect_button.setEnabled(not busy); self.local_button.setEnabled(not busy)
        self.download_button.setEnabled(not busy and self.current_metadata is not None)
        self.cancel_button.setEnabled(busy)
        self.progress.setVisible(busy)
        if indeterminate:
            self.progress.setRange(0, 0)
        elif not busy:
            self.progress.setRange(0, 100)

    def choose_local_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Selecionar vídeo", "", "Vídeos (*.mp4 *.mkv *.mov *.avi *.webm);;Todos (*)")
        if not filename:
            return
        selected_path = Path(filename)
        self.current_metadata = None
        self._set_busy(True, indeterminate=True)
        self.transfer_status.setText("Lendo metadados do arquivo local…")

        def operation(progress: Any, cancelled: Any) -> MediaMetadata:
            if cancelled.is_set():
                raise RuntimeError("Operação cancelada.")
            metadata = probe_media(selected_path)
            if cancelled.is_set():
                raise RuntimeError("Operação cancelada.")
            return metadata

        worker = FunctionWorker(operation)
        worker.signals.finished.connect(self._local_probe_finished)
        worker.signals.failed.connect(self._operation_failed)
        self.current_worker = worker
        self.thread_pool.start(worker)

    def _local_probe_finished(self, metadata: MediaMetadata) -> None:
        self.current_worker = None
        self._set_busy(False)
        self.transfer_status.setText(f"Arquivo local selecionado: {metadata.local_path}")
        self._load_thumbnail(None)
        self.show_metadata(metadata)
        self.media_selected.emit(metadata)

    def show_metadata(self, metadata: MediaMetadata) -> None:
        self.title_value.setText(metadata.title)
        total_seconds = round(metadata.duration_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.duration_value.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.resolution_value.setText(f"{metadata.width or '?'} × {metadata.height or '?'}")
        self.platform_value.setText(metadata.platform)

    def _load_thumbnail(self, url: str | None) -> None:
        if self.thumbnail_reply is not None:
            previous_reply = self.thumbnail_reply
            self.thumbnail_reply = None
            previous_reply.abort()
            previous_reply.deleteLater()
        self.thumbnail.setPixmap(QPixmap())
        self.thumbnail.setText("Miniatura indisponível" if not url else "Carregando miniatura…")
        if not url:
            return
        try:
            safe_url = validate_public_url(url)
        except ValueError:
            self.thumbnail.setText("Miniatura indisponível")
            return
        request = QNetworkRequest(QUrl(safe_url))
        request.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, "CortaFlowAI/0.1")
        reply = self.network.get(request)
        self.thumbnail_reply = reply
        reply.finished.connect(lambda: self._thumbnail_finished(reply))

    def _thumbnail_finished(self, reply: QNetworkReply) -> None:
        if reply is not self.thumbnail_reply:
            reply.deleteLater()
            return
        self.thumbnail_reply = None
        data = bytes(reply.readAll())
        has_error = reply.error() != QNetworkReply.NetworkError.NoError
        reply.deleteLater()
        if has_error or not data or len(data) > 5_000_000:
            self.thumbnail.setText("Miniatura indisponível")
            return
        image = QImage.fromData(data)
        if image.isNull():
            self.thumbnail.setText("Miniatura inválida")
            return
        pixmap = QPixmap.fromImage(image).scaled(
            self.thumbnail.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.thumbnail.setText("")
        self.thumbnail.setPixmap(pixmap)

    def confirm_authorization(self) -> bool:
        answer = QMessageBox.question(self, "Confirmação de autorização", "Confirmo que este vídeo me pertence, é licenciado ou tenho autorização para utilizá-lo.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes
