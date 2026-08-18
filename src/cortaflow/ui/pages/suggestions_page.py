"""Suggested clip review and adjustment page."""

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.clip import format_timestamp


class SuggestionsPage(QWidget):
    suggestions_changed = Signal(object)
    open_requested = Signal(object)
    review_requested = Signal(object)
    export_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.suggestions: list[ClipSuggestion] = []
        self.preview_paths: dict[int, Path] = {}
        self.preview_labels: dict[int, str] = {}
        self.source_path: Path | None = None
        self.preview_start_ms = 0
        self.preview_end_ms: int | None = None
        self._pending_seek_ms: int | None = None
        self._play_when_loaded = False
        self._updating = False
        self.preview_player = QMediaPlayer(self)
        self.preview_audio = QAudioOutput(self)
        self.preview_video = QVideoWidget(self)
        self.preview_player.setAudioOutput(self.preview_audio)
        self.preview_player.setVideoOutput(self.preview_video)
        self.preview_audio.setVolume(0.8)
        root_layout = QVBoxLayout(self)
        self.workspace_tabs = QTabWidget(self)
        self.results_tab = QWidget(self.workspace_tabs)
        self.review_tab = QWidget(self.workspace_tabs)
        layout = QVBoxLayout(self.results_tab)
        self.review_layout = QVBoxLayout(self.review_tab)
        self.review_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_tabs.addTab(self.results_tab, "Cortes encontrados")
        self.workspace_tabs.addTab(self.review_tab, "Editar e salvar")
        self.workspace_tabs.setTabEnabled(1, False)
        root_layout.addWidget(self.workspace_tabs)
        self.processing_status = QLabel(
            "Importe/transcreva a mídia e clique em Gerar cortes sugeridos."
        )
        self.processing_status.setWordWrap(True)
        layout.addWidget(self.processing_status)
        self.processing_progress = QProgressBar()
        self.processing_progress.setRange(0, 0)
        self.processing_progress.hide()
        layout.addWidget(self.processing_progress)
        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Início", "Fim", "Duração", "Título", "Trecho", "Potencial", "Motivo",
                "Enquadramento", "Status", "Versão",
            ]
        )
        for column in (3, 4, 6, 7):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._item_edited)
        self.table.currentCellChanged.connect(self._selection_changed)
        layout.addWidget(self.table, 2)
        self.preview_video.setMinimumHeight(240)
        layout.addWidget(self.preview_video, 1)
        self.preview_status = QLabel(
            "Selecione um corte. Durante a geração, a prévia rápida usa o vídeo original; "
            "depois ela é substituída pela versão vertical com legenda."
        )
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)
        row = QHBoxLayout()
        self.accept_button = QPushButton("Aceitar")
        self.accept_button.clicked.connect(lambda: self.set_status("accepted"))
        self.reject_button = QPushButton("Rejeitar")
        self.reject_button.clicked.connect(lambda: self.set_status("rejected"))
        self.open_button = QPushButton("Abrir no editor")
        self.open_button.clicked.connect(self.open_selected)
        self.review_button = QPushButton("Editar e salvar este corte")
        self.review_button.clicked.connect(self.review_selected)
        self.previous_preview_button = QPushButton("◀ Anterior")
        self.previous_preview_button.clicked.connect(self.select_previous)
        self.open_preview_button = QPushButton("Assistir corte")
        self.open_preview_button.clicked.connect(self.open_preview)
        self.next_preview_button = QPushButton("Próximo ▶")
        self.next_preview_button.clicked.connect(self.select_next)
        row.addWidget(self.accept_button)
        row.addWidget(self.reject_button)
        row.addWidget(self.open_button)
        row.addWidget(self.review_button)
        row.addWidget(self.previous_preview_button)
        row.addWidget(self.open_preview_button)
        row.addWidget(self.next_preview_button)
        row.addStretch()
        layout.addLayout(row)
        batch_row = QHBoxLayout()
        self.accept_all_button = QPushButton("Aceitar todos")
        self.accept_all_button.clicked.connect(lambda: self.set_all_status("accepted"))
        self.reject_all_button = QPushButton("Rejeitar todos")
        self.reject_all_button.clicked.connect(lambda: self.set_all_status("rejected"))
        batch_row.addWidget(self.accept_all_button)
        batch_row.addWidget(self.reject_all_button)
        batch_row.addWidget(QLabel("Aceitar nota mínima"))
        self.batch_score = QSpinBox()
        self.batch_score.setRange(0, 100)
        self.batch_score.setSuffix("%")
        self.batch_score.setValue(70)
        batch_row.addWidget(self.batch_score)
        self.accept_score_button = QPushButton("Aplicar por nota")
        self.accept_score_button.clicked.connect(self.accept_by_score)
        batch_row.addWidget(self.accept_score_button)
        self.export_button = QPushButton("Editar e salvar aceitos em lote")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_requested.emit)
        batch_row.addWidget(self.export_button)
        batch_row.addStretch()
        layout.addLayout(batch_row)
        self.preview_player.positionChanged.connect(self._preview_position_changed)
        self.preview_player.mediaStatusChanged.connect(self._preview_media_status_changed)
        self.preview_player.playbackStateChanged.connect(self._preview_playback_changed)
        self._update_actions()

    def attach_review_widget(self, widget: QWidget) -> None:
        """Place the final clip editor inside the suggested-cuts workspace."""
        self.review_layout.addWidget(widget)

    def show_results(self) -> None:
        self.workspace_tabs.setCurrentIndex(0)

    def show_review(self) -> None:
        self.workspace_tabs.setTabEnabled(1, True)
        self.workspace_tabs.setCurrentIndex(1)

    def set_media_source(self, source_path: Path | None) -> None:
        self.source_path = source_path.resolve() if source_path else None
        if self.table.currentRow() >= 0:
            self._load_selected_media()

    def set_processing(self, active: bool, message: str) -> None:
        self.processing_status.setText(message)
        self.processing_progress.setVisible(active)

    def set_suggestions(self, suggestions: list[ClipSuggestion]) -> None:
        self.suggestions = list(suggestions)
        self.workspace_tabs.setCurrentIndex(0)
        self.workspace_tabs.setTabEnabled(1, False)
        self.preview_paths = {}
        self.preview_labels = {}
        self._updating = True
        self.table.setRowCount(len(suggestions))
        for row, item in enumerate(suggestions):
            values = (
                format_timestamp(item.start_ms, True),
                format_timestamp(item.end_ms, True),
                format_timestamp(item.duration_ms, True),
                item.title,
                item.transcript_excerpt,
                f"{item.quality_score:.0%}",
                f"{self._editorial_text(item)} · {item.reason}",
                self._framing_text(item),
                self._status_text(item.status),
                "Original",
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if column not in (0, 1, 3):
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, cell)
            details = self._components_text(item.score_components)
            relevance = "—" if item.relevance_score is None else f"{item.relevance_score:.0%}"
            confidence = "—" if item.confidence_score is None else f"{item.confidence_score:.0%}"
            self.table.item(row, 5).setToolTip(
                f"Validade: {self._editorial_text(item)} · Relevância: {relevance} · "
                f"Confiança: {confidence} · Tendência: não avaliada\nComponentes: {details}"
            )
            self.table.item(row, 7).setToolTip(
                "O rosto falante/principal foi verificado em todas as amostras do intervalo, "
                "inclusive nas trocas de pessoa."
            )
        self._updating = False
        if suggestions:
            self.table.selectRow(0)
            self.processing_status.setText(
                f"{len(suggestions)} cortes encontrados. Preparando versões verticais com legenda…"
            )
        else:
            self.preview_player.stop()
            self.preview_player.setSource(QUrl())
            self.processing_status.setText(
                "Nenhum corte disponível ainda. Clique em Gerar cortes sugeridos."
            )
        self._update_export_button()
        self._update_actions()

    def set_previews(self, previews: list) -> None:
        self.preview_paths = {
            int(item.suggestion_index): Path(item.path)
            for item in previews
            if 0 <= int(item.suggestion_index) < len(self.suggestions)
        }
        self.preview_labels = {
            int(item.suggestion_index): self._preview_label(item)
            for item in previews
            if 0 <= int(item.suggestion_index) < len(self.suggestions)
        }
        for row in range(len(self.suggestions)):
            cell = self.table.item(row, 9)
            if cell:
                cell.setText(self.preview_labels.get(row, "Original"))
        if self.table.currentRow() >= 0:
            self._load_selected_media()

    def set_status(self, status: str) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self.suggestions):
            return
        if status == "accepted" and self.suggestions[row].framing_status == "needs_review":
            QMessageBox.warning(
                self,
                "Revise o enquadramento",
                "Este corte tem uma amostra facial fora da área segura. Use Editar e salvar "
                "este corte, ajuste o enquadramento e aprove a nova prévia.",
            )
            return
        payload = self.suggestions[row].model_dump()
        payload["status"] = status
        # Clicking Aceitar is an explicit one-by-one human review. Automatic
        # and batch acceptance remain blocked for editorially uncertain cuts.
        if status == "accepted":
            payload["editorial_status"] = "validated"
        self.suggestions[row] = ClipSuggestion.model_validate(payload)
        self.table.item(row, 8).setText(self._status_text(status))
        self.table.item(row, 6).setText(
            f"{self._editorial_text(self.suggestions[row])} · {self.suggestions[row].reason}"
        )
        self._update_export_button()
        self.suggestions_changed.emit(list(self.suggestions))

    def open_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.suggestions):
            self.open_requested.emit(self.suggestions[row])

    def review_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.suggestions):
            self.review_requested.emit(self.suggestions[row])

    def preview_for(self, suggestion: ClipSuggestion) -> Path | None:
        for row, item in enumerate(self.suggestions):
            if (
                item.start_ms == suggestion.start_ms
                and item.end_ms == suggestion.end_ms
                and item.title == suggestion.title
            ):
                path = self.preview_paths.get(row)
                return path.resolve() if path and path.is_file() else None
        return None

    def mark_suggestion_saved(self, suggestion: ClipSuggestion) -> None:
        """Mark the exact suggestion accepted after its final file was written."""
        for row, item in enumerate(self.suggestions):
            if (
                item.start_ms == suggestion.start_ms
                and item.end_ms == suggestion.end_ms
                and item.title == suggestion.title
            ):
                self.suggestions[row] = item.model_copy(
                    update={"status": "accepted", "editorial_status": "validated"}
                )
                self.table.item(row, 8).setText(self._status_text("accepted"))
                self.table.selectRow(row)
                self._update_export_button()
                self.suggestions_changed.emit(list(self.suggestions))
                return

    def open_preview(self) -> None:
        if not 0 <= self.table.currentRow() < len(self.suggestions):
            return
        if self.preview_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.preview_player.pause()
            return
        self._play_when_loaded = True
        self._load_selected_media()
        if self.preview_player.mediaStatus() in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            self._play_when_loaded = False
            self.preview_player.setPosition(self.preview_start_ms)
            self.preview_player.play()

    def select_previous(self) -> None:
        row = self.table.currentRow()
        if self.suggestions:
            self.table.selectRow(max(0, row - 1))

    def select_next(self) -> None:
        row = self.table.currentRow()
        if self.suggestions:
            self.table.selectRow(min(len(self.suggestions) - 1, max(0, row + 1)))

    def set_all_status(self, status: str) -> None:
        self.suggestions = [
            item.model_copy(
                update={
                    "status": (
                        item.status
                        if status == "accepted" and (
                            item.framing_status == "needs_review"
                            or item.editorial_status != "validated"
                        )
                        else status
                    )
                }
            )
            for item in self.suggestions
        ]
        for row in range(len(self.suggestions)):
            self.table.item(row, 8).setText(self._status_text(self.suggestions[row].status))
        self._update_export_button()
        self.suggestions_changed.emit(list(self.suggestions))

    def accept_by_score(self) -> None:
        threshold = self.batch_score.value() / 100
        self.suggestions = [
            item.model_copy(
                update={"status": "accepted" if item.quality_score >= threshold else item.status}
            )
            if (
                item.framing_status != "needs_review"
                and item.editorial_status == "validated"
            )
            else item
            for item in self.suggestions
        ]
        for row, item in enumerate(self.suggestions):
            self.table.item(row, 8).setText(self._status_text(item.status))
        self._update_export_button()
        self.suggestions_changed.emit(list(self.suggestions))

    def _update_export_button(self) -> None:
        self.export_button.setEnabled(
            any(item.status == "accepted" for item in self.suggestions)
        )

    def _update_actions(self) -> None:
        has_selection = 0 <= self.table.currentRow() < len(self.suggestions)
        for button in (
            self.accept_button,
            self.reject_button,
            self.open_button,
            self.review_button,
            self.open_preview_button,
        ):
            button.setEnabled(has_selection)
        row = self.table.currentRow()
        self.previous_preview_button.setEnabled(has_selection and row > 0)
        self.next_preview_button.setEnabled(
            has_selection and row < len(self.suggestions) - 1
        )
        for button in (
            self.accept_all_button,
            self.reject_all_button,
            self.accept_score_button,
        ):
            button.setEnabled(bool(self.suggestions))

    def _selection_changed(self, current_row: int, _current_column: int, *_args) -> None:
        self._update_actions()
        if 0 <= current_row < len(self.suggestions):
            self._play_when_loaded = False
            self._load_selected_media()

    def _load_selected_media(self) -> None:
        row = self.table.currentRow()
        if not 0 <= row < len(self.suggestions):
            return
        rendered = self.preview_paths.get(row)
        if rendered and rendered.is_file():
            path = rendered.resolve()
            start_ms = 0
            end_ms = None
            description = (
                f"{self.preview_labels.get(row, 'Prévia vertical pronta')}. "
                "Esta é exatamente a versão do corte selecionado para sua revisão."
            )
        elif self.source_path and self.source_path.is_file():
            suggestion = self.suggestions[row]
            path = self.source_path
            start_ms = suggestion.start_ms
            end_ms = suggestion.end_ms
            description = (
                "Visualização temporária do intervalo original; a versão vertical com legenda "
                "ainda está sendo renderizada."
            )
        else:
            self.preview_status.setText("A mídia original não está disponível para esta prévia.")
            return
        self.preview_start_ms = start_ms
        self.preview_end_ms = end_ms
        self._pending_seek_ms = start_ms
        current_path = Path(self.preview_player.source().toLocalFile()) if self.preview_player.source().isLocalFile() else None
        if current_path != path:
            self.preview_player.stop()
            self.preview_player.setSource(QUrl.fromLocalFile(str(path)))
        else:
            self.preview_player.setPosition(start_ms)
        self.preview_status.setText(description)

    def _preview_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status not in (
            QMediaPlayer.MediaStatus.LoadedMedia,
            QMediaPlayer.MediaStatus.BufferedMedia,
        ):
            return
        if self._pending_seek_ms is not None:
            self.preview_player.setPosition(self._pending_seek_ms)
            self._pending_seek_ms = None
        if self._play_when_loaded:
            self._play_when_loaded = False
            self.preview_player.play()

    def _preview_position_changed(self, position: int) -> None:
        if self.preview_end_ms is not None and position >= self.preview_end_ms:
            self.preview_player.pause()
            self.preview_player.setPosition(self.preview_start_ms)

    def _preview_playback_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.open_preview_button.setText(
            "Pausar corte"
            if state == QMediaPlayer.PlaybackState.PlayingState
            else "Assistir corte"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt virtual method
        self.preview_player.stop()
        self.preview_player.setSource(QUrl())
        self.preview_player.setVideoOutput(None)
        self.preview_player.setAudioOutput(None)
        super().closeEvent(event)

    @staticmethod
    def _preview_label(preview: object) -> str:
        subtitles = bool(getattr(preview, "subtitles_applied", False))
        reframe = bool(getattr(preview, "reframe_applied", False))
        framing_status = str(getattr(preview, "framing_status", "needs_review"))
        if framing_status == "needs_review":
            return "Revisar rosto no MP4"
        if framing_status == "no_face":
            return "Sem rosto · centro"
        if subtitles and reframe:
            return "Legenda + rosto conferido"
        if subtitles:
            return "Com legenda · central"
        if reframe:
            return "Com foco · sem legenda"
        return "Vertical pronta"

    def _item_edited(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() not in (0, 1, 3):
            return
        row = item.row()
        if not 0 <= row < len(self.suggestions):
            return
        original = self.suggestions[row]
        payload = original.model_dump()
        try:
            if item.column() == 0:
                payload["start_ms"] = self._parse_timestamp(item.text())
                payload["editorial_status"] = "needs_review"
            elif item.column() == 1:
                payload["end_ms"] = self._parse_timestamp(item.text())
                payload["editorial_status"] = "needs_review"
            else:
                payload["title"] = item.text().strip() or "Trecho sugerido"
            updated = ClipSuggestion.model_validate(payload)
        except ValueError as exc:
            self._updating = True
            replacement = (
                format_timestamp(original.start_ms, True)
                if item.column() == 0
                else format_timestamp(original.end_ms, True)
                if item.column() == 1
                else original.title
            )
            item.setText(replacement)
            self._updating = False
            QMessageBox.warning(self, "Ajuste inválido", str(exc))
            return
        self.suggestions[row] = updated
        self._updating = True
        self.table.item(row, 2).setText(format_timestamp(updated.duration_ms, True))
        self._updating = False
        self.suggestions_changed.emit(list(self.suggestions))

    @staticmethod
    def _parse_timestamp(value: str) -> int:
        parts = value.strip().replace(",", ".").split(":")
        if len(parts) != 3:
            raise ValueError("Use o formato HH:MM:SS.mmm.")
        hours, minutes = int(parts[0]), int(parts[1])
        seconds = float(parts[2])
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError("Tempo fora do intervalo válido.")
        return round((hours * 3600 + minutes * 60 + seconds) * 1000)

    @staticmethod
    def _status_text(status: str) -> str:
        return {"pending": "Pendente", "accepted": "Aceito", "rejected": "Rejeitado"}[status]

    @staticmethod
    def _components_text(components: dict[str, float]) -> str:
        return " · ".join(f"{name} {value:.0%}" for name, value in components.items()) or "—"

    @staticmethod
    def _editorial_text(suggestion: ClipSuggestion) -> str:
        return {
            "validated": "Ideia completa validada",
            "needs_review": "Revisar ideia/limites",
            "pending": "Validade pendente",
        }[suggestion.editorial_status]

    @staticmethod
    def _framing_text(suggestion: ClipSuggestion) -> str:
        if suggestion.framing_status == "validated":
            changes = f" · {suggestion.speaker_changes} troca(s)" if suggestion.speaker_changes else ""
            face_label = "rosto" if suggestion.visible_faces == 1 else "rostos"
            return f"Validado · {suggestion.visible_faces} {face_label}{changes}"
        if suggestion.framing_status == "no_face":
            return "Sem rosto · centro"
        if suggestion.framing_status == "needs_review":
            score = suggestion.framing_score or 0
            return f"Revisar · {score:.0%} seguro"
        return "Validando…"
