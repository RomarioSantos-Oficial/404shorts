from PySide6.QtWidgets import QPushButton

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.subtitle import SubtitleCue, Transcript, TranscriptWord
from cortaflow.ui import main_window
from cortaflow.ui.main_window import MainWindow
from cortaflow.ui.pages.history_page import HistoryPage


def test_main_window_has_all_navigation_steps(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "CortaFlow AI"
    assert window.navigation.count() == 6
    assert window.pages.count() == 6
    assert isinstance(window.pages.widget(5), HistoryPage)
    assert window.automatic_action.text() == "Gerar cortes sugeridos"
    assert window.suggestions_page.isAncestorOf(window.export_page)
    assert not window.automatic_action.isEnabled()


def test_every_button_in_the_automatic_review_flow_is_connected(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    flow_pages = [
        window.import_page,
        window.analysis_page,
        window.suggestions_page,
        window.editor_page,
        window.subtitles_page,
        window.history_page,
    ]
    flow_buttons = [button for page in flow_pages for button in page.findChildren(QPushButton)]

    # O editor não destrutivo adiciona comandos ao fluxo; evite acoplar o teste
    # a uma contagem fixa quando novos controles forem incluídos.
    assert len(flow_buttons) >= 53
    assert all(
        button.receivers("2clicked()") >= 1 or button.receivers("2toggled(bool)") >= 1
        for button in flow_buttons
    )
    button_texts = {button.text() for button in flow_buttons}
    assert {"Exportar sequência", "+ Texto", "+ Imagem", "Cancelar exportação"}.issubset(button_texts)


def test_automatic_toolbar_buttons_start_and_request_cancellation(
    qtbot, monkeypatch, tmp_path
) -> None:
    class FakePool:
        workers = []

        @classmethod
        def globalInstance(cls):
            return cls

        @classmethod
        def start(cls, worker):
            cls.workers.append(worker)

    window = MainWindow()
    qtbot.addWidget(window)
    source = tmp_path / "vídeo.mp4"
    source.write_bytes(b"fixture")
    word = TranscriptWord(text="Ideia.", start_ms=0, end_ms=6_000)
    window.project.source_path = source
    window.project.transcript = Transcript(
        language="pt",
        words=[word],
        cues=[SubtitleCue(start_ms=0, end_ms=6_000, text="Ideia.")],
    )
    window.automatic_action.setEnabled(True)
    monkeypatch.setattr(main_window, "QThreadPool", FakePool)

    window.automatic_action.trigger()

    assert len(FakePool.workers) == 1
    assert window.automatic_worker is FakePool.workers[0]
    assert not window.automatic_action.isEnabled()
    assert window.cancel_automatic_action.isEnabled()
    assert window.navigation.currentRow() == 2
    assert window.suggestions_page.processing_progress.isVisibleTo(window.suggestions_page)

    window.cancel_automatic_action.trigger()

    assert window.automatic_worker.cancel_event.is_set()
    assert not window.cancel_automatic_action.isEnabled()
    assert "Cancelamento solicitado" in window.suggestions_page.processing_status.text()


def test_settings_toolbar_button_opens_local_settings_dialog(qtbot, monkeypatch) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    calls = []
    monkeypatch.setattr(
        main_window.QInputDialog,
        "getInt",
        lambda *args, **kwargs: (calls.append(args[1]) or 1, False),
    )
    settings_action = next(
        action for action in window.findChildren(main_window.QAction)
        if action.text() == "Configurações"
    )

    settings_action.trigger()

    assert calls == ["Configurações locais"]


def test_selected_suggestion_opens_embedded_editor_with_exact_range(qtbot, tmp_path) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    source = tmp_path / "fonte.mp4"
    source.write_bytes(b"fixture")
    suggestion = ClipSuggestion(
        start_ms=2_500,
        end_ms=8_750,
        title="Trecho",
        transcript_excerpt="Uma ideia completa.",
        quality_score=.9,
        reason="Bom gancho.",
        framing_status="validated",
        framing_score=1,
        visible_faces=1,
    )
    window.project.source_path = source
    window.project.source_metadata = {"duration_seconds": 102.274, "width": 1920, "height": 1080}
    window.project.suggestions = [suggestion]
    window.suggestions_page.set_suggestions([suggestion])

    window._review_suggestion(suggestion)

    assert window.navigation.currentRow() == 2
    assert window.suggestions_page.workspace_tabs.currentIndex() == 1
    assert window.export_page.start_seconds.value() == 2.5
    assert window.export_page.end_seconds.value() == 8.75
    assert not window.export_page.timeline_mode.isChecked()
    assert window.export_page.start_seconds.isEnabled()
    assert window.export_page.end_seconds.isEnabled()


def test_toolbar_saves_the_cut_open_in_individual_review_without_batch_acceptance(
    qtbot, monkeypatch, tmp_path
) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    source = tmp_path / "fonte.mp4"
    source.write_bytes(b"fixture")
    suggestion = ClipSuggestion(
        start_ms=1_000,
        end_ms=5_000,
        title="Trecho aberto",
        transcript_excerpt="Ideia.",
        quality_score=.9,
        reason="Completo.",
    )
    window.project.source_path = source
    window.project.source_metadata = {
        "duration_seconds": 10,
        "width": 1920,
        "height": 1080,
    }
    window.project.suggestions = [suggestion]
    window.suggestions_page.set_suggestions([suggestion])
    window._review_suggestion(suggestion)
    calls = []
    monkeypatch.setattr(
        window.export_page,
        "save_current_suggestion",
        lambda: calls.append("individual") or True,
    )
    monkeypatch.setattr(
        window.export_page,
        "prepare_accepted",
        lambda: calls.append("batch") or False,
    )

    window.export_action.trigger()

    assert calls == ["individual"]
