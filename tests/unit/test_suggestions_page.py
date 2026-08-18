from pathlib import Path
import shutil

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.services.automatic_pipeline import AutomaticPreview
from cortaflow.ui.pages.suggestions_page import SuggestionsPage
from cortaflow.ui.pages import suggestions_page


def suggestion() -> ClipSuggestion:
    return ClipSuggestion(
        start_ms=1_000,
        end_ms=16_000,
        title="Título",
        transcript_excerpt="Uma frase completa.",
        quality_score=0.85,
        reason="Boa densidade.",
        editorial_status="validated",
        editorial_score=1.0,
        relevance_score=0.8,
        confidence_score=0.9,
    )


def test_suggestion_can_be_adjusted_accepted_and_opened(qtbot) -> None:
    page = SuggestionsPage()
    qtbot.addWidget(page)
    page.set_suggestions([suggestion()])
    page.table.selectRow(0)

    with qtbot.waitSignal(page.suggestions_changed, timeout=1_000):
        page.table.item(0, 0).setText("00:00:02.000")
    assert page.suggestions[0].start_ms == 2_000
    assert page.table.item(0, 2).text() == "00:00:14.000"

    with qtbot.waitSignal(page.suggestions_changed, timeout=1_000):
        page.set_status("accepted")
    assert page.suggestions[0].status == "accepted"
    assert page.table.item(0, 8).text() == "Aceito"

    with qtbot.waitSignal(page.open_requested, timeout=1_000) as emitted:
        page.open_selected()
    assert emitted.args[0] == page.suggestions[0]

    with qtbot.waitSignal(page.review_requested, timeout=1_000) as review:
        page.review_button.click()
    assert review.args[0] == page.suggestions[0]


def test_suggestions_support_batch_acceptance_and_preview_mapping(qtbot, tmp_path) -> None:
    page = SuggestionsPage()
    qtbot.addWidget(page)
    low = suggestion().model_copy(update={"quality_score": 0.6})
    high = suggestion().model_copy(
        update={"start_ms": 20_000, "end_ms": 35_000, "quality_score": 0.85}
    )
    page.set_suggestions([low, high])
    preview = tmp_path / "preview.mp4"
    preview.write_bytes(b"fixture")
    page.set_previews(
        [AutomaticPreview(1, preview, True, True, "validated", 1, "MP4 conferido.")]
    )
    assert page.table.item(1, 9).text() == "Legenda + rosto conferido"
    assert page.table.item(0, 9).text() == "Original"
    page.batch_score.setValue(70)
    with qtbot.waitSignal(page.suggestions_changed, timeout=1_000):
        page.accept_by_score()
    assert page.suggestions[0].status == "pending"
    assert page.suggestions[1].status == "accepted"
    assert page.export_button.isEnabled()
    with qtbot.waitSignal(page.export_requested, timeout=1_000):
        page.export_button.click()
    with qtbot.waitSignal(page.suggestions_changed, timeout=1_000):
        page.set_all_status("rejected")
    assert all(item.status == "rejected" for item in page.suggestions)
    assert not page.export_button.isEnabled()


def test_selected_cut_loads_embedded_source_then_rendered_preview(qtbot, tmp_path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    rendered = tmp_path / "vertical-com-legenda.mp4"
    shutil.copyfile(source, rendered)
    page = SuggestionsPage()
    qtbot.addWidget(page)
    page.set_media_source(source)
    page.set_suggestions([suggestion()])

    assert Path(page.preview_player.source().toLocalFile()) == source.resolve()
    assert page.preview_start_ms == 1_000
    assert page.preview_end_ms == 16_000
    assert "intervalo original" in page.preview_status.text()

    page.set_previews(
        [AutomaticPreview(0, rendered, True, True, "validated", 1, "MP4 conferido.")]
    )

    assert Path(page.preview_player.source().toLocalFile()) == rendered.resolve()
    assert page.preview_start_ms == 0
    assert page.preview_end_ms is None
    assert "Legenda + rosto conferido" in page.preview_status.text()


def test_previous_and_next_review_one_cut_at_a_time(qtbot) -> None:
    page = SuggestionsPage()
    qtbot.addWidget(page)
    second = suggestion().model_copy(update={"start_ms": 20_000, "end_ms": 30_000})
    page.set_suggestions([suggestion(), second])
    assert page.table.currentRow() == 0
    assert not page.previous_preview_button.isEnabled()
    assert page.next_preview_button.isEnabled()

    page.next_preview_button.click()

    assert page.table.currentRow() == 1
    assert page.previous_preview_button.isEnabled()
    assert not page.next_preview_button.isEnabled()


def test_processing_state_explains_empty_results_and_enables_buttons(qtbot) -> None:
    page = SuggestionsPage()
    qtbot.addWidget(page)
    assert not page.open_preview_button.isEnabled()
    page.set_processing(True, "Analisando cenas e rostos…")
    assert page.processing_progress.isVisibleTo(page)
    assert "rostos" in page.processing_status.text()

    page.set_suggestions([suggestion()])

    assert page.open_preview_button.isEnabled()


def test_unsafe_framing_requires_review_before_acceptance(qtbot, monkeypatch) -> None:
    page = SuggestionsPage()
    qtbot.addWidget(page)
    unsafe = suggestion().model_copy(
        update={"framing_status": "needs_review", "framing_score": .7, "visible_faces": 2}
    )
    warnings = []
    monkeypatch.setattr(
        suggestions_page.QMessageBox,
        "warning",
        lambda *args: warnings.append(args[2]),
    )
    page.set_suggestions([unsafe])

    page.set_status("accepted")

    assert page.suggestions[0].status == "pending"
    assert warnings and "área segura" in warnings[0]
    assert page.table.item(0, 7).text() == "Revisar · 70% seguro"


def test_editorial_review_blocks_batch_but_allows_explicit_single_review(qtbot) -> None:
    page = SuggestionsPage()
    qtbot.addWidget(page)
    pending = suggestion().model_copy(update={"editorial_status": "needs_review"})
    page.set_suggestions([pending])

    page.set_all_status("accepted")
    page.batch_score.setValue(0)
    page.accept_by_score()

    assert page.suggestions[0].status == "pending"
    assert "Revisar ideia/limites" in page.table.item(0, 6).text()

    page.set_status("accepted")
    assert page.suggestions[0].status == "accepted"
    assert page.suggestions[0].editorial_status == "validated"
