from pathlib import Path

from cortaflow.domain.analysis import AnalysisResult, ClipSuggestion, TimeRange
from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.ui.pages import analysis_page
from cortaflow.ui.pages.analysis_page import AnalysisPage


def test_analysis_page_runs_worker_and_emits_result(qtbot, monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "vídeo.mp4"
    source.write_bytes(b"fixture")
    transcript = Transcript(
        language="pt",
        words=[TranscriptWord(text="Frase.", start_ms=0, end_ms=10_000)],
    )
    expected = AnalysisResult(
        scenes=[TimeRange(start_ms=0, end_ms=10_000)],
        suggestions=[
            ClipSuggestion(
                start_ms=0,
                end_ms=10_000,
                title="Frase",
                transcript_excerpt="Frase.",
                quality_score=0.8,
                reason="Frase completa.",
            )
        ],
    )

    def fake_analyze(path, received, duration, selection, progress, cancelled, **kwargs):
        assert path == source
        assert received == transcript
        assert duration == 10_000
        assert selection.min_seconds == 5
        progress({"message": "Pontuando trechos…"})
        return expected

    monkeypatch.setattr(analysis_page, "analyze_media", fake_analyze)
    page = AnalysisPage()
    qtbot.addWidget(page)
    page.set_context(source, transcript, 10_000)
    with qtbot.waitSignal(page.analysis_finished, timeout=5_000) as emitted:
        page.start_analysis()
        assert page.current_worker is not None
    assert emitted.args == [expected]
    assert page.current_worker is None
    assert "1 cenas" in page.summary.text()


def test_analysis_duration_controls_are_limited_and_emitted(qtbot) -> None:
    page = AnalysisPage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.selection_settings_changed, timeout=1_000) as emitted:
        page.maximum_seconds.setValue(30)
    assert emitted.args[0].max_seconds == 30
    assert emitted.args[0].preferred_seconds == 30
    assert page.minimum_seconds.minimum() == 5
    assert page.maximum_seconds.maximum() == 179


def test_analysis_ranking_mode_and_auto_accept_are_persisted(qtbot) -> None:
    page = AnalysisPage()
    qtbot.addWidget(page)
    page.ranking_mode.setCurrentIndex(page.ranking_mode.findData("heuristic"))
    page.auto_accept.setChecked(True)
    with qtbot.waitSignal(page.selection_settings_changed, timeout=1_000) as emitted:
        page.auto_accept_score.setValue(75)
    settings = emitted.args[0]
    assert settings.ranking_mode == "heuristic"
    assert settings.auto_accept_threshold == 0.75


def test_analysis_selection_goal_and_topic_are_persisted(qtbot) -> None:
    page = AnalysisPage()
    qtbot.addWidget(page)
    page.selection_goal.setCurrentIndex(page.selection_goal.findData("topic"))
    with qtbot.waitSignal(page.selection_settings_changed, timeout=1_000) as emitted:
        page.topic_prompt.setText("segurança em Python")
    settings = emitted.args[0]
    assert settings.selection_goal == "topic"
    assert settings.topic_prompt == "segurança em Python"
    assert page.topic_prompt.isEnabled()
