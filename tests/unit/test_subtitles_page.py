from pathlib import Path

from cortaflow.domain.subtitle import SubtitleCue, Transcript, TranscriptWord
from cortaflow.ui.pages import subtitles_page
from cortaflow.ui.pages.subtitles_page import SubtitlesPage


def make_transcript(text: str = "Texto gerado") -> Transcript:
    return Transcript(
        language="pt",
        language_probability=0.95,
        words=[TranscriptWord(text="Texto", start_ms=0, end_ms=400)],
        cues=[SubtitleCue(start_ms=0, end_ms=800, text=text)],
    )


def test_manual_edit_is_marked_and_emitted(qtbot, tmp_path: Path) -> None:
    page = SubtitlesPage()
    qtbot.addWidget(page)
    page.set_media(tmp_path / "vídeo.mp4", make_transcript())

    with qtbot.waitSignal(page.transcript_changed, timeout=1_000):
        page.table.item(0, 2).setText("Texto corrigido")
    assert page.cues[0].text == "Texto corrigido"
    assert page.cues[0].manually_edited is True


def test_regeneration_preserves_manual_edit(qtbot, tmp_path: Path) -> None:
    page = SubtitlesPage()
    qtbot.addWidget(page)
    old = make_transcript("Correção humana")
    old = old.model_copy(
        update={"cues": [old.cues[0].model_copy(update={"manually_edited": True})]}
    )
    page.set_media(tmp_path / "vídeo.mp4", old)
    page._transcription_finished(make_transcript("Resultado novo"))
    assert page.cues[0].text == "Correção humana"


def test_transcription_runs_outside_ui_thread(qtbot, monkeypatch, tmp_path: Path) -> None:
    media = tmp_path / "vídeo.mp4"
    media.write_bytes(b"fixture")

    class FakeTranscriber:
        def __init__(self, model_size, model_cache, certificate_cache, allow_download=False):
            assert model_size == "small"
            assert model_cache.name == "faster-whisper"
            assert certificate_cache.name == "certificates"
            assert allow_download is False

        def transcribe(self, path, language, progress, cancelled):
            assert path == media
            assert language is None
            progress({"status": "transcribing", "device": "cpu", "position_ms": 800})
            return make_transcript()

    monkeypatch.setattr(subtitles_page, "FasterWhisperTranscriber", FakeTranscriber)
    page = SubtitlesPage()
    qtbot.addWidget(page)
    page.set_media(media)
    with qtbot.waitSignal(page.transcript_changed, timeout=5_000):
        page.start_transcription()
        assert page.current_worker is not None
        assert page.cancel_button.isEnabled()
    assert page.current_worker is None
    assert page.table.rowCount() == 1
    assert "concluída" in page.status.text()
