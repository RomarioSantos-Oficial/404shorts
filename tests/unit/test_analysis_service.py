from pathlib import Path
from threading import Event

from cortaflow.domain.analysis import ClipSelectionSettings, TimeRange
from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.services import analysis_service
from cortaflow.services.analysis_service import analyze_media


def test_analysis_combines_scenes_silences_and_suggestions(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "vídeo.mp4"
    source.write_bytes(b"fixture")
    transcript = Transcript(
        language="pt",
        words=[
            TranscriptWord(
                text=f"palavra{index}{'.' if index % 20 == 19 else ''}",
                start_ms=index * 500,
                end_ms=(index + 1) * 500,
            )
            for index in range(60)
        ],
    )
    monkeypatch.setattr(
        analysis_service,
        "detect_scenes",
        lambda path, progress, cancelled: [TimeRange(start_ms=0, end_ms=30_000)],
    )
    monkeypatch.setattr(
        analysis_service,
        "detect_silences",
        lambda path, progress, cancelled: [TimeRange(start_ms=12_000, end_ms=13_000)],
    )
    monkeypatch.setattr(analysis_service, "extract_audio_evidence", lambda *args, **kwargs: [])
    updates: list[dict] = []
    result = analyze_media(
        source,
        transcript,
        30_000,
        ClipSelectionSettings(min_seconds=5, preferred_seconds=15, max_seconds=30),
        updates.append,
        Event(),
    )
    assert len(result.scenes) == 1
    assert len(result.silences) == 1
    assert result.suggestions
    assert any(update["status"] == "suggestions" for update in updates)
    assert updates[-1]["status"] == "heuristic_ranking"
