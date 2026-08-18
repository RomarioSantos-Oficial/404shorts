from pathlib import Path

from cortaflow.domain.analysis import TimeRange
from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.services.audio_analysis import extract_audio_evidence
from cortaflow.services import audio_analysis


def test_extracts_normalized_energy_and_combines_transcript_vad() -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    transcript = Transcript(
        language="pt",
        words=[TranscriptWord(text="voz", start_ms=200, end_ms=800)],
    )
    evidence = extract_audio_evidence(
        source,
        transcript=transcript,
        silences=[TimeRange(start_ms=600, end_ms=1000)],
    )
    assert evidence
    assert all(0 <= item.energy <= 1 for item in evidence)
    assert any(item.voice_active for item in evidence if item.timestamp_ms < 600)
    assert not any(item.voice_active for item in evidence if 600 <= item.timestamp_ms < 1000)


def test_video_without_audio_returns_no_evidence(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "vídeo sem áudio.mp4"
    source.write_bytes(b"fixture")

    class EmptyContainer:
        class Streams:
            audio: list = []

        streams = Streams()

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(audio_analysis.av, "open", lambda _path: EmptyContainer())

    assert extract_audio_evidence(source) == []
