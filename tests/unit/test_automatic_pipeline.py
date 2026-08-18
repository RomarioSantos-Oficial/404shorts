from pathlib import Path

import pytest

from cortaflow.domain.analysis import AnalysisResult, ClipSelectionSettings, ClipSuggestion, TimeRange
from cortaflow.domain.editing import AudioSettings, ReframeSettings, SubtitleStyle
from cortaflow.domain.media import MediaMetadata
from cortaflow.domain.project import ExportSettings, ReframeKeyframe
from cortaflow.domain.subtitle import SubtitleCue, Transcript, TranscriptWord
from cortaflow.domain.tracking import (
    AudioEvidence,
    CropFrame,
    FaceBox,
    FaceTrackPoint,
    FramingValidation,
    SpeakerKeyframe,
)
from cortaflow.services import automatic_pipeline
from cortaflow.services.automatic_pipeline import create_automatic_cuts


def _transcript() -> Transcript:
    word = TranscriptWord(text="Ideia.", start_ms=0, end_ms=6_000)
    return Transcript(
        language="pt",
        words=[word],
        cues=[SubtitleCue(start_ms=0, end_ms=6_000, text="Ideia.")],
    )


def _suggestion(index: int) -> ClipSuggestion:
    return ClipSuggestion(
        start_ms=index * 10_000,
        end_ms=index * 10_000 + 6_000,
        title=f"Corte {index}",
        transcript_excerpt="Ideia.",
        quality_score=0.9 - index / 10,
        reason="Ideia completa.",
    )


def test_automatic_pipeline_runs_ordered_local_stages_and_previews(monkeypatch, tmp_path: Path) -> None:
    media = tmp_path / "mídia.mp4"
    media.write_bytes(b"fixture")
    model = tmp_path / "face.task"
    model.write_bytes(b"fixture")
    output_directory = tmp_path / "previews"
    output_directory.mkdir()
    events: list[str] = []

    class FakeTranscriber:
        def transcribe(self, path, progress, cancelled):
            events.append("transcribe")
            return _transcript()

    monkeypatch.setattr(
        automatic_pipeline,
        "probe_media",
        lambda path: MediaMetadata(
            source=str(path), title="Teste", duration_seconds=30, width=1920,
            height=1080, fps=30, local_path=path,
        ),
    )

    scenes = [TimeRange(start_ms=0, end_ms=30_000)]
    tracks = [
        FaceTrackPoint(
            track_id=1,
            timestamp_ms=0,
            box=FaceBox(x=0.1, y=0.2, width=0.2, height=0.3),
        ),
        FaceTrackPoint(
            track_id=2,
            timestamp_ms=0,
            box=FaceBox(x=0.65, y=0.2, width=0.2, height=0.3),
        ),
    ]
    face_crop = ReframeKeyframe(
        timestamp_ms=0,
        crop=CropFrame(x=0, y=0, width=608, height=1080),
    )
    speaker_crop = ReframeKeyframe(
        timestamp_ms=0,
        crop=CropFrame(x=100, y=0, width=608, height=1080),
    )
    evidence = [AudioEvidence(timestamp_ms=0, energy=0.8, voice_active=True)]

    monkeypatch.setattr(
        automatic_pipeline,
        "detect_scenes",
        lambda *args, **kwargs: (events.append("scenes") or scenes),
    )
    monkeypatch.setattr(
        automatic_pipeline,
        "detect_silences",
        lambda *args, **kwargs: (events.append("silences") or []),
    )
    monkeypatch.setattr(
        automatic_pipeline,
        "extract_audio_evidence",
        lambda *args, **kwargs: (events.append("audio") or evidence),
    )
    monkeypatch.setattr(
        automatic_pipeline,
        "analyze_faces",
        lambda *args, **kwargs: (events.append("faces") or tracks, [face_crop]),
    )
    monkeypatch.setattr(
        automatic_pipeline,
        "analyze_active_speaker",
        lambda *args, **kwargs: (
            events.append("speaker")
            or [SpeakerKeyframe(timestamp_ms=0, track_id=2, confidence=0.9, uncertain=False)],
            [speaker_crop],
        ),
    )

    def fake_suggest(*args, **kwargs):
        events.append("suggest")
        assert kwargs["face_tracks"] == tracks
        assert kwargs["audio_evidence"] == evidence
        return [_suggestion(0), _suggestion(1), _suggestion(2)]

    monkeypatch.setattr(automatic_pipeline, "suggest_clips", fake_suggest)

    def fake_render(source, destination, *args, **kwargs):
        events.append("preview")
        destination.write_bytes(b"preview")
        return destination

    monkeypatch.setattr(automatic_pipeline, "render_project_export", fake_render)
    monkeypatch.setattr(
        automatic_pipeline,
        "validate_rendered_preview",
        lambda *args, expected_face, **kwargs: FramingValidation(
            status="validated" if expected_face else "no_face",
            score=1,
            face_samples=1 if expected_face else 0,
            safe_samples=1 if expected_face else 0,
            unsafe_samples=0,
            max_visible_faces=1 if expected_face else 0,
            speaker_changes=0,
            uncertain_speaker_samples=0,
            message="MP4 conferido.",
        ),
    )
    updates: list[dict] = []
    result = create_automatic_cuts(
        media,
        None,
        ClipSelectionSettings(max_results=3),
        None,
        FakeTranscriber(),
        model,
        output_directory,
        ExportSettings(),
        SubtitleStyle(),
        ReframeSettings(),
        AudioSettings(),
        2,
        updates.append,
    )
    assert events == [
        "transcribe", "scenes", "silences", "audio", "faces", "speaker", "suggest",
        "preview", "preview",
    ]
    assert len(result.previews) == 2
    assert result.previews[0].subtitles_applied
    assert not result.previews[1].subtitles_applied
    assert all(item.reframe_applied for item in result.previews)
    assert len(result.speaker_keyframes) == 1
    assert result.reframe_keyframes[0].crop.x > speaker_crop.crop.x
    assert result.reframe_keyframes[0].face_safe
    assert result.analysis.suggestions[0].framing_status == "validated"
    assert result.analysis.suggestions[0].framing_score == 1
    assert result.analysis.suggestions[0].visible_faces == 2
    assert result.analysis.suggestions[1].framing_status == "no_face"
    assert all(item.path.is_file() for item in result.previews)
    assert result.preview_directory.parent == (tmp_path / "previews").resolve()
    assert result.preview_directory.name == "Cortes automáticos - mídia"
    assert all(item.path.parent == result.preview_directory for item in result.previews)
    assert updates[0]["status"] == "media"
    assert updates[-1]["status"] == "complete"
    ready_index = next(index for index, item in enumerate(updates) if item["status"] == "suggestions_ready")
    preview_index = next(index for index, item in enumerate(updates) if item["status"] == "preview")
    assert ready_index < preview_index
    assert len(updates[ready_index]["suggestions"]) == 3
    preview_updates = [item for item in updates if item["status"] == "preview_ready"]
    assert len(preview_updates) == 2
    assert len(preview_updates[-1]["previews"]) == 2


def test_automatic_pipeline_reserves_a_new_folder_without_overwriting(tmp_path: Path) -> None:
    first = automatic_pipeline._create_run_directory(tmp_path, "Meu vídeo")
    second = automatic_pipeline._create_run_directory(tmp_path, "Meu vídeo")
    assert first.name == "Cortes automáticos - Meu vídeo"
    assert second.name == "Cortes automáticos - Meu vídeo (2)"


def test_automatic_pipeline_never_downloads_when_transcriber_was_not_authorized(
    monkeypatch, tmp_path: Path
) -> None:
    media = tmp_path / "mídia.mp4"
    media.write_bytes(b"fixture")
    output_directory = tmp_path / "previews"
    output_directory.mkdir()
    monkeypatch.setattr(
        automatic_pipeline,
        "probe_media",
        lambda path: MediaMetadata(source=str(path), title="Teste", duration_seconds=10, local_path=path),
    )
    with pytest.raises(RuntimeError, match="nenhum download foi autorizado"):
        create_automatic_cuts(
            media,
            None,
            ClipSelectionSettings(),
            None,
            None,
            None,
            output_directory,
            ExportSettings(),
            SubtitleStyle(),
            ReframeSettings(),
            AudioSettings(),
            0,
        )
