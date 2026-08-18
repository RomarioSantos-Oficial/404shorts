"""Orchestration for scene, silence and clip analysis."""

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

from cortaflow.domain.analysis import AnalysisResult, ClipSelectionSettings
from cortaflow.domain.subtitle import Transcript
from cortaflow.domain.tracking import FaceTrackPoint
from cortaflow.services.audio_analysis import extract_audio_evidence
from cortaflow.services.clip_scoring import ClipRanker, suggest_clips
from cortaflow.services.scene_detection import AnalysisCancelled, detect_scenes, detect_silences


def analyze_media(
    media_path: Path,
    transcript: Transcript,
    total_duration_ms: int,
    selection_settings: ClipSelectionSettings | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Event | None = None,
    *,
    face_tracks: list[FaceTrackPoint] | None = None,
    ranker: ClipRanker | None = None,
) -> AnalysisResult:
    """Run the Phase 5 local analyzers and return one validated result."""
    if cancelled and cancelled.is_set():
        raise AnalysisCancelled("Análise cancelada.")
    scenes = detect_scenes(media_path, progress=progress, cancelled=cancelled)
    if cancelled and cancelled.is_set():
        raise AnalysisCancelled("Análise cancelada.")
    silences = detect_silences(media_path, progress=progress, cancelled=cancelled)
    if cancelled and cancelled.is_set():
        raise AnalysisCancelled("Análise cancelada.")
    if progress:
        progress({"status": "audio", "message": "Medindo energia e atividade de voz…"})
    audio_evidence = extract_audio_evidence(
        media_path,
        transcript,
        silences,
        progress=progress,
        cancelled=cancelled,
    )
    if cancelled and cancelled.is_set():
        raise AnalysisCancelled("Análise cancelada.")
    if progress:
        progress({"status": "suggestions", "message": "Pontuando trechos…"})
    suggestions = suggest_clips(
        transcript,
        total_duration_ms,
        silences=silences,
        scenes=scenes,
        settings=selection_settings,
        audio_evidence=audio_evidence,
        face_tracks=face_tracks,
        ranker=ranker,
        progress=progress,
        cancelled=cancelled,
    )
    return AnalysisResult(scenes=scenes, silences=silences, suggestions=suggestions)
