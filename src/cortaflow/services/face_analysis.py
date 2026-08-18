"""Local video pipeline for anonymous face tracking and 9:16 reframing."""

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import cv2

from cortaflow.domain.analysis import TimeRange
from cortaflow.domain.project import ReframeKeyframe
from cortaflow.domain.tracking import FaceBox, FaceObservation, FaceTrackPoint
from cortaflow.services.auto_reframe import generate_reframe_keyframes
from cortaflow.services.face_detection import MediaPipeFaceLandmarker
from cortaflow.services.face_tracking import FaceTracker


class FaceAnalysisCancelled(RuntimeError):
    """Raised when the user cancels face analysis."""


class FaceDetectorProtocol(Protocol):
    def detect(self, rgb_frame: Any, timestamp_ms: int) -> list[FaceBox | FaceObservation]: ...
    def close(self) -> None: ...


def analyze_faces(
    media_path: Path,
    model_path: Path,
    scenes: list[TimeRange] | None = None,
    selected_track_id: int | None = None,
    sample_interval_ms: int = 200,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Event | None = None,
    detector_factory: Callable[[Path], FaceDetectorProtocol] = MediaPipeFaceLandmarker,
) -> tuple[list[FaceTrackPoint], list[ReframeKeyframe]]:
    """Sample video frames, assign temporary IDs and generate smoothed crops."""
    media_path = media_path.resolve()
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    if sample_interval_ms <= 0:
        raise ValueError("O intervalo de detecção deve ser positivo.")
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise RuntimeError("Não foi possível abrir o vídeo para analisar rostos.")
    source_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = max(0, round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    duration_ms = round(total_frames / fps * 1000) if fps > 0 else 0
    scene_boundaries = sorted(scene.start_ms for scene in (scenes or []) if scene.start_ms > 0)
    next_scene_index = 0
    tracker = FaceTracker(max_gap_ms=max(1200, sample_interval_ms * 4))
    detector = detector_factory(model_path)
    tracks: list[FaceTrackPoint] = []
    next_sample_ms = 0
    frame_index = 0
    try:
        while True:
            if cancelled and cancelled.is_set():
                raise FaceAnalysisCancelled("Análise de rostos cancelada.")
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = round(frame_index / fps * 1000) if fps > 0 else next_sample_ms
            frame_index += 1
            if timestamp_ms < next_sample_ms:
                continue
            while next_scene_index < len(scene_boundaries) and scene_boundaries[next_scene_index] <= timestamp_ms:
                tracker.reset()
                next_scene_index += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            observations = detector.detect(rgb, timestamp_ms)
            points = tracker.update(timestamp_ms, observations)
            tracks.extend(points)
            next_sample_ms = timestamp_ms + sample_interval_ms
            if progress:
                progress(
                    {
                        "status": "faces",
                        "position_ms": timestamp_ms,
                        "duration_ms": duration_ms,
                        "faces": len(points),
                    }
                )
    finally:
        capture.release()
        detector.close()
    keyframes = generate_reframe_keyframes(
        tracks,
        source_width,
        source_height,
        selected_track_id=selected_track_id,
        scene_boundaries_ms=scene_boundaries,
    )
    return tracks, keyframes
