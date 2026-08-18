"""Audit the face framing that was actually written to a rendered preview."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import Event

from cortaflow.domain.tracking import FaceTrackPoint, FramingValidation
from cortaflow.services.face_analysis import analyze_faces


def validate_rendered_preview(
    preview_path: Path,
    model_path: Path,
    *,
    expected_face: bool,
    sample_interval_ms: int = 400,
    cancelled: Event | None = None,
) -> FramingValidation:
    """Detect faces in the finished MP4 instead of trusting pre-render coordinates."""
    tracks, _ = analyze_faces(
        preview_path,
        model_path,
        [],
        sample_interval_ms=sample_interval_ms,
        cancelled=cancelled,
    )
    return validate_rendered_face_tracks(tracks, expected_face=expected_face)


def validate_rendered_face_tracks(
    tracks: list[FaceTrackPoint],
    *,
    expected_face: bool,
) -> FramingValidation:
    """Require the largest visible face to be complete and near the horizontal center."""
    grouped: dict[int, list[FaceTrackPoint]] = defaultdict(list)
    for point in tracks:
        grouped[point.timestamp_ms].append(point)
    if not grouped:
        if expected_face:
            return FramingValidation(
                status="needs_review",
                score=0,
                face_samples=0,
                safe_samples=0,
                unsafe_samples=0,
                max_visible_faces=0,
                speaker_changes=0,
                uncertain_speaker_samples=0,
                message=(
                    "O trecho original possui rosto, mas nenhum rosto pôde ser confirmado "
                    "no MP4 renderizado."
                ),
            )
        return FramingValidation(
            status="no_face",
            score=1,
            face_samples=0,
            safe_samples=0,
            unsafe_samples=0,
            max_visible_faces=0,
            speaker_changes=0,
            uncertain_speaker_samples=0,
            message="Nenhum rosto era esperado nem foi detectado no MP4 renderizado.",
        )

    safe_samples = 0
    unsafe_samples = 0
    changes = 0
    previous_track: int | None = None
    for faces in grouped.values():
        focus = max(faces, key=lambda item: item.box.width * item.box.height)
        if previous_track is not None and focus.track_id != previous_track:
            changes += 1
        previous_track = focus.track_id
        if _rendered_face_is_safe(focus):
            safe_samples += 1
        else:
            unsafe_samples += 1
    total = safe_samples + unsafe_samples
    score = safe_samples / total if total else 0
    status = "validated" if unsafe_samples == 0 else "needs_review"
    message = (
        f"Rosto conferido no MP4 pronto em {total} amostras."
        if status == "validated"
        else (
            f"O MP4 pronto cortou ou descentralizou o rosto principal em "
            f"{unsafe_samples} de {total} amostras."
        )
    )
    return FramingValidation(
        status=status,
        score=score,
        face_samples=total,
        safe_samples=safe_samples,
        unsafe_samples=unsafe_samples,
        max_visible_faces=max(len(faces) for faces in grouped.values()),
        speaker_changes=changes,
        uncertain_speaker_samples=0,
        message=message,
    )


def _rendered_face_is_safe(point: FaceTrackPoint) -> bool:
    box = point.box
    center_x = box.x + box.width / 2
    return (
        box.x >= 0.04
        and box.x + box.width <= 0.96
        and box.y >= 0.01
        and box.y + box.height <= 0.99
        and 0.24 <= center_x <= 0.76
    )
