"""Validate and repair automatic face framing before publishing suggestions."""

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.editing import ReframeSettings
from cortaflow.domain.project import ReframeKeyframe, resolve_reframe_at
from cortaflow.domain.tracking import (
    FaceTrackPoint,
    FramingValidation,
    SpeakerKeyframe,
)
from cortaflow.services.auto_reframe import keep_face_in_safe_area


def ensure_safe_reframe(
    tracks: list[FaceTrackPoint],
    speakers: list[SpeakerKeyframe],
    keyframes: list[ReframeKeyframe],
    source_width: int,
    source_height: int,
) -> list[ReframeKeyframe]:
    """Nudge sampled crops so the selected/likely speaker's full face stays safe."""
    if not tracks or not keyframes or min(source_width, source_height) <= 0:
        return list(keyframes)
    grouped = _group_tracks(tracks)
    decisions = {item.timestamp_ms: item for item in speakers}
    by_timestamp = {item.timestamp_ms: item for item in keyframes}
    for timestamp, faces in grouped.items():
        focus = _focus_face(faces, decisions.get(timestamp))
        keyframe = by_timestamp.get(timestamp)
        if not focus or not keyframe:
            continue
        corrected = keep_face_in_safe_area(
            keyframe.crop,
            focus.box,
            source_width,
            source_height,
        )
        by_timestamp[timestamp] = keyframe.model_copy(
            update={"crop": corrected, "face_safe": True}
        )
    return [by_timestamp[timestamp] for timestamp in sorted(by_timestamp)]


def validate_suggestion_reframe(
    suggestion: ClipSuggestion,
    tracks: list[FaceTrackPoint],
    speakers: list[SpeakerKeyframe],
    keyframes: list[ReframeKeyframe],
    settings: ReframeSettings,
    source_width: int,
    source_height: int,
) -> FramingValidation:
    """Check the chosen face at every sample, including every detected face switch."""
    relevant = [
        point
        for point in tracks
        if suggestion.start_ms <= point.timestamp_ms < suggestion.end_ms
    ]
    if not relevant:
        return FramingValidation(
            status="no_face",
            score=1,
            face_samples=0,
            safe_samples=0,
            unsafe_samples=0,
            max_visible_faces=0,
            speaker_changes=0,
            uncertain_speaker_samples=0,
            message="Nenhum rosto detectado; será usado o enquadramento central.",
        )
    if not keyframes or min(source_width, source_height) <= 0:
        timestamps = {item.timestamp_ms for item in relevant}
        return FramingValidation(
            status="needs_review",
            score=0,
            face_samples=len(timestamps),
            safe_samples=0,
            unsafe_samples=len(timestamps),
            max_visible_faces=max(len(items) for items in _group_tracks(relevant).values()),
            speaker_changes=0,
            uncertain_speaker_samples=0,
            message="Há rostos, mas o reenquadramento automático não está disponível.",
        )

    from cortaflow.services.auto_reframe import apply_reframe_settings

    effective = apply_reframe_settings(keyframes, settings, source_width, source_height)
    grouped = _group_tracks(relevant)
    decisions = {item.timestamp_ms: item for item in speakers}
    safe_samples = 0
    unsafe_samples = 0
    uncertain_samples = 0
    changes = 0
    previous_focus: int | None = None
    for timestamp, faces in grouped.items():
        decision = decisions.get(timestamp)
        focus = _focus_face(faces, decision)
        if not focus:
            continue
        if decision and decision.uncertain:
            uncertain_samples += 1
        if previous_focus is not None and focus.track_id != previous_focus:
            changes += 1
        previous_focus = focus.track_id
        crop = resolve_reframe_at(effective, timestamp)
        if crop and _face_is_safe(focus, crop, source_width, source_height):
            safe_samples += 1
        else:
            unsafe_samples += 1

    total = safe_samples + unsafe_samples
    score = safe_samples / total if total else 0
    status = "validated" if unsafe_samples == 0 else "needs_review"
    if status == "validated":
        message = (
            f"Enquadramento validado em {total} amostras; "
            f"até {max(len(items) for items in grouped.values())} rosto(s) por quadro."
        )
    else:
        message = (
            f"Revisão necessária: {unsafe_samples} de {total} amostras deixaram "
            "o rosto principal fora da área segura."
        )
    return FramingValidation(
        status=status,
        score=score,
        face_samples=total,
        safe_samples=safe_samples,
        unsafe_samples=unsafe_samples,
        max_visible_faces=max(len(items) for items in grouped.values()),
        speaker_changes=changes,
        uncertain_speaker_samples=uncertain_samples,
        message=message,
    )


def _group_tracks(tracks: list[FaceTrackPoint]) -> dict[int, list[FaceTrackPoint]]:
    grouped: dict[int, list[FaceTrackPoint]] = {}
    for point in tracks:
        grouped.setdefault(point.timestamp_ms, []).append(point)
    return grouped


def _focus_face(
    faces: list[FaceTrackPoint],
    decision: SpeakerKeyframe | None,
) -> FaceTrackPoint | None:
    if not faces:
        return None
    if decision and decision.track_id is not None:
        selected = next((item for item in faces if item.track_id == decision.track_id), None)
        if selected:
            return selected
    return max(faces, key=lambda item: item.box.width * item.box.height)


def _face_is_safe(point, crop, source_width: int, source_height: int) -> bool:
    face_left = point.box.x * source_width
    face_right = (point.box.x + point.box.width) * source_width
    face_top = point.box.y * source_height
    face_bottom = (point.box.y + point.box.height) * source_height
    horizontal_margin = crop.width * .12
    vertical_margin = crop.height * .04
    return (
        crop.x + horizontal_margin <= face_left
        and face_right <= crop.x + crop.width - horizontal_margin
        and crop.y + vertical_margin <= face_top
        and face_bottom <= crop.y + crop.height - vertical_margin
    )
