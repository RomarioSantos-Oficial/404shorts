"""Versioned, portable project document."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from cortaflow.domain.analysis import ClipSelectionSettings, ClipSuggestion, TimeRange
from cortaflow.domain.clip import ClipRange
from cortaflow.domain.editing import (
    AudioSettings,
    LayerItem,
    ReframeSettings,
    SequenceDocument,
    SubtitleStyle,
    TimelineClip,
)
from cortaflow.domain.subtitle import Transcript
from cortaflow.domain.tracking import CropFrame, FaceTrackPoint, SpeakerKeyframe, SpeakerOverride


class ReframeKeyframe(BaseModel):
    timestamp_ms: int = Field(ge=0)
    crop: CropFrame
    manual: bool = False
    scene_reset: bool = False
    face_safe: bool = True


class WatermarkSettings(BaseModel):
    enabled: bool = False
    image_path: Path | None = None
    position: Literal[
        "top-left", "top", "top-right", "left", "center", "right",
        "bottom-left", "bottom", "bottom-right", "custom",
    ] = "bottom-right"
    width_percent: float = Field(default=18, ge=2, le=80)
    opacity: float = Field(default=0.75, ge=0.05, le=1)
    margin_percent: float = Field(default=3, ge=0, le=25)
    custom_x_percent: float = Field(default=50, ge=0, le=100)
    custom_y_percent: float = Field(default=50, ge=0, le=100)


class ExportSettings(BaseModel):
    width: int = 1080
    height: int = 1920
    fps: float = 30
    codec: str = "h264"
    quality: int = Field(default=20, ge=0, le=51)
    use_nvenc: bool = True
    normalize_audio: bool = False
    watermark: WatermarkSettings = Field(default_factory=WatermarkSettings)


class ProjectDocument(BaseModel):
    format_version: int = 1
    name: str = "Projeto sem título"
    source_path: Path | None = None
    source_metadata: dict = Field(default_factory=dict)
    clips: list[ClipRange] = Field(default_factory=list)
    timeline_clips: list[TimelineClip] = Field(default_factory=list)
    layers: list[LayerItem] = Field(default_factory=list)
    sequences: list[SequenceDocument] = Field(default_factory=list)
    active_sequence_id: str | None = None
    transcript: Transcript | None = None
    scenes: list[TimeRange] = Field(default_factory=list)
    silences: list[TimeRange] = Field(default_factory=list)
    suggestions: list[ClipSuggestion] = Field(default_factory=list)
    clip_selection: ClipSelectionSettings = Field(default_factory=ClipSelectionSettings)
    face_tracks: list[FaceTrackPoint] = Field(default_factory=list)
    selected_face_track_id: int | None = Field(default=None, gt=0)
    speaker_keyframes: list[SpeakerKeyframe] = Field(default_factory=list)
    speaker_overrides: list[SpeakerOverride] = Field(default_factory=list)
    reframe_keyframes: list[ReframeKeyframe] = Field(default_factory=list)
    reframe_settings: ReframeSettings = Field(default_factory=ReframeSettings)
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
    audio_settings: AudioSettings = Field(default_factory=AudioSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


def resolve_reframe_at(keyframes: list[ReframeKeyframe], timestamp_ms: int) -> CropFrame | None:
    """Choose the latest keyframe; manual frames win at equal timestamps."""
    eligible = [item for item in keyframes if item.timestamp_ms <= timestamp_ms]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item.timestamp_ms, item.manual)).crop
