"""Persistent editor timeline and property settings."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TimelineClip(BaseModel):
    clip_id: str = Field(min_length=1)
    track: Literal["video", "audio"]
    source_start_ms: int = Field(ge=0)
    source_end_ms: int = Field(gt=0)
    timeline_start_ms: int = Field(ge=0)
    label: str = "Clipe"
    transition_ms: int = Field(default=0, ge=0, le=5000)

    @model_validator(mode="after")
    def validate_range(self) -> "TimelineClip":
        if self.source_end_ms <= self.source_start_ms:
            raise ValueError("O fim do clipe deve ser posterior ao início.")
        if self.transition_ms > self.duration_ms:
            raise ValueError("A transição não pode exceder a duração do clipe.")
        return self

    @property
    def duration_ms(self) -> int:
        return self.source_end_ms - self.source_start_ms

    @property
    def timeline_end_ms(self) -> int:
        return self.timeline_start_ms + self.duration_ms


class SubtitleStyle(BaseModel):
    font_name: str = "Arial"
    font_size: int = Field(default=62, ge=12, le=160)
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFD54F"
    outline_color: str = "#000000"
    outline_width: int = Field(default=4, ge=0, le=12)
    shadow: int = Field(default=2, ge=0, le=12)
    background: bool = False
    position: Literal["top", "center", "bottom"] = "bottom"
    max_words: int = Field(default=7, ge=2, le=7)
    preset: Literal["clean", "dynamic", "viral"] = "dynamic"
    animated: bool = True


class AudioSettings(BaseModel):
    volume: float = Field(default=1.0, ge=0, le=2)
    normalize: bool = False


class ReframeSettings(BaseModel):
    aspect_ratio: Literal["9:16", "1:1", "4:5", "original"] = "9:16"
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)
    zoom: float = Field(default=1.0, ge=0.25, le=4)
    smoothing: float = Field(default=0.2, ge=0, le=1)
    max_speed_px: int = Field(default=80, ge=1, le=1000)
    automatic: bool = True
