"""Media metadata domain models."""

from pathlib import Path
from pydantic import BaseModel, Field


class MediaFormat(BaseModel):
    format_id: str
    selector: str
    label: str
    width: int | None = None
    height: int | None = None
    extension: str | None = None
    fps: float | None = None
    has_audio: bool = False


class MediaMetadata(BaseModel):
    source: str
    title: str
    duration_seconds: float = Field(ge=0)
    width: int | None = None
    height: int | None = None
    fps: float | None = Field(default=None, gt=0)
    platform: str = "Arquivo local"
    thumbnail_url: str | None = None
    formats: list[MediaFormat] = Field(default_factory=list)
    local_path: Path | None = None
