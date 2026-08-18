"""Transcription and subtitle models."""

from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    probability: float | None = None


class SubtitleCue(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    manually_edited: bool = False


class Transcript(BaseModel):
    language: str
    language_probability: float | None = None
    words: list[TranscriptWord] = Field(default_factory=list)
    cues: list[SubtitleCue] = Field(default_factory=list)

