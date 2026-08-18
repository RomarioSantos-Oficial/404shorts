"""Clip ranges and timestamp helpers."""

from pydantic import BaseModel, Field, model_validator


class ClipRange(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "ClipRange":
        if self.end_ms <= self.start_ms:
            raise ValueError("O fim do corte deve ser posterior ao início.")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def format_timestamp(milliseconds: int, include_millis: bool = False) -> str:
    """Format non-negative milliseconds for UI or FFmpeg."""
    milliseconds = max(0, milliseconds)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    base = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{base}.{millis:03d}" if include_millis else base

