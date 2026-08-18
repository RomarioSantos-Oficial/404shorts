"""Anonymous face tracking and reframing data."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FaceBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    confidence: float = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "FaceBox":
        if self.x + self.width > 1.000_001 or self.y + self.height > 1.000_001:
            raise ValueError("A caixa facial deve permanecer dentro do quadro.")
        return self


class FaceObservation(BaseModel):
    box: FaceBox
    mouth_openness: float | None = Field(default=None, ge=0, le=1)


class FaceTrackPoint(BaseModel):
    track_id: int = Field(gt=0)
    timestamp_ms: int = Field(ge=0)
    box: FaceBox
    mouth_openness: float | None = Field(default=None, ge=0, le=1)


class CropFrame(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AudioEvidence(BaseModel):
    timestamp_ms: int = Field(ge=0)
    energy: float = Field(ge=0, le=1)
    voice_active: bool = False


class SpeakerKeyframe(BaseModel):
    timestamp_ms: int = Field(ge=0)
    track_id: int | None = Field(default=None, gt=0)
    confidence: float = Field(default=0, ge=0, le=1)
    uncertain: bool = True
    manual: bool = False


class SpeakerOverride(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    track_id: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SpeakerOverride":
        if self.end_ms <= self.start_ms:
            raise ValueError("O fim da correção deve ser posterior ao início.")
        return self


class FramingValidation(BaseModel):
    """Geometric and speaker evidence collected for one suggested clip."""

    status: Literal["validated", "needs_review", "no_face"]
    score: float = Field(ge=0, le=1)
    face_samples: int = Field(ge=0)
    safe_samples: int = Field(ge=0)
    unsafe_samples: int = Field(ge=0)
    max_visible_faces: int = Field(ge=0)
    speaker_changes: int = Field(ge=0)
    uncertain_speaker_samples: int = Field(ge=0)
    message: str

    @model_validator(mode="after")
    def validate_sample_totals(self) -> "FramingValidation":
        if self.safe_samples + self.unsafe_samples != self.face_samples:
            raise ValueError("As amostras seguras e inseguras devem totalizar as amostras faciais.")
        if self.uncertain_speaker_samples > self.face_samples:
            raise ValueError("A incerteza de falante não pode superar as amostras faciais.")
        return self
