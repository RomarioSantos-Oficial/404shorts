"""Analysis results and suggested clip models."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TimeRange(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TimeRange":
        if self.end_ms <= self.start_ms:
            raise ValueError("O fim deve ser posterior ao início.")
        return self


class ClipSelectionSettings(BaseModel):
    min_seconds: int = Field(default=5, ge=5, le=179)
    max_seconds: int = Field(default=179, ge=5, le=179)
    preferred_seconds: int = Field(default=60, ge=5, le=179)
    max_results: int = Field(default=12, ge=1, le=50)
    ranking_mode: Literal["automatic", "heuristic"] = "automatic"
    selection_goal: Literal["balanced", "faithful", "viral", "topic"] = "balanced"
    topic_prompt: str = Field(default="", max_length=240)
    audience: str = Field(default="", max_length=160)
    vocabulary: list[str] = Field(default_factory=list, max_length=60)
    auto_accept_threshold: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_durations(self) -> "ClipSelectionSettings":
        if self.min_seconds > self.max_seconds:
            raise ValueError("A duração mínima não pode superar a máxima.")
        if not self.min_seconds <= self.preferred_seconds <= self.max_seconds:
            raise ValueError("A duração preferida deve ficar entre a mínima e a máxima.")
        if self.selection_goal == "topic" and not self.topic_prompt.strip():
            raise ValueError("Informe o tema desejado para usar a seleção por tema.")
        return self


class ClipSuggestion(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)
    title: str
    transcript_excerpt: str
    quality_score: float = Field(ge=0, le=1)
    reason: str
    score_components: dict[str, float] = Field(default_factory=dict)
    editorial_status: Literal["pending", "validated", "needs_review"] = "pending"
    editorial_score: float | None = Field(default=None, ge=0, le=1)
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    potential_score: float | None = Field(default=None, ge=0, le=1)
    production_quality_score: float | None = Field(default=None, ge=0, le=1)
    trend_status: Literal["not_evaluated", "evaluated"] = "not_evaluated"
    central_claim: str = ""
    payoff: str = ""
    evidence_start: str = ""
    evidence_end: str = ""
    opening_dependency: Literal["none", "repairable", "strong", "unknown"] = "unknown"
    ending_state: Literal["complete", "repairable", "ongoing", "unknown"] = "unknown"
    after_continues_same_answer: bool = False
    repair_history: list[str] = Field(default_factory=list, max_length=12)
    context_before: str = ""
    context_after: str = ""
    resegmented_from_long_unit: bool = False
    selection_goal: Literal["balanced", "faithful", "viral", "topic"] = "balanced"
    topic_prompt: str = ""
    audience: str = ""
    vocabulary: list[str] = Field(default_factory=list, max_length=60)
    status: Literal["pending", "accepted", "rejected"] = "pending"
    framing_status: Literal["pending", "validated", "needs_review", "no_face"] = "pending"
    framing_score: float | None = Field(default=None, ge=0, le=1)
    visible_faces: int = Field(default=0, ge=0)
    speaker_changes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "ClipSuggestion":
        if self.end_ms <= self.start_ms:
            raise ValueError("O fim da sugestão deve ser posterior ao início.")
        return self

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class AnalysisResult(BaseModel):
    scenes: list[TimeRange] = Field(default_factory=list)
    silences: list[TimeRange] = Field(default_factory=list)
    suggestions: list[ClipSuggestion] = Field(default_factory=list)
