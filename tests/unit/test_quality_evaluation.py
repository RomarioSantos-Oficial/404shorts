import pytest

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.clip import ClipRange
from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.services.quality_evaluation import evaluate_clip_quality


def suggestion(start_ms: int, end_ms: int, text: str) -> ClipSuggestion:
    return ClipSuggestion(
        start_ms=start_ms,
        end_ms=end_ms,
        title=text[:20],
        transcript_excerpt=text,
        quality_score=0.8,
        reason="Referência controlada.",
    )


def test_quality_metrics_compare_independent_human_intervals() -> None:
    words = [
        TranscriptWord(
            text=f"tema{index}{'.' if index in (9, 19) else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(20)
    ]
    transcript = Transcript(language="pt", words=words)
    metrics = evaluate_clip_quality(
        [
            suggestion(0, 5_000, "primeiro assunto útil completo"),
            suggestion(5_000, 10_000, "segundo tópico diferente completo"),
            suggestion(2_000, 7_000, "trecho extra sem referência"),
        ],
        [ClipRange(start_ms=0, end_ms=5_000), ClipRange(start_ms=5_000, end_ms=10_000)],
        transcript,
    )
    assert metrics.matched_count == 2
    assert metrics.precision == pytest.approx(2 / 3, abs=0.0001)
    assert metrics.recall == 1
    assert metrics.diversity > 0.7
    assert metrics.cut_speech_rate > 0
    assert metrics.subtitle_legibility > 0


def test_empty_no_speech_case_is_a_correct_result() -> None:
    metrics = evaluate_clip_quality([], [], Transcript(language="pt"))
    assert metrics.precision == 1
    assert metrics.recall == 1
    assert metrics.cut_speech_rate == 0
    assert metrics.subtitle_legibility == 1


def test_invalid_iou_is_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate_clip_quality([], [], Transcript(language="pt"), minimum_iou=0)
