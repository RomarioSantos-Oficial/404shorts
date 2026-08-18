"""Reproducible quality metrics for automatic clip selection.

The evaluator deliberately keeps human reference cuts separate from the ranking
algorithm.  It can therefore be used with controlled media without teaching the
ranker the expected answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import re

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.clip import ClipRange
from cortaflow.domain.subtitle import SubtitleCue, Transcript
from cortaflow.services.subtitles import clip_subtitle_track, group_words


@dataclass(frozen=True)
class QualityMetrics:
    selected_count: int
    reference_count: int
    matched_count: int
    precision: float
    recall: float
    diversity: float
    cut_speech_rate: float
    subtitle_legibility: float


def evaluate_clip_quality(
    suggestions: list[ClipSuggestion],
    human_cuts: list[ClipRange],
    transcript: Transcript,
    *,
    minimum_iou: float = 0.5,
) -> QualityMetrics:
    """Compare automatic choices with independent human-marked intervals.

    A suggestion matches at most one reference cut when their temporal
    intersection-over-union reaches ``minimum_iou``.  Speech cuts are boundaries
    placed inside an idea (without sentence punctuation or a natural pause).
    Subtitle legibility checks cue size, reading speed and the two-line budget.
    """
    if not 0 < minimum_iou <= 1:
        raise ValueError("O IoU mínimo deve ficar entre zero e um.")

    matched = _greedy_matches(suggestions, human_cuts, minimum_iou)
    precision = matched / len(suggestions) if suggestions else float(not human_cuts)
    recall = matched / len(human_cuts) if human_cuts else float(not suggestions)
    return QualityMetrics(
        selected_count=len(suggestions),
        reference_count=len(human_cuts),
        matched_count=matched,
        precision=round(precision, 4),
        recall=round(recall, 4),
        diversity=round(_diversity(suggestions), 4),
        cut_speech_rate=round(_cut_speech_rate(suggestions, transcript), 4),
        subtitle_legibility=round(_subtitle_legibility(suggestions, transcript), 4),
    )


def _greedy_matches(
    suggestions: list[ClipSuggestion],
    human_cuts: list[ClipRange],
    minimum_iou: float,
) -> int:
    possible = sorted(
        (
            (_temporal_iou(suggestion, reference), suggestion_index, reference_index)
            for suggestion_index, suggestion in enumerate(suggestions)
            for reference_index, reference in enumerate(human_cuts)
        ),
        reverse=True,
    )
    used_suggestions: set[int] = set()
    used_references: set[int] = set()
    for overlap, suggestion_index, reference_index in possible:
        if overlap < minimum_iou:
            break
        if suggestion_index in used_suggestions or reference_index in used_references:
            continue
        used_suggestions.add(suggestion_index)
        used_references.add(reference_index)
    return len(used_suggestions)


def _temporal_iou(suggestion: ClipSuggestion, reference: ClipRange) -> float:
    intersection = max(
        0,
        min(suggestion.end_ms, reference.end_ms)
        - max(suggestion.start_ms, reference.start_ms),
    )
    union = max(suggestion.end_ms, reference.end_ms) - min(
        suggestion.start_ms, reference.start_ms
    )
    return intersection / max(1, union)


def _diversity(suggestions: list[ClipSuggestion]) -> float:
    if len(suggestions) < 2:
        return 1.0
    similarities: list[float] = []
    for left, right in combinations(suggestions, 2):
        left_terms = _terms(left.transcript_excerpt)
        right_terms = _terms(right.transcript_excerpt)
        union = left_terms | right_terms
        similarities.append(len(left_terms & right_terms) / len(union) if union else 1.0)
    return max(0.0, 1.0 - sum(similarities) / len(similarities))


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wÀ-ÿ]+", text.lower())
        if len(token) >= 4
    }


def _cut_speech_rate(suggestions: list[ClipSuggestion], transcript: Transcript) -> float:
    if not suggestions or not transcript.words:
        return 0.0
    unsafe = 0
    for suggestion in suggestions:
        start_index = next(
            (
                index
                for index, word in enumerate(transcript.words)
                if word.end_ms > suggestion.start_ms
            ),
            len(transcript.words),
        )
        end_index = next(
            (
                index
                for index, word in enumerate(transcript.words)
                if word.end_ms >= suggestion.end_ms
            ),
            len(transcript.words) - 1,
        )
        if start_index > 0:
            previous = transcript.words[start_index - 1]
            current = transcript.words[start_index]
            natural_start = (
                _ends_sentence(previous.text)
                or current.start_ms - previous.end_ms >= 350
            )
            unsafe += int(not natural_start)
        if 0 <= end_index < len(transcript.words) - 1:
            current = transcript.words[end_index]
            following = transcript.words[end_index + 1]
            natural_end = (
                _ends_sentence(current.text)
                or following.start_ms - current.end_ms >= 350
            )
            unsafe += int(not natural_end)
    return unsafe / (2 * len(suggestions))


def _subtitle_legibility(
    suggestions: list[ClipSuggestion],
    transcript: Transcript,
) -> float:
    if not suggestions:
        return 1.0
    source_cues = transcript.cues or group_words(transcript.words)
    checks: list[bool] = []
    for suggestion in suggestions:
        cues, _ = clip_subtitle_track(
            source_cues,
            transcript.words,
            ClipRange(start_ms=suggestion.start_ms, end_ms=suggestion.end_ms),
        )
        for cue in cues:
            checks.append(_cue_is_legible(cue))
    return sum(checks) / len(checks) if checks else 0.0


def _cue_is_legible(cue: SubtitleCue) -> bool:
    duration_seconds = (cue.end_ms - cue.start_ms) / 1000
    word_count = len(cue.text.replace("\\N", " ").split())
    character_count = len(cue.text.replace("\\N", " ").strip())
    estimated_lines = 1 if character_count <= 28 else 2
    characters_per_second = character_count / max(0.001, duration_seconds)
    return (
        1 <= word_count <= 7
        and estimated_lines <= 2
        and character_count <= 84
        and 0.35 <= duration_seconds <= 7
        and characters_per_second <= 21
    )


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", ":", ";"))
