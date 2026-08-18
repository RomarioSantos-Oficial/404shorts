"""Deterministic editorial boundary checks for spoken clip candidates.

These checks are deliberately conservative.  A visual scene cut is not a
speech boundary, and a target duration is never allowed to split a word or an
unfinished utterance.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from cortaflow.domain.analysis import TimeRange
from cortaflow.domain.subtitle import TranscriptWord


MINIMUM_SPEECH_GAP_MS = 80
LONG_DISCOURSE_PAUSE_MS = 1_500
TRANSCRIPT_PAUSE_MS = 450
CONTEXT_WINDOW_MS = 30_000
CONTEXT_WORD_LIMIT = 90

_DEPENDENT_OPENINGS = {
    "agora", "aí", "assim", "claro", "daí", "e", "ela", "elas", "ele", "eles",
    "então", "essa", "essas", "esse", "esses", "isso", "isto", "mas",
    "porque", "pois", "também", "aquilo", "aham", "uhum", "exato",
    "cara", "exatamente", "nossa", "pô", "tipo",
}
_DANGLING_ENDINGS = {
    "a", "aí", "ao", "as", "assim", "com", "como", "da", "das", "de",
    "do", "dos", "e", "ela", "ele", "em", "então", "essa", "esse", "isso",
    "lá", "mas", "na", "nas", "no", "nos", "o", "os", "para", "pela",
    "pelo", "porque", "por", "pra", "que", "quando", "se", "sem", "tipo",
    "um", "uma",
}


@dataclass(frozen=True)
class EditorialBoundaryAssessment:
    """Explain whether a word range is safe and understandable on its own."""

    start_safe: bool
    end_safe: bool
    start_independent: bool
    end_complete: bool
    score: float
    reason: str
    start_gap_ms: int
    end_gap_ms: int

    @property
    def passes(self) -> bool:
        return all(
            (
                self.start_safe,
                self.end_safe,
                self.start_independent,
                self.end_complete,
            )
        )


def assess_word_range(
    words: list[TranscriptWord],
    start_index: int,
    end_index: int,
    silences: list[TimeRange] | None = None,
) -> EditorialBoundaryAssessment:
    """Validate cuts at complete utterances without using visual scene changes."""
    if not 0 <= start_index <= end_index < len(words):
        raise IndexError("Intervalo de palavras inválido.")
    silences = silences or []
    first = words[start_index]
    last = words[end_index]
    previous = words[start_index - 1] if start_index > 0 else None
    following = words[end_index + 1] if end_index + 1 < len(words) else None
    start_gap = max(0, first.start_ms - previous.end_ms) if previous else first.start_ms
    end_gap = max(0, following.start_ms - last.end_ms) if following else MINIMUM_SPEECH_GAP_MS
    starts_after_silence = bool(
        previous and _silence_between(silences, previous.end_ms, first.start_ms)
    )
    ends_before_silence = bool(
        following and _silence_between(silences, last.end_ms, following.start_ms)
    )
    # The automatic pipeline always supplies detected silences.  When this
    # service is called without that audio evidence (for example by an older
    # project or a unit test), a completed sentence is the conservative
    # fallback: the cut still lands exactly between transcript words, never in
    # the middle of one.  Visual scene changes are intentionally not accepted.
    has_audio_boundaries = bool(silences)
    previous_is_complete = bool(previous and _complete_ending(previous.text))
    last_is_complete = _complete_ending(last.text)
    start_safe = (
        previous is None
        or start_gap >= MINIMUM_SPEECH_GAP_MS
        or starts_after_silence
        or (not has_audio_boundaries and previous_is_complete)
    )
    ends_after_transcript_pause = end_gap >= TRANSCRIPT_PAUSE_MS
    end_safe = (
        following is None
        or end_gap >= MINIMUM_SPEECH_GAP_MS
        or ends_before_silence
        or (not has_audio_boundaries and last_is_complete)
    )
    start_independent = _independent_opening(words[start_index : min(end_index + 1, start_index + 12)])
    # Some Whisper outputs contain reliable word timestamps but little or no
    # sentence punctuation. A meaningful pause is still a conservative speech
    # boundary; it should be reviewable, not discarded as an unfinished idea.
    end_complete = last_is_complete or ends_before_silence or ends_after_transcript_pause
    checks = (start_safe, end_safe, start_independent, end_complete)
    score = sum(checks) / len(checks)
    problems: list[str] = []
    if not start_safe:
        problems.append("início sem pausa entre falas")
    if not end_safe:
        problems.append("fim sem pausa entre falas")
    if not start_independent:
        problems.append("início depende do contexto anterior")
    if not end_complete:
        problems.append("fala ou ideia ainda não concluída")
    reason = "limites de fala seguros e ideia fechada" if not problems else "; ".join(problems)
    return EditorialBoundaryAssessment(
        start_safe=start_safe,
        end_safe=end_safe,
        start_independent=start_independent,
        end_complete=end_complete,
        score=score,
        reason=reason,
        start_gap_ms=start_gap,
        end_gap_ms=end_gap,
    )


def context_around(
    words: list[TranscriptWord],
    start_index: int,
    end_index: int,
) -> tuple[str, str]:
    """Return bounded context before/after a candidate for semantic review."""
    start_ms = words[start_index].start_ms
    end_ms = words[end_index].end_ms
    before = [
        word.text
        for word in words[max(0, start_index - CONTEXT_WORD_LIMIT) : start_index]
        if word.end_ms >= start_ms - CONTEXT_WINDOW_MS
    ]
    after = [
        word.text
        for word in words[end_index + 1 : end_index + 1 + CONTEXT_WORD_LIMIT]
        if word.start_ms <= end_ms + CONTEXT_WINDOW_MS
    ]
    return " ".join(before), " ".join(after)


def belongs_to_oversized_discourse(
    words: list[TranscriptWord],
    start_index: int,
    end_index: int,
    maximum_ms: int,
) -> bool:
    """Mark candidates extracted from a continuous discourse longer than the limit.

    The mark triggers a second semantic completeness check.  The returned
    candidate itself always remains within ``maximum_ms``.
    """
    left = start_index
    while left > 0 and not _major_discourse_break(words[left - 1], words[left]):
        left -= 1
    right = end_index
    while right + 1 < len(words) and not _major_discourse_break(words[right], words[right + 1]):
        right += 1
    return words[right].end_ms - words[left].start_ms > maximum_ms


def _major_discourse_break(left: TranscriptWord, right: TranscriptWord) -> bool:
    return (
        right.start_ms - left.end_ms >= LONG_DISCOURSE_PAUSE_MS
        and _complete_ending(left.text)
    )


def _silence_between(silences: list[TimeRange], left_ms: int, right_ms: int) -> bool:
    if right_ms < left_ms:
        return False
    return any(
        silence.end_ms >= left_ms - 120 and silence.start_ms <= right_ms + 120
        for silence in silences
    )


def _independent_opening(words: list[TranscriptWord]) -> bool:
    if not words:
        return False
    tokens = re.findall(r"[\wÀ-ÿ]+", " ".join(word.text for word in words).lower())
    if not tokens:
        return False
    first = tokens[0]
    if first in _DEPENDENT_OPENINGS:
        return False
    if first in {"sim", "não"} and len(tokens) > 1 and tokens[1] in _DEPENDENT_OPENINGS:
        return False
    return True


def _complete_ending(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or cleaned.endswith(("...", "…", ":", ";", ",", "-")):
        return False
    tokens = re.findall(r"[\wÀ-ÿ]+", cleaned.lower())
    if not tokens or tokens[-1] in _DANGLING_ENDINGS:
        return False
    return cleaned.endswith((".", "!", "?", "?!", "!?"))
