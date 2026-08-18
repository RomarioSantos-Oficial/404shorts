from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from cortaflow.domain.subtitle import TranscriptWord


TOPIC_PAUSE_MS = 650
MIN_TOPIC_MS = 18_000
MAX_TOPIC_MS = 95_000
MIN_TOPIC_WORDS = 12

_STOPWORDS = {
    "a", "ao", "aos", "as", "até", "com", "como", "da", "das", "de", "do", "dos",
    "e", "é", "em", "essa", "esse", "esta", "este", "eu", "foi", "foram", "há",
    "isso", "já", "mais", "mas", "me", "mesmo", "na", "não", "nas", "nem", "no", "nos",
    "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "que", "se", "sem",
    "ser", "só", "sua", "suas", "também", "tem", "têm", "um", "uma", "umas", "uns", "vai",
    "você", "vocês", "aqui", "agora", "então", "quando", "porque", "como", "this", "that",
    "with", "from", "into", "about", "what", "when", "will", "have", "were", "your",
}


@dataclass(frozen=True)
class TopicSegment:
    """A coherent transcript interval used to diversify and explain candidates."""

    start_index: int
    end_index: int
    start_ms: int
    end_ms: int
    keywords: tuple[str, ...]
    change_score: float = 0.0

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def label(self) -> str:
        return ", ".join(self.keywords[:3]) or "tópico geral"


def segment_topics(
    words: list[TranscriptWord],
    *,
    minimum_ms: int = MIN_TOPIC_MS,
    maximum_ms: int = MAX_TOPIC_MS,
) -> list[TopicSegment]:
    """Split a transcript at likely topic changes using only word timestamps.

    This is intentionally a cheap first pass. It uses speech pauses as candidate
    boundaries and lexical cohesion between adjacent chunks, so it works offline
    before an optional semantic model reviews the resulting clips.
    """
    if not words:
        return []
    minimum_ms = max(MIN_TOPIC_MS, minimum_ms)
    maximum_ms = max(minimum_ms, maximum_ms)
    chunks = _make_chunks(words)
    if not chunks:
        return [_segment(words, 0, len(words) - 1, 0.0)]

    boundaries = [0]
    current_start = 0
    current_terms: set[str] = set()
    current_end = 0
    current_start_ms = words[0].start_ms
    for chunk_index, (start, end) in enumerate(chunks):
        chunk_terms = _terms(words[start : end + 1])
        proposed_end_ms = words[end].end_ms
        current_terms |= chunk_terms
        current_end = end
        if chunk_index + 1 >= len(chunks):
            continue
        next_start, _ = chunks[chunk_index + 1]
        next_terms = _terms(words[next_start : chunks[chunk_index + 1][1] + 1])
        gap_ms = words[next_start].start_ms - words[end].end_ms
        duration_ms = proposed_end_ms - current_start_ms
        change_score = _change_score(current_terms, next_terms)
        forced = proposed_end_ms - current_start_ms >= maximum_ms
        changed = (
            duration_ms >= minimum_ms
            and gap_ms >= TOPIC_PAUSE_MS
            and change_score >= 0.62
            and len(current_terms) >= 3
        )
        if forced or changed:
            boundaries.append(next_start)
            current_start = next_start
            current_terms = set()
            current_start_ms = words[next_start].start_ms

    boundaries = sorted(set(boundaries))
    segments: list[TopicSegment] = []
    for index, start in enumerate(boundaries):
        end = (boundaries[index + 1] - 1) if index + 1 < len(boundaries) else len(words) - 1
        if end < start:
            continue
        segments.append(_segment(words, start, end, _boundary_change_score(words, start)))
    return segments or [_segment(words, 0, len(words) - 1, 0.0)]


def _make_chunks(words: list[TranscriptWord]) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    start = 0
    for index in range(len(words) - 1):
        gap_ms = words[index + 1].start_ms - words[index].end_ms
        punctuation = words[index].text.rstrip().endswith((".", "!", "?", ":", ";"))
        if (gap_ms >= TOPIC_PAUSE_MS or punctuation) and index - start + 1 >= MIN_TOPIC_WORDS:
            chunks.append((start, index))
            start = index + 1
    if start < len(words):
        chunks.append((start, len(words) - 1))
    return chunks


def _segment(words: list[TranscriptWord], start: int, end: int, change_score: float) -> TopicSegment:
    terms = Counter(_terms(words[start : end + 1]))
    keywords = tuple(term for term, _ in terms.most_common(6))
    return TopicSegment(
        start_index=start,
        end_index=end,
        start_ms=words[start].start_ms,
        end_ms=words[end].end_ms,
        keywords=keywords,
        change_score=round(change_score, 3),
    )


def _terms(words: list[TranscriptWord]) -> set[str]:
    return {
        token
        for word in words
        for token in re.findall(r"[\wÀ-ÿ]+", word.text.lower())
        if len(token) >= 4 and token not in _STOPWORDS and not token.isdigit()
    }


def _change_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right) / max(1, len(left | right))
    return 1.0 - overlap


def _boundary_change_score(words: list[TranscriptWord], start: int) -> float:
    if start <= 0:
        return 0.0
    before = _terms(words[max(0, start - MIN_TOPIC_WORDS) : start])
    after = _terms(words[start : min(len(words), start + MIN_TOPIC_WORDS)])
    return _change_score(before, after)
