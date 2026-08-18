"""Deterministic recovery of subideas from discourse units longer than 179 seconds."""

from __future__ import annotations

from dataclasses import dataclass

from cortaflow.domain.subtitle import TranscriptWord


MAX_CLIP_MS = 179_000
MIN_CLIP_MS = 5_000
DISCOURSE_PAUSE_MS = 1_500


@dataclass(frozen=True)
class ResolvedSubidea:
    """A physically safe, sentence-complete interval recovered from a long unit."""

    start_index: int
    end_index: int
    start_ms: int
    end_ms: int
    reason: str

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def resolve_long_discourse(
    words: list[TranscriptWord],
    *,
    minimum_ms: int = MIN_CLIP_MS,
    maximum_ms: int = MAX_CLIP_MS,
) -> list[ResolvedSubidea]:
    """Return safe subideas when a continuous discourse exceeds the clip limit.

    The resolver never cuts in the middle of a word. It first builds complete
    sentence units, then packs adjacent units while the physical 179-second
    limit remains safe. A final short unit is merged into its predecessor when
    possible, preventing isolated fragments from being suggested.
    """
    if not words or maximum_ms <= 0:
        return []
    if words[-1].end_ms - words[0].start_ms <= maximum_ms:
        return []

    units = _sentence_units(words)
    resolved: list[ResolvedSubidea] = []
    start_unit = 0
    while start_unit < len(units):
        start_index, _ = units[start_unit]
        end_unit = start_unit
        last_valid: int | None = None
        while end_unit < len(units):
            _, end_index = units[end_unit]
            duration = words[end_index].end_ms - words[start_index].start_ms
            if duration > maximum_ms:
                break
            last_valid = end_unit
            end_unit += 1
        if last_valid is None:
            # A single sentence can exceed the hard limit. Keep the largest
            # complete-word prefix that still fits, and leave it for review.
            end_index = _largest_safe_end(words, start_index, maximum_ms)
            if end_index < start_index:
                start_unit += 1
                continue
            last_valid = end_unit if end_unit < len(units) else len(units) - 1
            resolved.append(
                ResolvedSubidea(
                    start_index=start_index,
                    end_index=end_index,
                    start_ms=words[start_index].start_ms,
                    end_ms=words[end_index].end_ms,
                    reason="frase longa limitada ao máximo físico de 179 s",
                )
            )
            start_unit += 1
            continue

        end_index = units[last_valid][1]
        duration = words[end_index].end_ms - words[start_index].start_ms
        if duration >= minimum_ms:
            resolved.append(
                ResolvedSubidea(
                    start_index=start_index,
                    end_index=end_index,
                    start_ms=words[start_index].start_ms,
                    end_ms=words[end_index].end_ms,
                    reason="subideia fechada em limite de frase dentro de 179 s",
                )
            )
        next_unit = last_valid + 1
        if next_unit <= start_unit:
            next_unit = start_unit + 1
        start_unit = next_unit

    return _deduplicate(resolved)


def _sentence_units(words: list[TranscriptWord]) -> list[tuple[int, int]]:
    units: list[tuple[int, int]] = []
    start = 0
    for index, word in enumerate(words):
        next_word = words[index + 1] if index + 1 < len(words) else None
        ends_unit = _complete_ending(word.text)
        paused = bool(next_word and next_word.start_ms - word.end_ms >= DISCOURSE_PAUSE_MS)
        if ends_unit or paused or index == len(words) - 1:
            units.append((start, index))
            start = index + 1
    return units


def _largest_safe_end(words: list[TranscriptWord], start_index: int, maximum_ms: int) -> int:
    limit = words[start_index].start_ms + maximum_ms
    safe = start_index - 1
    for index in range(start_index, len(words)):
        if words[index].end_ms > limit:
            break
        safe = index
    return safe


def _complete_ending(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned) and cleaned.endswith((".", "!", "?", "?!", "!?"))


def _deduplicate(items: list[ResolvedSubidea]) -> list[ResolvedSubidea]:
    seen: set[tuple[int, int]] = set()
    unique: list[ResolvedSubidea] = []
    for item in items:
        key = (item.start_ms, item.end_ms)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
