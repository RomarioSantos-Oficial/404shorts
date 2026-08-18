from cortaflow.domain.subtitle import TranscriptWord
from cortaflow.services.topic_segmentation import segment_topics


def _words(tokens: list[str], start_index: int = 0) -> list[TranscriptWord]:
    return [
        TranscriptWord(
            text=token,
            start_ms=(start_index + index) * 700,
            end_ms=(start_index + index) * 700 + 200,
        )
        for index, token in enumerate(tokens)
    ]


def test_segment_topics_separates_distinct_blocks_after_pause() -> None:
    first = ["corinthians", "memphis", "depayed", "contrato", "torcida", "presidente"] * 5
    second = ["palmeiras", "leila", "stjd", "gestao", "torcedor", "comunicacao"] * 5
    words = _words(first) + _words(second, len(first) + 1)
    words[len(first) - 1] = words[len(first) - 1].model_copy(
        update={"end_ms": words[len(first) - 1].end_ms - 500}
    )

    segments = segment_topics(words, minimum_ms=5_000, maximum_ms=60_000)

    assert len(segments) >= 2
    assert "corinthians" in segments[0].keywords
    assert "palmeiras" in segments[-1].keywords


def test_segment_topics_returns_one_segment_for_empty_or_uniform_input() -> None:
    words = _words(["analise"] * 20)
    segments = segment_topics(words, minimum_ms=5_000, maximum_ms=60_000)

    assert len(segments) == 1
    assert segments[0].start_index == 0
    assert segments[0].end_index == len(words) - 1
