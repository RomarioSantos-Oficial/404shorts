from pathlib import Path

from cortaflow.domain.subtitle import SubtitleCue, TranscriptWord
from cortaflow.services.long_discourse import resolve_long_discourse
from cortaflow.services.subtitles import export_vtt, group_words


def _words() -> list[TranscriptWord]:
    words: list[TranscriptWord] = []
    for index in range(24):
        start = index * 10_000
        text = f"Ideia {index}." if index % 3 == 2 else f"palavra{index}"
        words.append(
            TranscriptWord(
                text=text,
                start_ms=start,
                end_ms=start + 1_000,
                probability=0.99,
            )
        )
    return words


def test_long_discourse_is_resolved_into_physical_safe_subideas() -> None:
    resolved = resolve_long_discourse(_words(), maximum_ms=60_000)

    assert len(resolved) >= 3
    assert all(item.duration_ms <= 60_000 for item in resolved)
    assert all(item.duration_ms >= 5_000 for item in resolved)
    assert all(_words()[item.end_index].text.endswith(".") for item in resolved)


def test_subtitle_presets_reduce_reading_density() -> None:
    words = [
        TranscriptWord(text=f"palavra{index}", start_ms=index * 500, end_ms=index * 500 + 400)
        for index in range(10)
    ]

    clean = group_words(words, preset="clean")
    viral = group_words(words, preset="viral")

    assert len(viral) >= len(clean)
    assert all(len(cue.text.split()) <= 4 for cue in viral)


def test_export_vtt_writes_webvtt_timestamps(tmp_path: Path) -> None:
    destination = tmp_path / "captions.vtt"
    export_vtt(
        [SubtitleCue(start_ms=1_250, end_ms=3_500, text="Uma legenda clara com informação suficiente para quebrar em duas linhas.")],
        destination,
    )

    content = destination.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n")
    assert "00:00:01.250 --> 00:00:03.500" in content
    assert "Uma legenda clara com informação" in content
    assert "suficiente para quebrar em duas linhas." in content
    assert (chr(92) + "N") not in content
