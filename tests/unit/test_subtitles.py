from pathlib import Path
from cortaflow.domain.subtitle import SubtitleCue, TranscriptWord
from cortaflow.domain.subtitle import Transcript
from cortaflow.domain.editing import SubtitleStyle
from cortaflow.domain.clip import ClipRange
from cortaflow.services.subtitles import (
    build_burn_subtitles_command,
    export_subtitles,
    clip_subtitle_track,
    group_words,
    load_transcript,
    merge_manual_corrections,
    save_transcript,
)


def words(*values: str) -> list[TranscriptWord]:
    return [TranscriptWord(text=value, start_ms=i * 300, end_ms=(i + 1) * 300) for i, value in enumerate(values)]


def test_groups_at_punctuation_and_avoids_orphan() -> None:
    cues = group_words(words("Esta", "frase", "termina.", "Outra"))
    assert [cue.text for cue in cues] == ["Esta frase termina. Outra"]


def test_limits_words_per_cue() -> None:
    cues = group_words(words("um", "dois", "três", "quatro", "cinco", "seis", "sete", "oito"))
    assert max(len(cue.text.split()) for cue in cues) <= 7


def test_exports_srt_and_animated_ass(tmp_path: Path) -> None:
    cues = [SubtitleCue(start_ms=0, end_ms=1000, text="Olá mundo")]
    srt = export_subtitles(cues, tmp_path / "legenda.srt")
    ass = export_subtitles(cues, tmp_path / "legenda.ass", animated=True)
    assert "Olá mundo" in srt.read_text(encoding="utf-8-sig")
    assert "\\k" in ass.read_text(encoding="utf-8-sig")


def test_manual_correction_survives_regeneration() -> None:
    old = [SubtitleCue(start_ms=0, end_ms=1000, text="Texto corrigido", manually_edited=True)]
    generated = [SubtitleCue(start_ms=100, end_ms=900, text="texto errado")]
    assert merge_manual_corrections(old, generated)[0].text == "Texto corrigido"


def test_transcript_json_round_trip_with_accents(tmp_path: Path) -> None:
    transcript = Transcript(
        language="pt",
        words=words("Olá", "você"),
        cues=[SubtitleCue(start_ms=0, end_ms=600, text="Olá você")],
    )
    path = save_transcript(transcript, tmp_path / "transcrição.json")
    assert load_transcript(path) == transcript
    assert "Olá você" in path.read_text(encoding="utf-8")


def test_burn_command_uses_subtitles_filter_and_optional_audio(tmp_path: Path) -> None:
    source = tmp_path / "vídeo fonte.mp4"
    subtitle = tmp_path / "legenda com acento.ass"
    destination = tmp_path / "saída.mp4"
    command = build_burn_subtitles_command(source, subtitle, destination)
    subtitle_filter = command[command.index("-vf") + 1]
    assert subtitle_filter.startswith("subtitles=filename='")
    assert "legenda com acento.ass" in subtitle_filter
    assert "0:a?" in command
    assert command[command.index("-c:v") + 1] == "libx264"


def test_ass_uses_configured_subtitle_style(tmp_path: Path) -> None:
    path = export_subtitles(
        [SubtitleCue(start_ms=0, end_ms=1000, text="Estilo")],
        tmp_path / "estilo.ass",
        animated=True,
        style=SubtitleStyle(font_name="Verdana", font_size=72, position="top", background=True),
    )
    content = path.read_text(encoding="utf-8-sig")
    assert "Verdana" in content
    assert ",72," in content
    assert "{\\k" in content


def test_clips_words_and_shifts_timestamps_to_output_zero() -> None:
    source_words = [
        TranscriptWord(text="antes", start_ms=3_500, end_ms=4_100),
        TranscriptWord(text="dentro", start_ms=4_500, end_ms=5_000),
        TranscriptWord(text="agora", start_ms=5_100, end_ms=5_700),
    ]
    cues, clipped_words = clip_subtitle_track(
        [SubtitleCue(start_ms=3_500, end_ms=5_700, text="antes dentro agora")],
        source_words,
        ClipRange(start_ms=4_350, end_ms=6_000),
    )
    assert [word.text for word in clipped_words] == ["dentro", "agora"]
    assert clipped_words[0].start_ms == 150
    assert cues[0].start_ms == 150
    assert cues[0].text == "dentro agora"


def test_ass_declares_resolution_safe_margins_and_two_lines(tmp_path: Path) -> None:
    path = export_subtitles(
        [SubtitleCue(start_ms=0, end_ms=1500, text="uma legenda suficientemente longa para duas linhas")],
        tmp_path / "vertical.ass",
        animated=True,
        resolution=(540, 960),
    )
    content = path.read_text(encoding="utf-8-sig")
    assert "PlayResX: 540" in content
    assert "PlayResY: 960" in content
    assert "\\N" in content
    assert "&H004FD5FF" in content
    style_line = next(line for line in content.splitlines() if line.startswith("Style: CortaFlow"))
    assert ",31," in style_line
