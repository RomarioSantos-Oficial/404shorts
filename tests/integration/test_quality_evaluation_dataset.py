from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from cortaflow.domain.analysis import ClipSelectionSettings
from cortaflow.domain.clip import ClipRange
from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.infrastructure.ffmpeg import find_executable
from cortaflow.services.clip_scoring import HeuristicClipRanker, suggest_clips
from cortaflow.services.media_probe import probe_media
from cortaflow.services.quality_evaluation import evaluate_clip_quality
from cortaflow.services.subtitles import group_words


@dataclass(frozen=True)
class ControlledCase:
    name: str
    category: str
    color: str
    first_idea: str = ""
    second_idea: str = ""
    size: str = "320x180"
    two_people: bool = False
    has_speech: bool = True


CASES = (
    ControlledCase(
        "podcast-duas-pessoas", "podcast com dois participantes", "202b38",
        "Como uma conversa sincera revela ideias que ninguém percebe no começo?",
        "O convidado responde com exemplo prático e conclui a história claramente.",
        two_people=True,
    ),
    ControlledCase(
        "aula", "aula", "23405a",
        "Aprenda o primeiro passo para resolver este problema de maneira simples.",
        "Portanto o exemplo demonstra a técnica e confirma o resultado esperado.",
    ),
    ControlledCase(
        "entrevista", "entrevista", "493548",
        "Por que esta experiência mudou completamente sua decisão profissional naquela época?",
        "A resposta explica o motivo e termina com uma conclusão surpreendente.",
        two_people=True,
    ),
    ControlledCase(
        "sem-fala", "vídeo sem fala", "35604a", has_speech=False,
    ),
    ControlledCase(
        "pouca-luz", "vídeo com pouca luz", "050505",
        "Atenção para o detalhe importante mesmo quando a imagem está escura.",
        "Finalmente a explicação oferece valor e encerra o assunto sem dúvida.",
    ),
    ControlledCase(
        "depoimento", "depoimento", "5b4032",
        "Imagine descobrir uma solução depois de enfrentar este problema por anos.",
        "O resultado trouxe alegria e transformou completamente minha rotina diária.",
        size="180x320",
    ),
    ControlledCase(
        "tutorial", "tutorial", "314c63",
        "Veja como fazer esta configuração corretamente em apenas três passos.",
        "Use este ajuste final porque ele evita o erro mais comum.",
    ),
    ControlledCase(
        "noticia", "notícia explicativa", "39495c",
        "Você sabe por que esta mudança afeta tantas pessoas agora?",
        "A análise apresenta a causa principal e resume as consequências claramente.",
    ),
    ControlledCase(
        "debate", "debate", "543746",
        "Ninguém percebe o argumento central quando observa apenas os números iniciais.",
        "Então a opinião contrária mostra outro exemplo e fecha o raciocínio.",
        two_people=True,
    ),
    ControlledCase(
        "produto", "demonstração de produto", "294c45",
        "Descubra como este recurso resolve rapidamente uma dificuldade muito comum.",
        "O exemplo prático confirma o valor e mostra o resultado final.",
    ),
)


def _make_transcript(case: ControlledCase) -> Transcript:
    if not case.has_speech:
        return Transcript(language="pt")
    words: list[TranscriptWord] = []
    for sentence_index, sentence in enumerate((case.first_idea, case.second_idea)):
        tokens = sentence.split()
        sentence_start = sentence_index * 6_000
        for index, token in enumerate(tokens):
            start_ms = sentence_start + round(index * 6_000 / len(tokens))
            end_ms = sentence_start + round((index + 1) * 6_000 / len(tokens))
            words.append(TranscriptWord(text=token, start_ms=start_ms, end_ms=end_ms))
    return Transcript(language="pt", words=words, cues=group_words(words))


def _create_controlled_video(case: ControlledCase, destination: Path) -> None:
    video_source = f"color=c=0x{case.color}:s={case.size}:r=12:d=12"
    command = [
        str(find_executable("ffmpeg")), "-hide_banner", "-loglevel", "error",
        "-nostdin", "-y", "-f", "lavfi", "-i", video_source,
    ]
    if case.has_speech:
        # Synthetic audio keeps the fixture redistributable. The transcript is a
        # controlled reference, so this test evaluates selection rather than ASR.
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:d=12"]
    if case.two_people:
        width, height = (int(value) for value in case.size.split("x"))
        filter_value = (
            f"drawbox=x={width // 12}:y={height // 5}:w={width // 3}:h={height * 3 // 5}:"
            "color=0x8aa5c2:t=fill,"
            f"drawbox=x={width * 7 // 12}:y={height // 5}:w={width // 3}:h={height * 3 // 5}:"
            "color=0xc28a8a:t=fill"
        )
        command += ["-vf", filter_value]
    command += [
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
    ]
    if case.has_speech:
        command += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    else:
        command += ["-an"]
    command.append(str(destination))
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr[-800:]


def test_ten_legally_controlled_videos_against_human_marked_cuts(tmp_path: Path) -> None:
    all_metrics = []
    observed_categories: set[str] = set()
    for case in CASES:
        media_path = tmp_path / f"{case.name}.mp4"
        _create_controlled_video(case, media_path)
        metadata = probe_media(media_path)
        assert 11.9 <= metadata.duration_seconds <= 12.1
        assert metadata.width and metadata.height

        transcript = _make_transcript(case)
        references = (
            [ClipRange(start_ms=0, end_ms=6_000), ClipRange(start_ms=6_000, end_ms=12_000)]
            if case.has_speech else []
        )
        suggestions = suggest_clips(
            transcript,
            12_000,
            settings=ClipSelectionSettings(
                min_seconds=5,
                preferred_seconds=6,
                max_seconds=7,
                max_results=2,
                ranking_mode="heuristic",
            ),
            ranker=HeuristicClipRanker(),
        )
        metrics = evaluate_clip_quality(suggestions, references, transcript)
        all_metrics.append(metrics)
        observed_categories.add(case.category)

    assert len(CASES) == 10
    assert {
        "podcast com dois participantes", "aula", "entrevista",
        "vídeo sem fala", "vídeo com pouca luz",
    } <= observed_categories

    macro_precision = sum(item.precision for item in all_metrics) / len(all_metrics)
    macro_diversity = sum(item.diversity for item in all_metrics) / len(all_metrics)
    macro_cut_speech = sum(item.cut_speech_rate for item in all_metrics) / len(all_metrics)
    macro_legibility = sum(item.subtitle_legibility for item in all_metrics) / len(all_metrics)
    print(
        "10 vídeos controlados: "
        f"precisão={macro_precision:.3f}, diversidade={macro_diversity:.3f}, "
        f"fala_cortada={macro_cut_speech:.3f}, legibilidade={macro_legibility:.3f}"
    )
    assert macro_precision >= 0.90
    assert macro_diversity >= 0.80
    assert macro_cut_speech <= 0.05
    assert macro_legibility >= 0.90
