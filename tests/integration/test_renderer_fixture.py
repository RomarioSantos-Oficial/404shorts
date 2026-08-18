from pathlib import Path

import pytest

from cortaflow.domain.clip import ClipRange
from cortaflow.domain.editing import AudioSettings, ReframeSettings, SubtitleStyle, TimelineClip
from cortaflow.domain.project import ExportSettings, ReframeKeyframe, WatermarkSettings
from cortaflow.domain.subtitle import SubtitleCue, TranscriptWord
from cortaflow.domain.tracking import CropFrame
from cortaflow.services.media_probe import probe_media
from cortaflow.services.renderer import render
from cortaflow.services.export_service import render_project_export


def test_renders_vertical_preview_atomically(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    output = tmp_path / "prévia vertical.mp4"
    events = []
    render(source, output, ExportSettings(use_nvenc=False), ClipRange(start_ms=0, end_ms=1000), preview=True, progress=events.append)
    metadata = probe_media(output)
    assert (metadata.width, metadata.height) == (540, 960)
    assert events and events[-1]["progress"] == "end"
    assert not list(tmp_path.glob("*.rendering.mp4"))


def test_renders_interpolated_reframe_keyframes(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    output = tmp_path / "movimento vertical.mp4"
    keyframes = [
        ReframeKeyframe(timestamp_ms=0, crop=CropFrame(x=0, y=0, width=100, height=180)),
        ReframeKeyframe(timestamp_ms=800, crop=CropFrame(x=220, y=0, width=100, height=180)),
    ]
    render(
        source,
        output,
        ExportSettings(use_nvenc=False),
        ClipRange(start_ms=0, end_ms=1000),
        preview=True,
        crop_keyframes=keyframes,
    )
    metadata = probe_media(output)
    assert (metadata.width, metadata.height) == (540, 960)
    assert metadata.duration_seconds == pytest.approx(1.0, abs=.1)


@pytest.mark.parametrize(
    "preview,expected_size",
    [(True, (540, 960)), (False, (720, 1280))],
)
def test_renders_watermark_in_preview_and_final_pipeline(
    tmp_path: Path, preview: bool, expected_size: tuple[int, int]
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    watermark = tmp_path / "marca.ppm"
    # Imagem RGB vermelha 4 × 4, formato simples e legalmente controlado.
    watermark.write_bytes(b"P6\n4 4\n255\n" + bytes((255, 0, 0)) * 16)
    output = tmp_path / f"com marca-{'preview' if preview else 'final'}.mp4"
    settings = ExportSettings(
        width=720,
        height=1280,
        use_nvenc=False,
        watermark=WatermarkSettings(
            enabled=True,
            image_path=watermark,
            position="top-left",
            width_percent=15,
            opacity=0.5,
        ),
    )

    render_project_export(
        source,
        output,
        settings,
        ClipRange(start_ms=0, end_ms=800),
        [],
        SubtitleStyle(),
        [],
        preview,
        [],
        ReframeSettings(),
        AudioSettings(),
        (320, 180),
        [],
    )

    metadata = probe_media(output)
    assert (metadata.width, metadata.height) == expected_size
    assert metadata.duration_seconds == pytest.approx(.8, abs=.12)


@pytest.mark.parametrize("width,height", [(720, 1280), (1080, 1920)])
def test_renders_readable_shifted_subtitles_at_final_resolutions(
    tmp_path: Path, width: int, height: int
) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    output = tmp_path / f"legenda-{width}x{height}.mp4"
    render_project_export(
        source,
        output,
        ExportSettings(width=width, height=height, use_nvenc=False),
        ClipRange(start_ms=400, end_ms=1000),
        [SubtitleCue(start_ms=500, end_ms=900, text="Legenda dentro do corte")],
        SubtitleStyle(font_size=62),
        [],
        False,
        [
            TranscriptWord(text="Legenda", start_ms=500, end_ms=600),
            TranscriptWord(text="dentro", start_ms=610, end_ms=700),
            TranscriptWord(text="do", start_ms=710, end_ms=760),
            TranscriptWord(text="corte", start_ms=770, end_ms=900),
        ],
        ReframeSettings(),
        AudioSettings(),
        (320, 180),
        [],
    )
    metadata = probe_media(output)
    assert (metadata.width, metadata.height) == (width, height)
    assert metadata.duration_seconds == pytest.approx(.6, abs=.1)


def test_renders_complete_edited_timeline(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    output = tmp_path / "timeline-editada.mp4"
    clips = [
        TimelineClip(clip_id="v1", track="video", source_start_ms=0, source_end_ms=700, timeline_start_ms=0, transition_ms=100),
        TimelineClip(clip_id="v2", track="video", source_start_ms=1000, source_end_ms=1700, timeline_start_ms=900, transition_ms=100),
        TimelineClip(clip_id="a1", track="audio", source_start_ms=0, source_end_ms=700, timeline_start_ms=0, transition_ms=100),
        TimelineClip(clip_id="a2", track="audio", source_start_ms=1000, source_end_ms=1700, timeline_start_ms=900, transition_ms=100),
    ]
    render_project_export(
        source,
        output,
        ExportSettings(use_nvenc=False),
        ClipRange(start_ms=0, end_ms=1600),
        [],
        SubtitleStyle(),
        [],
        True,
        [],
        ReframeSettings(aspect_ratio="1:1"),
        AudioSettings(volume=.75),
        (320, 180),
        clips,
    )
    metadata = probe_media(output)
    assert (metadata.width, metadata.height) == (540, 540)
    assert metadata.duration_seconds == pytest.approx(1.6, abs=.15)
