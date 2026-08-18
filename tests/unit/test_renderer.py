from pathlib import Path
from cortaflow.domain.clip import ClipRange
from cortaflow.domain.editing import AudioSettings, ReframeSettings, TimelineClip
from cortaflow.domain.project import ExportSettings, ReframeKeyframe, WatermarkSettings
from cortaflow.domain.tracking import CropFrame
from cortaflow.services.renderer import build_motion_crop_filter, build_render_command, build_timeline_render_command, choose_video_encoder, output_dimensions


def test_encoder_falls_back_to_software() -> None:
    settings = ExportSettings(codec="h264", use_nvenc=True)
    assert choose_video_encoder(settings, {"libx264"}) == "libx264"
    assert choose_video_encoder(settings, {"libx264", "h264_nvenc"}) == "h264_nvenc"


def test_preview_command_has_vertical_filters_and_optional_audio(tmp_path: Path) -> None:
    command = build_render_command(Path("C:/a b.mp4"), tmp_path / "preview.mp4", ExportSettings(use_nvenc=False), ClipRange(start_ms=0, end_ms=1000), CropFrame(x=10, y=0, width=608, height=1080), preview=True, encoders={"libx264"})
    filters = command[command.index("-vf") + 1]
    assert "crop=608:1080:10:0" in filters
    assert "scale=540:960" in filters
    assert "0:a?" in command
    assert command[command.index("-c:v") + 1] == "libx264"


def test_motion_crop_uses_interpolated_per_frame_expressions() -> None:
    keyframes = [
        ReframeKeyframe(timestamp_ms=1000, crop=CropFrame(x=0, y=0, width=608, height=1080)),
        ReframeKeyframe(timestamp_ms=2000, crop=CropFrame(x=600, y=0, width=608, height=1080)),
    ]
    value = build_motion_crop_filter(keyframes, ClipRange(start_ms=1000, end_ms=3000))
    assert value is not None
    assert value.startswith("crop=608:1080")
    assert "t\\,1.000" in value
    assert "600" in value


def test_motion_crop_jumps_at_scene_cut_instead_of_panning_between_speakers() -> None:
    keyframes = [
        ReframeKeyframe(
            timestamp_ms=0,
            crop=CropFrame(x=0, y=0, width=608, height=1080),
        ),
        ReframeKeyframe(
            timestamp_ms=1000,
            crop=CropFrame(x=1000, y=0, width=608, height=1080),
            scene_reset=True,
        ),
    ]

    value = build_motion_crop_filter(keyframes, ClipRange(start_ms=0, end_ms=2000))

    assert value is not None
    assert "if(lt(t\\,1.000)\\,0\\,1000)" in value
    assert "1000*(t-0.000)" not in value


def test_motion_crop_limits_automatic_expression_and_preserves_manual_keyframes() -> None:
    keyframes = [
        ReframeKeyframe(
            timestamp_ms=index * 400,
            crop=CropFrame(x=999 if index == 100 else index, y=0, width=608, height=1080),
            manual=index == 100,
        )
        for index in range(200)
    ]
    value = build_motion_crop_filter(keyframes, ClipRange(start_ms=0, end_ms=80_000))
    assert value is not None
    assert "999" in value
    assert value.count("if(lt") <= 120


def test_render_command_consumes_volume_and_normalization(tmp_path: Path) -> None:
    command = build_render_command(
        Path("C:/source.mp4"), tmp_path / "out.mp4", ExportSettings(use_nvenc=False),
        ClipRange(start_ms=0, end_ms=1000), encoders={"libx264"},
        audio_settings=AudioSettings(volume=0.5, normalize=True),
    )
    assert command[command.index("-af") + 1].startswith("volume=0.500,loudnorm=")


def test_render_command_places_transparent_watermark_after_reframe(tmp_path: Path) -> None:
    watermark = tmp_path / "minha marca.png"
    settings = ExportSettings(
        use_nvenc=False,
        watermark=WatermarkSettings(
            enabled=True,
            image_path=watermark,
            position="bottom-right",
            width_percent=20,
            opacity=0.6,
            margin_percent=4,
        ),
    )
    command = build_render_command(
        Path("C:/source.mp4"), tmp_path / "out.mp4", settings,
        ClipRange(start_ms=0, end_ms=1000), encoders={"libx264"},
    )
    graph = command[command.index("-filter_complex") + 1]
    assert str(watermark.resolve()) in command
    assert "scale=216:-1[watermark]" in graph
    assert "colorchannelmixer=aa=0.600" in graph
    assert "overlay=x='W-w-43':y='H-h-77'" in graph
    assert command[command.index("-map") + 1] == "[vout]"


def test_output_dimensions_follow_project_aspect_ratio() -> None:
    settings = ExportSettings(width=1080, height=1920)
    assert output_dimensions(settings, ReframeSettings(aspect_ratio="1:1")) == (1080, 1080)
    assert output_dimensions(settings, ReframeSettings(aspect_ratio="4:5")) == (1080, 1350)
    assert output_dimensions(settings, ReframeSettings(aspect_ratio="original"), (1920, 1080)) == (1920, 1080)
    assert output_dimensions(settings, ReframeSettings(aspect_ratio="1:1"), preview=True) == (540, 540)


def test_timeline_command_consumes_positions_transitions_and_audio(tmp_path: Path) -> None:
    clips = [
        TimelineClip(clip_id="v1", track="video", source_start_ms=0, source_end_ms=1000, timeline_start_ms=0, transition_ms=200),
        TimelineClip(clip_id="v2", track="video", source_start_ms=1000, source_end_ms=2000, timeline_start_ms=1500),
        TimelineClip(clip_id="a1", track="audio", source_start_ms=0, source_end_ms=1000, timeline_start_ms=500),
    ]
    command = build_timeline_render_command(
        Path("C:/source.mp4"), tmp_path / "timeline.mp4", ExportSettings(use_nvenc=False),
        clips, encoders={"libx264"}, audio_settings=AudioSettings(volume=0.75),
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "split=2" in graph
    assert "fade=t=in" in graph
    assert "adelay=500|500" in graph
    assert "volume=0.750" in graph
    assert "d=2.500" in graph


def test_timeline_command_applies_custom_watermark_position(tmp_path: Path) -> None:
    clips = [
        TimelineClip(
            clip_id="v1", track="video", source_start_ms=0,
            source_end_ms=1000, timeline_start_ms=0,
        )
    ]
    watermark = tmp_path / "logo.webp"
    settings = ExportSettings(
        use_nvenc=False,
        watermark=WatermarkSettings(
            enabled=True, image_path=watermark, position="custom",
            custom_x_percent=25, custom_y_percent=75,
        ),
    )
    command = build_timeline_render_command(
        Path("C:/source.mp4"), tmp_path / "timeline.mp4", settings,
        clips, encoders={"libx264"}, source_has_audio=False,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert str(watermark.resolve()) in command
    assert "overlay=x='(W-w)*0.2500':y='(H-h)*0.7500'" in graph
