from pathlib import Path

from cortaflow.domain.clip import ClipRange
from cortaflow.domain.project import ExportSettings
from cortaflow.services import renderer


def test_runtime_nvenc_failure_retries_with_software(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "final.mp4"
    source.write_bytes(b"fixture")
    attempts: list[set[str]] = []

    monkeypatch.setattr(renderer, "available_encoders", lambda: {"h264_nvenc", "libx264"})

    def fake_attempt(
        source, temporary, settings, clip, crop, subtitle, preview, keyframes,
        encoders, progress, cancelled, reframe_settings, audio_settings, source_size,
    ):
        attempts.append(encoders)
        if "h264_nvenc" in encoders:
            return False, "NVENC initialization failed"
        temporary.write_bytes(b"software output")
        return True, ""

    monkeypatch.setattr(renderer, "_render_attempt", fake_attempt)
    updates: list[dict] = []
    result = renderer.render(
        source,
        destination,
        ExportSettings(use_nvenc=True),
        ClipRange(start_ms=0, end_ms=1000),
        progress=updates.append,
    )
    assert result == destination
    assert destination.read_bytes() == b"software output"
    assert len(attempts) == 2
    assert updates[0]["progress"] == "fallback"
