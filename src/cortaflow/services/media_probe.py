"""Local media metadata service."""

from pathlib import Path
from typing import Any
from cortaflow.domain.media import MediaMetadata
from cortaflow.infrastructure.ffmpeg import run_ffprobe_json


def parse_frame_rate(value: Any) -> float | None:
    """Convert an FFprobe frame-rate value such as ``30000/1001`` to FPS."""
    if value in (None, "", "0/0"):
        return None
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/", maxsplit=1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def metadata_from_probe(path: Path, payload: dict[str, Any]) -> MediaMetadata:
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    duration = payload.get("format", {}).get("duration") or video.get("duration") or 0
    fps = parse_frame_rate(video.get("avg_frame_rate")) or parse_frame_rate(video.get("r_frame_rate"))
    return MediaMetadata(
        source=str(path),
        title=path.stem,
        duration_seconds=float(duration),
        width=video.get("width"),
        height=video.get("height"),
        fps=fps,
        local_path=path.resolve(),
    )


def probe_media(path: Path) -> MediaMetadata:
    return metadata_from_probe(path, run_ffprobe_json(path))
