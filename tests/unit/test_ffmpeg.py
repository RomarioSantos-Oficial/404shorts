from pathlib import Path

import pytest

from cortaflow.infrastructure import ffmpeg


def test_missing_ffmpeg_has_a_specific_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: None)
    monkeypatch.setattr(ffmpeg.Path, "home", classmethod(lambda _cls: tmp_path))

    with pytest.raises(ffmpeg.FFmpegNotFoundError, match="ffmpeg"):
        ffmpeg.find_executable("ffmpeg")
