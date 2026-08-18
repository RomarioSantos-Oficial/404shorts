from pathlib import Path
from cortaflow.domain.clip import ClipRange
from cortaflow.services.transcoder import build_clip_command


def test_command_is_argument_list_and_supports_optional_audio(tmp_path: Path) -> None:
    command = build_clip_command(Path("C:/Vídeos/vídeo com espaço.mp4"), tmp_path / "saída.mp4", ClipRange(start_ms=1250, end_ms=3250))
    assert command[0].endswith("ffmpeg.exe")
    assert "00:00:01.250" in command
    assert "00:00:02.000" in command
    assert "0:a?" in command
    assert isinstance(command, list)

