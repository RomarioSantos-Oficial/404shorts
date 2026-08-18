from pathlib import Path
import pytest
from cortaflow.infrastructure.paths import ensure_safe_output_directory, sanitize_filename


@pytest.mark.parametrize(("value", "expected"), [("vídeo: teste?.mp4", "vídeo_ teste_.mp4"), ("CON", "_CON"), ("  ", "video")])
def test_sanitize_filename(value: str, expected: str) -> None:
    assert sanitize_filename(value) == expected


def test_rejects_drive_root() -> None:
    with pytest.raises(ValueError):
        ensure_safe_output_directory(Path(Path.cwd().anchor))

