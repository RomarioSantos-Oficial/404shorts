from pathlib import Path

import pytest

from cortaflow.domain.subtitle import SubtitleCue
from cortaflow.services.media_probe import probe_media
from cortaflow.services.subtitles import burn_subtitles


def test_burns_ass_subtitle_into_local_fixture(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    destination = tmp_path / "vídeo legendado.mp4"
    result = burn_subtitles(
        source,
        destination,
        [SubtitleCue(start_ms=0, end_ms=1_500, text="Legenda autorizada")],
    )
    assert result == destination.resolve()
    assert result.stat().st_size > 0
    metadata = probe_media(result)
    assert metadata.duration_seconds == pytest.approx(2.0, abs=0.1)
