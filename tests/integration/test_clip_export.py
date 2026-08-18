from pathlib import Path
from cortaflow.domain.clip import ClipRange
from cortaflow.services.media_probe import probe_media
from cortaflow.services.transcoder import export_clip


def test_exports_artificial_clip(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    output = tmp_path / "corte com acento.mp4"
    export_clip(source, output, ClipRange(start_ms=250, end_ms=1250))
    metadata = probe_media(output)
    assert output.exists()
    assert 0.8 <= metadata.duration_seconds <= 1.2

