from pathlib import Path
from cortaflow.services.scene_detection import detect_scenes, detect_silences


def test_analyzes_artificial_fixture() -> None:
    source = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    scenes = detect_scenes(source)
    silences = detect_silences(source)
    assert isinstance(scenes, list)
    assert silences == []

