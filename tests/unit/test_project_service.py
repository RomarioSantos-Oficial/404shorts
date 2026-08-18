from pathlib import Path
import os

import pytest

from cortaflow.domain.clip import ClipRange
from cortaflow.domain.project import ExportSettings, ProjectDocument, WatermarkSettings
from cortaflow.domain.tracking import SpeakerKeyframe, SpeakerOverride
from cortaflow.services.project_service import (
    UnsupportedProjectVersion,
    load_project,
    recovery_available,
    save_autosave,
    save_project,
)


def test_project_round_trip_with_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "projeto com acento.cortaflow.json"
    project = ProjectDocument(
        name="Edição pública",
        source_path=Path("Vídeos/ação.mp4"),
        clips=[ClipRange(start_ms=10, end_ms=1000)],
        speaker_keyframes=[
            SpeakerKeyframe(
                timestamp_ms=500,
                track_id=2,
                confidence=1,
                uncertain=False,
                manual=True,
            )
        ],
        speaker_overrides=[SpeakerOverride(start_ms=0, end_ms=1000, track_id=2)],
    )
    save_project(project, path)
    assert load_project(path) == project
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_source_inside_project_folder_is_saved_as_relative_path(tmp_path: Path) -> None:
    project_folder = tmp_path / "Projeto portátil"
    source = project_folder / "mídia" / "vídeo fonte.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fixture")
    path = project_folder / "edição.cortaflow.json"

    save_project(ProjectDocument(source_path=source), path)

    raw_payload = path.read_text(encoding="utf-8")
    assert '"source_path": "mídia\\\\vídeo fonte.mp4"' in raw_payload
    assert load_project(path).source_path == source.resolve()


def test_watermark_inside_project_folder_is_portable(tmp_path: Path) -> None:
    project_folder = tmp_path / "Projeto portátil"
    watermark = project_folder / "marca" / "logo.png"
    watermark.parent.mkdir(parents=True)
    watermark.write_bytes(b"fixture")
    path = project_folder / "edição.cortaflow.json"
    project = ProjectDocument(
        export=ExportSettings(
            watermark=WatermarkSettings(enabled=True, image_path=watermark)
        )
    )

    save_project(project, path)

    assert '"image_path": "marca\\\\logo.png"' in path.read_text(encoding="utf-8")
    assert load_project(path).export.watermark.image_path == watermark.resolve()


def test_newer_autosave_is_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    save_project(ProjectDocument(), path)
    old_time = path.stat().st_mtime - 10
    os.utime(path, (old_time, old_time))
    save_autosave(ProjectDocument(name="Recuperado"), path)
    assert recovery_available(path)


def test_rejects_unknown_project_version(tmp_path: Path) -> None:
    path = tmp_path / "future.json"
    path.write_text('{"format_version": 99}', encoding="utf-8")
    with pytest.raises(UnsupportedProjectVersion):
        load_project(path)
