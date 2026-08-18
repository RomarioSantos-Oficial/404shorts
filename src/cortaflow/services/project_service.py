"""Atomic project persistence and crash recovery."""

import json
from pathlib import Path
from cortaflow.domain.project import ProjectDocument


class UnsupportedProjectVersion(ValueError):
    pass


def save_project(project: ProjectDocument, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = project.model_dump(mode="json")
    if project.source_path and project.source_path.is_absolute():
        try:
            payload["source_path"] = str(project.source_path.resolve().relative_to(path.parent.resolve()))
        except ValueError:
            pass
    watermark_path = project.export.watermark.image_path
    if watermark_path and watermark_path.is_absolute():
        try:
            payload["export"]["watermark"]["image_path"] = str(
                watermark_path.resolve().relative_to(path.parent.resolve())
            )
        except ValueError:
            pass
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_project(path: Path) -> ProjectDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise UnsupportedProjectVersion("Versão de projeto ainda não suportada.")
    project = ProjectDocument.model_validate(payload)
    if project.source_path and not project.source_path.is_absolute():
        candidate = (path.parent / project.source_path).resolve()
        if candidate.exists():
            project.source_path = candidate
    watermark_path = project.export.watermark.image_path
    if watermark_path and not watermark_path.is_absolute():
        candidate = (path.parent / watermark_path).resolve()
        if candidate.exists():
            project.export.watermark.image_path = candidate
    return project


def autosave_path(project_path: Path) -> Path:
    return project_path.with_suffix(project_path.suffix + ".autosave")


def save_autosave(project: ProjectDocument, project_path: Path) -> Path:
    return save_project(project, autosave_path(project_path))


def recovery_available(project_path: Path) -> bool:
    recovery = autosave_path(project_path)
    return recovery.exists() and (not project_path.exists() or recovery.stat().st_mtime > project_path.stat().st_mtime)
