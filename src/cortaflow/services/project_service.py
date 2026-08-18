"""Persistence for portable CortaFlow projects."""

import json
from pathlib import Path

from cortaflow.domain.project import ProjectDocument


class UnsupportedProjectVersion(ValueError):
    pass


def _relative_if_inside(value: Path | None, project_dir: Path) -> str | None:
    if not value:
        return None
    if not value.is_absolute():
        return str(value)
    try:
        return value.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(value)


def _normalise_layer_paths(payload: dict, project_dir: Path) -> None:
    """Make image and font assets in both project and sequence layers portable."""
    layer_groups = [payload.get("layers", [])]
    layer_groups.extend(
        sequence.get("layers", [])
        for sequence in payload.get("sequences", [])
        if isinstance(sequence, dict)
    )
    for layers in layer_groups:
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if layer.get("source_path"):
                layer["source_path"] = _relative_if_inside(Path(layer["source_path"]), project_dir)
            font_name = layer.get("font_name")
            if font_name and Path(font_name).is_absolute():
                layer["font_name"] = _relative_if_inside(Path(font_name), project_dir)


def _resolve_layer_paths(project: ProjectDocument, project_dir: Path) -> None:
    groups = [project.layers]
    groups.extend(sequence.layers for sequence in project.sequences)
    for layers in groups:
        for layer in layers:
            if layer.source_path:
                candidate = Path(layer.source_path)
                if not candidate.is_absolute():
                    candidate = (project_dir / candidate).resolve()
                if candidate.exists():
                    layer.source_path = str(candidate)
            font_path = Path(layer.font_name)
            if font_path.is_absolute() or "/" in layer.font_name or "\\" in layer.font_name:
                if not font_path.is_absolute():
                    font_path = (project_dir / font_path).resolve()
                if font_path.exists():
                    layer.font_name = str(font_path)


def save_project(project: ProjectDocument, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = project.model_dump(mode="json")
    project_dir = path.parent.resolve()
    if project.source_path:
        payload["source_path"] = _relative_if_inside(project.source_path, project_dir)
    watermark_path = project.export.watermark.image_path
    if watermark_path:
        payload["export"]["watermark"]["image_path"] = _relative_if_inside(watermark_path, project_dir)
    _normalise_layer_paths(payload, project_dir)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_project(path: Path) -> ProjectDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise UnsupportedProjectVersion("Versão de projeto ainda não suportada.")
    project = ProjectDocument.model_validate(payload)
    project_dir = path.parent.resolve()
    if project.source_path and not project.source_path.is_absolute():
        candidate = (project_dir / project.source_path).resolve()
        if candidate.exists():
            project.source_path = candidate
    watermark_path = project.export.watermark.image_path
    if watermark_path and not watermark_path.is_absolute():
        candidate = (project_dir / watermark_path).resolve()
        if candidate.exists():
            project.export.watermark.image_path = candidate
    _resolve_layer_paths(project, project_dir)
    return project


def autosave_path(project_path: Path) -> Path:
    return project_path.with_suffix(project_path.suffix + ".autosave")


def save_autosave(project: ProjectDocument, project_path: Path) -> Path:
    return save_project(project, autosave_path(project_path))


def recovery_available(project_path: Path) -> bool:
    recovery = autosave_path(project_path)
    return recovery.exists() and (not project_path.exists() or recovery.stat().st_mtime > project_path.stat().st_mtime)
