"""Authorized downloads through yt-dlp's Python API."""

from collections.abc import Callable
import importlib.util
import ipaddress
from pathlib import Path
import shutil
import sysconfig
from threading import Event
from typing import Any
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from cortaflow.domain.media import MediaFormat, MediaMetadata
from cortaflow.infrastructure.ffmpeg import FFmpegNotFoundError, find_executable
from cortaflow.infrastructure.paths import ensure_safe_output_directory, sanitize_filename


class DownloadCancelled(RuntimeError):
    pass


class JavaScriptRuntimeNotFound(RuntimeError):
    """Raised when the current YouTube extraction requirements are unavailable."""


def find_javascript_runtime() -> tuple[str, Path]:
    """Locate a supported JavaScript runtime, preferring the project-local Deno."""
    scripts_directory = Path(sysconfig.get_path("scripts"))
    candidates = (
        ("deno", scripts_directory / "deno.exe"),
        ("deno", scripts_directory / "deno"),
        ("deno", Path(shutil.which("deno") or "")),
        ("node", Path(shutil.which("node") or "")),
        ("bun", Path(shutil.which("bun") or "")),
        ("quickjs", Path(shutil.which("qjs") or "")),
    )
    for runtime, path in candidates:
        if str(path) not in {"", "."} and path.is_file():
            return runtime, path.resolve()
    raise JavaScriptRuntimeNotFound(
        "O componente JavaScript necessário para baixar do YouTube não foi encontrado. "
        "Reinstale as dependências oficiais do projeto (yt-dlp[default,deno])."
    )


def _javascript_options(required: bool) -> dict[str, Any]:
    try:
        runtime, path = find_javascript_runtime()
    except JavaScriptRuntimeNotFound:
        if required:
            raise
        return {}
    if importlib.util.find_spec("yt_dlp_ejs") is None:
        if required:
            raise JavaScriptRuntimeNotFound(
                "O componente yt-dlp-ejs necessário para baixar do YouTube não está instalado."
            )
        return {}
    return {"js_runtimes": {runtime: {"path": str(path)}}}


def validate_public_url(value: str) -> str:
    cleaned = value.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Informe uma URL HTTP ou HTTPS válida.")
    hostname = parsed.hostname.lower().rstrip(".")
    if parsed.username or parsed.password or hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("A URL não pode conter credenciais nem apontar para o computador local.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address and not address.is_global:
        raise ValueError("A URL deve apontar para um endereço público.")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("A porta informada na URL é inválida.") from exc
    return cleaned


def inspect_url(url: str) -> MediaMetadata:
    safe_url = validate_public_url(url)
    try:
        with YoutubeDL(_metadata_options()) as client:
            extracted = client.extract_info(safe_url, download=False)
            info = client.sanitize_info(extracted)
    except DownloadError as exc:
        raise RuntimeError("Não foi possível consultar esse vídeo. Verifique a URL, a disponibilidade e a conexão.") from exc
    if not isinstance(info, dict):
        raise RuntimeError("A plataforma não retornou metadados válidos para este vídeo.")
    if info.get("is_live"):
        raise ValueError("Transmissões ao vivo não são suportadas nesta versão.")
    formats = _video_formats(info.get("formats") or [])
    return MediaMetadata(
        source=safe_url,
        title=str(info.get("title") or "Vídeo"),
        duration_seconds=float(info.get("duration") or 0),
        width=info.get("width"),
        height=info.get("height"),
        platform=str(info.get("extractor_key") or "Plataforma compatível"),
        thumbnail_url=info.get("thumbnail"),
        formats=formats,
    )


def download_media(
    url: str,
    output_dir: Path,
    format_selector: str | None,
    suggested_title: str,
    progress: Callable[[dict[str, Any]], None],
    cancelled: Event,
) -> Path:
    safe_url = validate_public_url(url)
    destination = ensure_safe_output_directory(output_dir)
    observed_paths: list[Path] = []

    def hook(status: dict[str, Any]) -> None:
        if cancelled.is_set():
            raise DownloadCancelled("Download cancelado.")
        _remember_paths(status, observed_paths)
        progress(status)

    def postprocessor_hook(status: dict[str, Any]) -> None:
        if cancelled.is_set():
            raise DownloadCancelled("Download cancelado.")
        _remember_paths(status, observed_paths)
        if status.get("status") == "finished":
            progress({"status": "postprocessing", "info_dict": status.get("info_dict", {})})

    options = _download_options(destination, format_selector, suggested_title, hook, postprocessor_hook)
    try:
        with YoutubeDL(options) as client:
            info = client.extract_info(safe_url, download=True)
            _remember_paths(info, observed_paths)
            observed_paths.append(Path(client.prepare_filename(info)))
            return _resolve_downloaded_path(destination, info, observed_paths)
    except DownloadCancelled:
        raise
    except DownloadError as exc:
        if cancelled.is_set():
            raise DownloadCancelled("Download cancelado.") from exc
        raise RuntimeError("Falha ao baixar o vídeo. Verifique a URL e a conexão.") from exc


def _metadata_options() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreconfig": True,
        # Prefer the Windows trust store. This keeps TLS verification enabled
        # while supporting locally trusted certificate authorities.
        "compat_opts": ["no-certifi"],
        **_javascript_options(required=False),
    }


def _video_formats(raw_formats: list[dict[str, Any]]) -> list[MediaFormat]:
    formats: list[MediaFormat] = []
    seen: set[tuple[int | None, str | None, float | None, bool]] = set()
    for item in reversed(raw_formats):
        format_id = str(item.get("format_id") or "").strip()
        if not format_id or item.get("vcodec") in {None, "none"}:
            continue
        has_audio = item.get("acodec") not in {None, "none"}
        key = (item.get("height"), item.get("ext"), item.get("fps"), has_audio)
        if key in seen:
            continue
        seen.add(key)
        selector = format_id if has_audio else f"{format_id}+ba/{format_id}"
        audio_label = "com áudio" if has_audio else "vídeo + melhor áudio"
        fps = item.get("fps")
        fps_label = f" · {round(float(fps))} fps" if fps else ""
        formats.append(
            MediaFormat(
                format_id=format_id,
                selector=selector,
                label=f"{item.get('height') or '?'}p{fps_label} · {item.get('ext') or '?'} · {audio_label}",
                width=item.get("width"),
                height=item.get("height"),
                extension=item.get("ext"),
                fps=fps,
                has_audio=has_audio,
            )
        )
    return formats


def _download_options(
    destination: Path,
    format_selector: str | None,
    suggested_title: str,
    progress_hook: Callable[[dict[str, Any]], None],
    postprocessor_hook: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    safe_title = sanitize_filename(suggested_title, fallback="video").replace("%", "%%")
    try:
        ffmpeg_path = find_executable("ffmpeg")
    except FFmpegNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg não foi encontrado. Instale ou configure o FFmpeg antes de baixar."
        ) from exc
    return {
        "paths": {"home": str(destination), "temp": str(destination)},
        "outtmpl": {"default": f"{safe_title} [%(id)s].%(ext)s"},
        "format": format_selector or "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "ignoreconfig": True,
        "compat_opts": ["no-certifi"],
        "extractor_args": {"youtube": {"player_client": ["android", "mweb"]}},
        "ffmpeg_location": str(ffmpeg_path),
        "progress_hooks": [progress_hook],
        "postprocessor_hooks": [postprocessor_hook],
        "quiet": True,
        "no_warnings": True,
        "windowsfilenames": True,
        "trim_file_name": 180,
        "overwrites": False,
        "continuedl": True,
        **_javascript_options(required=True),
    }


def _remember_paths(payload: dict[str, Any], paths: list[Path]) -> None:
    for key in ("filename", "filepath", "_filename"):
        value = payload.get(key)
        if value:
            paths.append(Path(str(value)))
    info = payload.get("info_dict")
    if isinstance(info, dict):
        _remember_paths(info, paths)
    for key in ("requested_downloads", "requested_formats"):
        items = payload.get(key) or []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    _remember_paths(item, paths)


def _resolve_downloaded_path(destination: Path, info: dict[str, Any], observed_paths: list[Path]) -> Path:
    destination = destination.resolve()
    candidates: list[Path] = []
    for candidate in observed_paths:
        resolved = candidate.resolve() if candidate.is_absolute() else (destination / candidate).resolve()
        if resolved.is_relative_to(destination) and resolved.is_file() and resolved.suffix.lower() not in {".part", ".ytdl"}:
            candidates.append(resolved)
    video_id = str(info.get("id") or "").strip()
    if video_id:
        candidates.extend(path.resolve() for path in destination.iterdir() if path.is_file() and f"[{video_id}]" in path.stem and path.suffix.lower() not in {".part", ".ytdl"})
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise RuntimeError("O download terminou, mas o arquivo final não foi localizado com segurança.")
    return max(unique, key=lambda path: (path.suffix.lower() == ".mp4", path.stat().st_mtime_ns))
