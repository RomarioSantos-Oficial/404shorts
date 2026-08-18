"""Pinned, verified local assets for optional semantic clip ranking."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import Event
import tempfile
from typing import Any
from urllib.request import Request, urlopen
import zipfile


LLAMA_TAG = "b10410"
LLAMA_ARCHIVE_NAME = f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"
LLAMA_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/{LLAMA_ARCHIVE_NAME}"
)
LLAMA_SIZE = 34_557_018
LLAMA_SHA256 = "943f047c39843a8051a750424957852079f740bfeb6a9fa4b155d720b52d576e"

QWEN_REVISION = "bc640142c66e1fdd12af0bd68f40445458f3869b"
QWEN_FILENAME = "Qwen3-4B-Q4_K_M.gguf"
QWEN_URL = (
    f"https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/{QWEN_REVISION}/{QWEN_FILENAME}"
)
QWEN_SIZE = 2_497_280_256
QWEN_SHA256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
OLLAMA_MODEL_NAME = "cortaflow-qwen3:4b"
OLLAMA_HOST = "http://127.0.0.1:11434"

ProgressCallback = Callable[[dict[str, Any]], None]


class SemanticModelError(RuntimeError):
    pass


class SemanticModelCancelled(SemanticModelError):
    pass


@dataclass(frozen=True)
class SemanticAssets:
    llama_cli: Path
    model: Path
    backend: str = "CPU"

    @property
    def cli_command(self) -> list[str]:
        return [str(self.llama_cli), "cli"] if self.llama_cli.name == "llama.exe" else [str(self.llama_cli)]


@dataclass(frozen=True)
class OllamaAssets:
    executable: Path
    model_name: str = OLLAMA_MODEL_NAME
    host: str = OLLAMA_HOST
    backend: str = "Ollama"


def find_ollama_assets(timeout_seconds: float = 2.0) -> OllamaAssets | None:
    """Find the existing signed Ollama runtime and registered local model only."""
    candidates: list[Path] = []
    command = shutil.which("ollama")
    if command:
        candidates.append(Path(command))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")
    executable = next((item.resolve() for item in candidates if item.is_file()), None)
    if not executable:
        return None
    try:
        request = Request(f"{OLLAMA_HOST}/api/tags", headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - localhost only
            payload = json.loads(response.read().decode("utf-8"))
        names = {
            str(item.get("name", ""))
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
    except (OSError, ValueError, TypeError):
        return None
    return OllamaAssets(executable) if OLLAMA_MODEL_NAME in names else None


class SemanticModelManager:
    """Install official pinned assets without executing an unverified download."""

    def __init__(self, cache_directory: Path) -> None:
        self.cache_directory = cache_directory.resolve()
        self.download_directory = self.cache_directory / "downloads"
        self.runtime_directory = self.cache_directory / f"llama-{LLAMA_TAG}-vulkan"
        self.model_directory = self.cache_directory / "qwen3-4b"

    def assets(self) -> SemanticAssets | None:
        cli, backend = self._existing_runtime()
        model = self.model_directory / QWEN_FILENAME
        if cli and model.is_file() and model.stat().st_size == QWEN_SIZE:
            return SemanticAssets(cli, model, backend)
        return None

    def _existing_runtime(self) -> tuple[Path | None, str]:
        """Prefer a complete project-local CPU runtime, then the managed cache."""
        source_root = Path(__file__).resolve().parents[3]
        searched: set[Path] = set()
        for root in (Path.cwd(), source_root):
            root = root.resolve()
            if root in searched:
                continue
            searched.add(root)
            for directory_name in ("llhama", "llama"):
                runtime_root = root / directory_name
                if not runtime_root.is_dir():
                    continue
                matches = sorted(runtime_root.rglob("llama.exe"))
                matches.extend(sorted(runtime_root.rglob("llama-cli.exe")))
                if matches:
                    executable = matches[0]
                    backend = "CPU" if "cpu" in str(executable.parent).lower() else "Local"
                    return executable, backend
        for name in ("llama.exe", "llama-cli.exe"):
            executable = self.runtime_directory / name
            if executable.is_file():
                return executable, "Vulkan"
        return None, "indisponível"

    def is_ready(self) -> bool:
        return self.assets() is not None

    def install(
        self,
        progress: ProgressCallback | None = None,
        cancelled: Event | None = None,
    ) -> SemanticAssets:
        existing = self.assets()
        if existing:
            return existing
        self.download_directory.mkdir(parents=True, exist_ok=True)
        llama_archive = self._download(
            LLAMA_URL,
            self.download_directory / LLAMA_ARCHIVE_NAME,
            LLAMA_SIZE,
            LLAMA_SHA256,
            "llama.cpp Vulkan",
            progress,
            cancelled,
        )
        self._install_runtime(llama_archive)
        self.model_directory.mkdir(parents=True, exist_ok=True)
        self._download(
            QWEN_URL,
            self.model_directory / QWEN_FILENAME,
            QWEN_SIZE,
            QWEN_SHA256,
            "Qwen3-4B Q4_K_M",
            progress,
            cancelled,
        )
        assets = self.assets()
        if not assets:
            raise SemanticModelError("A instalação terminou sem localizar os ativos semânticos.")
        return assets

    def _download(
        self,
        url: str,
        destination: Path,
        expected_size: int,
        expected_sha256: str,
        label: str,
        progress: ProgressCallback | None,
        cancelled: Event | None,
    ) -> Path:
        if destination.is_file() and destination.stat().st_size == expected_size:
            if _sha256(destination) == expected_sha256:
                return destination
            raise SemanticModelError(f"O arquivo existente de {label} falhou na verificação SHA-256.")
        partial = destination.with_name(destination.name + ".part")
        downloaded = partial.stat().st_size if partial.is_file() else 0
        if downloaded > expected_size:
            raise SemanticModelError(f"O download parcial de {label} possui tamanho inválido.")
        request = Request(url, headers={"User-Agent": "CortaFlow-AI/0.1"})
        if downloaded:
            request.add_header("Range", f"bytes={downloaded}-")
        with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned HTTPS origins
            resumed = downloaded > 0 and getattr(response, "status", None) == 206
            mode = "ab" if resumed else "wb"
            if not resumed:
                downloaded = 0
            with partial.open(mode) as stream:
                while True:
                    if cancelled and cancelled.is_set():
                        raise SemanticModelCancelled("Instalação da IA semântica cancelada.")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(
                            {
                                "status": "model_download",
                                "label": label,
                                "downloaded_bytes": downloaded,
                                "total_bytes": expected_size,
                            }
                        )
        if partial.stat().st_size != expected_size:
            raise SemanticModelError(f"O download de {label} terminou com tamanho inesperado.")
        if progress:
            progress({"status": "model_verification", "label": label})
        if _sha256(partial) != expected_sha256:
            raise SemanticModelError(f"O download de {label} falhou na verificação SHA-256.")
        partial.replace(destination)
        return destination

    def _install_runtime(self, archive: Path) -> None:
        if any(
            (self.runtime_directory / name).is_file()
            for name in ("llama.exe", "llama-cli.exe")
        ):
            return
        if self.runtime_directory.exists():
            raise SemanticModelError(
                "A pasta do llama.cpp está incompleta. Remova-a manualmente antes de reinstalar."
            )
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="llama-install-", dir=self.cache_directory) as temporary:
            temporary_path = Path(temporary)
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    target = (temporary_path / member.filename).resolve()
                    if not target.is_relative_to(temporary_path.resolve()):
                        raise SemanticModelError("O pacote do llama.cpp contém um caminho inseguro.")
                package.extractall(temporary_path)
            cli = next(temporary_path.rglob("llama.exe"), None) or next(
                temporary_path.rglob("llama-cli.exe"), None
            )
            if not cli:
                raise SemanticModelError("O pacote verificado não contém llama-cli.exe.")
            cli.parent.replace(self.runtime_directory)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
