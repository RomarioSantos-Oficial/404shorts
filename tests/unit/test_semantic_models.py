import io
import json
from pathlib import Path

from cortaflow.services import semantic_models


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_find_ollama_assets_requires_existing_runtime_and_registered_model(
    monkeypatch, tmp_path: Path
) -> None:
    executable = tmp_path / "ollama.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(semantic_models.shutil, "which", lambda command: str(executable))
    payload = json.dumps({"models": [{"name": semantic_models.OLLAMA_MODEL_NAME}]}).encode()
    monkeypatch.setattr(
        semantic_models,
        "urlopen",
        lambda request, timeout: _Response(payload),
    )
    assets = semantic_models.find_ollama_assets()
    assert assets is not None
    assert assets.executable == executable.resolve()
    assert assets.model_name == "cortaflow-qwen3:4b"


def test_find_ollama_assets_does_not_pull_a_missing_model(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "ollama.exe"
    executable.write_bytes(b"fixture")
    monkeypatch.setattr(semantic_models.shutil, "which", lambda command: str(executable))
    monkeypatch.setattr(
        semantic_models,
        "urlopen",
        lambda request, timeout: _Response(b'{"models":[]}'),
    )
    assert semantic_models.find_ollama_assets() is None
