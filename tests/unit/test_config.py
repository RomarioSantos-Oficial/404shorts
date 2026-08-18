from pathlib import Path

from cortaflow.config import AppConfig


def test_config_creates_runtime_directories(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "dados",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    config.ensure_directories()
    assert config.data_dir.is_dir()
    assert config.cache_dir.is_dir()
    assert config.log_dir.is_dir()

