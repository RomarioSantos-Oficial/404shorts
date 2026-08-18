import logging
from pathlib import Path

from cortaflow.logging_config import configure_logging


def test_logging_creates_file(tmp_path: Path) -> None:
    path = configure_logging(tmp_path)
    logging.getLogger("test").info("mensagem segura")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert path.exists()
    assert "mensagem segura" in path.read_text(encoding="utf-8")

