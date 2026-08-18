import json
from pathlib import Path

import pytest

from cortaflow.infrastructure.database import (
    enqueue_task,
    get_setting,
    initialize_database,
    list_project_history,
    record_project_history,
    recover_interrupted_tasks,
    set_setting,
    update_task_status,
)


def test_database_schema_is_created(tmp_path) -> None:
    connection = initialize_database(tmp_path / "dados" / "cortaflow.db")
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    connection.close()
    assert {"project_history", "task_queue", "settings"} <= tables


def test_history_is_updated_without_duplicate_rows(tmp_path: Path) -> None:
    connection = initialize_database(tmp_path / "cortaflow.db")
    project_path = tmp_path / "projeto com espaço.cortaflow.json"
    record_project_history(connection, project_path, "Primeiro nome")
    record_project_history(connection, project_path, "Nome atualizado")

    history = list_project_history(connection)
    connection.close()

    assert len(history) == 1
    assert history[0]["display_name"] == "Nome atualizado"
    assert history[0]["project_path"] == str(project_path.resolve())


def test_task_queue_payload_and_status_are_persisted(tmp_path: Path) -> None:
    connection = initialize_database(tmp_path / "cortaflow.db")
    task_id = enqueue_task(connection, "export", {"destination": "Vídeos/corte.mp4"})
    update_task_status(connection, task_id, "running")
    row = connection.execute(
        "SELECT kind, payload_json, status FROM task_queue WHERE id=?",
        (task_id,),
    ).fetchone()
    connection.close()

    assert row is not None
    assert row[0] == "export"
    assert json.loads(row[1]) == {"destination": "Vídeos/corte.mp4"}
    assert row[2] == "running"


def test_task_queue_rejects_unknown_status(tmp_path: Path) -> None:
    connection = initialize_database(tmp_path / "cortaflow.db")
    task_id = enqueue_task(connection, "export", {})
    with pytest.raises(ValueError):
        update_task_status(connection, task_id, "unknown")
    connection.close()


def test_interrupted_tasks_and_settings_are_recovered(tmp_path: Path) -> None:
    connection = initialize_database(tmp_path / "cortaflow.db")
    task_id = enqueue_task(connection, "export", {})
    update_task_status(connection, task_id, "running")
    set_setting(connection, "max_concurrent_tasks", 2)

    assert recover_interrupted_tasks(connection) == 1
    status = connection.execute(
        "SELECT status FROM task_queue WHERE id=?",
        (task_id,),
    ).fetchone()[0]
    assert status == "failed"
    assert get_setting(connection, "max_concurrent_tasks", 1) == 2
    assert get_setting(connection, "missing", "fallback") == "fallback"
    connection.close()
