"""Local SQLite history and task queue."""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path


def initialize_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS project_history (
            id INTEGER PRIMARY KEY, project_path TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL, last_opened_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_queue (
            id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', created_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
    """)
    connection.commit()
    return connection


def record_project_history(connection: sqlite3.Connection, project_path: Path, display_name: str) -> None:
    connection.execute(
        """
        INSERT INTO project_history(project_path, display_name, last_opened_utc)
        VALUES (?, ?, ?)
        ON CONFLICT(project_path) DO UPDATE SET
            display_name=excluded.display_name,
            last_opened_utc=excluded.last_opened_utc
        """,
        (str(project_path.resolve()), display_name, datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()


def list_project_history(connection: sqlite3.Connection, limit: int = 100) -> list[dict[str, str]]:
    rows = connection.execute(
        "SELECT project_path, display_name, last_opened_utc FROM project_history ORDER BY last_opened_utc DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"project_path": row[0], "display_name": row[1], "last_opened_utc": row[2]}
        for row in rows
    ]


def enqueue_task(connection: sqlite3.Connection, kind: str, payload: dict) -> int:
    cursor = connection.execute(
        "INSERT INTO task_queue(kind, payload_json, status, created_utc) VALUES (?, ?, 'pending', ?)",
        (kind, json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    return int(cursor.lastrowid)


def update_task_status(connection: sqlite3.Connection, task_id: int, status: str) -> None:
    if status not in {"pending", "running", "completed", "failed", "cancelled"}:
        raise ValueError("Status de tarefa inválido.")
    connection.execute("UPDATE task_queue SET status=? WHERE id=?", (status, task_id))
    connection.commit()


def recover_interrupted_tasks(connection: sqlite3.Connection) -> int:
    """Mark tasks left running by an interrupted process as failed."""
    cursor = connection.execute(
        "UPDATE task_queue SET status='failed' WHERE status='running'"
    )
    connection.commit()
    return cursor.rowcount


def set_setting(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        """
        INSERT INTO settings(key, value_json) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
        """,
        (key, json.dumps(value, ensure_ascii=False)),
    )
    connection.commit()


def get_setting(connection: sqlite3.Connection, key: str, default: object = None) -> object:
    row = connection.execute(
        "SELECT value_json FROM settings WHERE key=?",
        (key,),
    ).fetchone()
    return json.loads(row[0]) if row else default
