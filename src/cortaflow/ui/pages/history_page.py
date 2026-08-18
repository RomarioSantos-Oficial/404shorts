"""Recent local projects backed by SQLite."""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from cortaflow.config import AppConfig
from cortaflow.infrastructure.database import initialize_database, list_project_history


class HistoryPage(QWidget):
    open_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Projeto", "Caminho", "Última abertura (UTC)"])
        layout.addWidget(self.table)
        row = QHBoxLayout()
        refresh = QPushButton("Atualizar")
        refresh.clicked.connect(self.refresh)
        open_button = QPushButton("Abrir selecionado")
        open_button.clicked.connect(self.open_selected)
        row.addWidget(refresh)
        row.addWidget(open_button)
        row.addStretch()
        layout.addLayout(row)

    def refresh(self) -> None:
        config = AppConfig()
        connection = initialize_database(config.data_dir / "cortaflow.db")
        try:
            entries = list_project_history(connection)
        finally:
            connection.close()
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            for column, value in enumerate(
                (entry["display_name"], entry["project_path"], entry["last_opened_utc"])
            ):
                self.table.setItem(row, column, QTableWidgetItem(value))

    def open_selected(self) -> None:
        row = self.table.currentRow()
        if row >= 0 and self.table.item(row, 1):
            self.open_requested.emit(Path(self.table.item(row, 1).text()))
