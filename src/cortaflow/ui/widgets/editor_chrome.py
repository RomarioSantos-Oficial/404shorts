"""Professional editor chrome widgets.

These widgets intentionally contain presentation only. Editing commands remain in
``EditorPage`` so the visual redesign does not replace the existing domain or
service layer.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class EditorToolRail(QFrame):
    """Compact CapCut-inspired tool rail for the complete editor workspace."""

    tool_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("editorToolRail")
        self.setMinimumWidth(82)
        self.setMaximumWidth(96)
        self.buttons: dict[str, QToolButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(6)

        brand = QLabel("CF")
        brand.setObjectName("editorBrandMark")
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(brand)

        label = QLabel("EDITAR")
        label.setObjectName("editorRailLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addSpacing(6)

        tools = (
            ("media", "Mídia", "▧"),
            ("audio", "Áudio", "♫"),
            ("text", "Texto", "T"),
            ("captions", "Legendas", "CC"),
            ("effects", "Efeitos", "✦"),
            ("transitions", "Transições", "◈"),
            ("image", "Imagem", "▣"),
        )
        for key, title, glyph in tools:
            button = self._create_tool_button(key, title, glyph)
            layout.addWidget(button)

        layout.addStretch(1)

        ai_button = self._create_tool_button("ai", "IA", "AI")
        ai_button.setObjectName("editorAiButton")
        layout.addWidget(ai_button)

        help_label = QLabel("F1 ajuda")
        help_label.setObjectName("editorRailHint")
        help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(help_label)

    def _create_tool_button(self, key: str, title: str, glyph: str) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("editorToolButton")
        button.setText(f"{glyph}\n{title}")
        button.setToolTip(title)
        button.setCheckable(True)
        button.setAutoExclusive(True)
        button.setMinimumHeight(52)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda _checked=False, name=key: self._request(name))
        self.buttons[key] = button
        return button

    def _request(self, key: str) -> None:
        button = self.buttons.get(key)
        if button is not None:
            button.setChecked(True)
        self.tool_requested.emit(key)

    def activate(self, key: str) -> None:
        button = self.buttons.get(key)
        if button is not None:
            button.setChecked(True)


class EditorSectionHeader(QFrame):
    """Small header used to give the workspace a clear professional hierarchy."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("editorSectionHeader")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("editorSectionTitle")
        layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("editorSectionSubtitle")
            layout.addWidget(subtitle_label)


__all__ = ["EditorToolRail", "EditorSectionHeader"]

