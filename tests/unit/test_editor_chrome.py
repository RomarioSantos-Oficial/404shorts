from cortaflow.ui.widgets.editor_chrome import EditorToolRail


def test_editor_tool_rail_exposes_connected_professional_tools(qtbot) -> None:
    rail = EditorToolRail()
    qtbot.addWidget(rail)
    requested = []
    rail.tool_requested.connect(requested.append)

    expected = {"media", "audio", "text", "captions", "effects", "transitions", "image", "ai"}
    assert expected.issubset(rail.buttons)

    rail.buttons["text"].click()
    assert requested == ["text"]
    assert rail.buttons["text"].isChecked()

    rail.buttons["effects"].click()
    assert requested[-1] == "effects"
    assert rail.buttons["effects"].isChecked()
    assert not rail.buttons["text"].isChecked()
