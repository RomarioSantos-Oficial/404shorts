from cortaflow.domain.editing import AudioSettings, ReframeSettings, SubtitleStyle
from pathlib import Path

from cortaflow.domain.project import ExportSettings, WatermarkSettings
from cortaflow.ui.widgets.properties_panel import PropertiesPanel


def test_properties_emit_valid_persistent_settings(qtbot) -> None:
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    panel.set_settings(
        ReframeSettings(),
        SubtitleStyle(),
        AudioSettings(),
        ExportSettings(),
    )
    with qtbot.waitSignal(panel.settings_changed, timeout=1_000) as emitted:
        panel.font_size.setValue(70)
    reframe, subtitle, audio, export = emitted.args[0]
    assert subtitle.font_size == 70
    assert reframe.aspect_ratio == "9:16"
    assert audio.volume == 1
    assert export.codec == "h264"


def test_panel_has_all_six_property_tabs(qtbot) -> None:
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    assert [panel.tabs.tabText(index) for index in range(panel.tabs.count())] == [
        "Corte", "Enquadramento", "Rosto", "Legenda", "Áudio", "Exportação"
    ]


def test_editor_export_change_preserves_watermark_settings(qtbot) -> None:
    panel = PropertiesPanel()
    qtbot.addWidget(panel)
    watermark = WatermarkSettings(
        enabled=True,
        image_path=Path("C:/logo.png"),
        position="top-left",
    )
    panel.set_settings(
        ReframeSettings(), SubtitleStyle(), AudioSettings(),
        ExportSettings(watermark=watermark),
    )
    with qtbot.waitSignal(panel.settings_changed, timeout=1_000) as emitted:
        panel.quality.setValue(18)
    assert emitted.args[0][3].watermark == watermark
