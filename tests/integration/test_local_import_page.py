from pathlib import Path

from cortaflow.ui.pages import import_page
from cortaflow.ui.pages.import_page import ImportPage


def test_local_import_runs_probe_off_the_ui_thread(qtbot, monkeypatch) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "vídeo teste.mp4"
    monkeypatch.setattr(
        import_page.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(fixture), "Vídeos (*.mp4)"),
    )
    page = ImportPage()
    qtbot.addWidget(page)
    with qtbot.waitSignal(page.media_selected, timeout=5_000) as emitted:
        page.choose_local_file()
        assert not page.local_button.isEnabled()
    metadata = emitted.args[0]
    assert metadata.local_path == fixture.resolve()
    assert metadata.duration_seconds == 2.0
    assert page.local_button.isEnabled()
    assert "vídeo teste.mp4" in page.transfer_status.text()
