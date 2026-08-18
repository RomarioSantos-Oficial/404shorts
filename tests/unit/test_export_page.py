from pathlib import Path

from PySide6.QtGui import QColor, QImage

from cortaflow.domain.project import ExportSettings
from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.ui.pages import export_page
from cortaflow.ui.pages.export_page import ExportPage, _progress_microseconds


def test_ffmpeg_na_progress_is_treated_as_zero() -> None:
    assert _progress_microseconds({"out_time_us": "N/A"}) == 0
    assert _progress_microseconds({"out_time_us": "N/A", "out_time_ms": "500000"}) == 500_000
    assert _progress_microseconds({"out_time_us": "invalid"}) == 0


def test_standalone_preview_validation_does_not_create_a_duplicate_export_flow(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")

    class FakeConfig:
        cache_dir = tmp_path / "cache"
        data_dir = tmp_path / "data"

    def fake_render(
        source, destination, settings, clip, cues, style, keyframes, preview,
        words, reframe, audio, source_size, timeline, progress, cancelled,
    ):
        progress({"out_time_us": "500000", "fps": "30", "speed": "1x", "progress": "continue", "encoder": "libx264"})
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"preview" if preview else b"final")
        return destination

    monkeypatch.setattr(export_page, "AppConfig", FakeConfig)
    monkeypatch.setattr(export_page, "render_project_export", fake_render)
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(source, 1000, ExportSettings(use_nvenc=False), [], page.subtitle_style, [], [])
    page.generate_preview()
    qtbot.waitUntil(lambda: page.current_worker is None, timeout=5_000)
    assert page.preview_path and page.preview_path.is_file()
    page.approve_preview()
    assert not page.batch_button.isEnabled()
    assert [job.status for job in page.jobs] == ["completed"]


def test_failed_job_does_not_block_next_queue_item(qtbot, monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    calls = 0

    class FakeConfig:
        cache_dir = tmp_path / "cache"
        data_dir = tmp_path / "data"

    def fake_render(
        source, destination, settings, clip, cues, style, keyframes, preview,
        words, reframe, audio, source_size, timeline, progress, cancelled,
    ):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("falha simulada")
        destination.write_bytes(b"ok")
        return destination

    monkeypatch.setattr(export_page, "render_project_export", fake_render)
    monkeypatch.setattr(export_page, "AppConfig", FakeConfig)
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(source, 1000, ExportSettings(), [], page.subtitle_style, [], [])
    clip = page._selected_clip()
    assert clip is not None
    page._enqueue_final_job(tmp_path / "one.mp4", clip)
    page._enqueue_final_job(tmp_path / "two.mp4", clip)
    qtbot.waitUntil(lambda: page.current_worker is None and not page.queue, timeout=5_000)
    assert [job.status for job in page.jobs] == ["failed", "completed"]
    assert (tmp_path / "two.mp4").is_file()
    connection = export_page.initialize_database(tmp_path / "data" / "cortaflow.db")
    statuses = [row[0] for row in connection.execute("SELECT status FROM task_queue ORDER BY id")]
    connection.close()
    assert statuses == ["failed", "completed"]


def test_watermark_controls_are_persisted_and_sent_to_preview(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    image = QImage(20, 10, QImage.Format.Format_ARGB32)
    image.fill(QColor("red"))
    watermark = tmp_path / "minha marca.png"
    assert image.save(str(watermark))
    captured = []

    class FakeConfig:
        cache_dir = tmp_path / "cache"
        data_dir = tmp_path / "data"

    def fake_render(
        source, destination, settings, clip, cues, style, keyframes, preview,
        words, reframe, audio, source_size, timeline, progress, cancelled,
    ):
        captured.append(settings.watermark)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"preview")
        return destination

    monkeypatch.setattr(export_page, "AppConfig", FakeConfig)
    monkeypatch.setattr(export_page, "render_project_export", fake_render)
    monkeypatch.setattr(
        export_page.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(watermark), "Imagens (*.png)"),
    )
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(source, 1_000, ExportSettings(), [], page.subtitle_style, [], [])

    page.choose_watermark()
    page.watermark_position.setCurrentIndex(page.watermark_position.findData("custom"))
    page.watermark_width.setValue(24)
    page.watermark_opacity.setValue(0.55)
    page.watermark_x.setValue(25)
    page.watermark_y.setValue(75)

    assert page.settings.watermark.enabled
    assert page.settings.watermark.image_path == watermark.resolve()
    assert page.settings.watermark.position == "custom"
    assert page.watermark_x.isEnabled()
    assert page.preview_stack.currentWidget() is page.watermark_overlay
    assert not page.watermark_overlay.isHidden()
    page.generate_preview()
    qtbot.waitUntil(lambda: page.current_worker is None, timeout=5_000)
    assert captured
    assert captured[-1].width_percent == 24
    assert captured[-1].opacity == 0.55
    assert captured[-1].custom_x_percent == 25
    assert captured[-1].custom_y_percent == 75


def test_accepted_cuts_require_preview_then_choose_final_folder(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    final_folder = tmp_path / "finais"
    final_folder.mkdir()
    suggestion = ClipSuggestion(
        start_ms=1_000,
        end_ms=6_000,
        title="Melhor trecho",
        transcript_excerpt="Ideia completa.",
        quality_score=0.9,
        reason="Boa abertura.",
        status="accepted",
    )

    class FakeConfig:
        cache_dir = tmp_path / "cache"
        data_dir = tmp_path / "data"

    def fake_render(
        source, destination, settings, clip, cues, style, keyframes, preview,
        words, reframe, audio, source_size, timeline, progress, cancelled,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"preview" if preview else b"final-1080")
        return destination

    monkeypatch.setattr(export_page, "AppConfig", FakeConfig)
    monkeypatch.setattr(export_page, "render_project_export", fake_render)
    monkeypatch.setattr(
        export_page.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(final_folder),
    )
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(
        source, 10_000, ExportSettings(width=1080, height=1920),
        [], page.subtitle_style, [], [suggestion],
    )

    assert page.prepare_accepted()
    assert page.start_seconds.value() == 1
    assert page.end_seconds.value() == 6
    assert not page.batch_button.isEnabled()
    page.generate_preview()
    qtbot.waitUntil(lambda: page.current_worker is None, timeout=5_000)
    page.approve_preview()
    assert page.batch_button.isEnabled()
    page.enqueue_accepted()
    qtbot.waitUntil(lambda: page.current_worker is None and not page.queue, timeout=5_000)
    final = final_folder / "01-Melhor trecho.mp4"
    assert final.read_bytes() == b"final-1080"
    assert page.jobs[-1].settings.width == 1080
    assert not page.jobs[-1].preview


def test_reviewing_one_suggestion_saves_exact_cut_without_batch_acceptance_or_second_preview(
    qtbot, monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    final_folder = tmp_path / "escolhida"
    final_folder.mkdir()
    suggestion = ClipSuggestion(
        start_ms=2_500,
        end_ms=8_750,
        title="Corte: principal?",
        transcript_excerpt="Trecho completo.",
        quality_score=.91,
        reason="Gancho forte.",
    )

    class FakeConfig:
        cache_dir = tmp_path / "cache"
        data_dir = tmp_path / "data"

    def fake_render(
        source, destination, settings, clip, cues, style, keyframes, preview,
        words, reframe, audio, source_size, timeline, progress, cancelled,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"preview" if preview else b"exact-final")
        return destination

    monkeypatch.setattr(export_page, "AppConfig", FakeConfig)
    monkeypatch.setattr(export_page, "render_project_export", fake_render)
    monkeypatch.setattr(
        export_page.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(final_folder),
    )
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(source, 10_000, ExportSettings(), [], page.subtitle_style, [], [suggestion])
    assert page.prepare_suggestion(suggestion)

    with qtbot.waitSignal(page.suggestion_saved, timeout=5_000) as saved:
        page.approve_preview()
    destination = final_folder / "Corte_ principal_.mp4"

    assert saved.args[0] == suggestion
    assert destination.read_bytes() == b"exact-final"
    assert page.jobs[-1].clip.start_ms == 2_500
    assert page.jobs[-1].clip.end_ms == 8_750
    assert not page.jobs[-1].preview
    assert not page.jobs[-1].use_timeline


def test_individual_review_stays_on_exact_range_after_watermark_settings_sync(
    qtbot, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    watermark = tmp_path / "logo.png"
    image = QImage(20, 10, QImage.Format.Format_ARGB32)
    image.fill(QColor("red"))
    assert image.save(str(watermark))
    suggestion = ClipSuggestion(
        start_ms=2_500,
        end_ms=8_750,
        title="Corte exato",
        transcript_excerpt="Trecho.",
        quality_score=.9,
        reason="Completo.",
    )
    timeline = [
        export_page.TimelineClip(
            clip_id="video-1",
            track="video",
            source_start_ms=0,
            source_end_ms=10_000,
            timeline_start_ms=0,
        )
    ]
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(
        source,
        10_000,
        ExportSettings(),
        [],
        page.subtitle_style,
        [],
        [suggestion],
        timeline_clips=timeline,
        source_size=(1920, 1080),
    )
    assert page.prepare_suggestion(suggestion)
    settings = ExportSettings(
        watermark=export_page.WatermarkSettings(
            enabled=True,
            image_path=watermark,
            position="top-right",
        )
    )

    page.set_context(
        source,
        10_000,
        settings,
        [],
        page.subtitle_style,
        [],
        [suggestion],
        timeline_clips=timeline,
        source_size=(1920, 1080),
    )

    assert not page.timeline_mode.isChecked()
    assert not page.timeline_mode.isEnabled()
    assert page.start_seconds.value() == 2.5
    assert page.end_seconds.value() == 8.75
    assert page.approve_button.isEnabled()
    assert not page.watermark_overlay.watermark_rect().isEmpty()


def test_review_subtitle_edit_is_emitted_and_invalidates_preview(qtbot, tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    cue = export_page.SubtitleCue(start_ms=1_000, end_ms=3_000, text="Texto antigo")
    page = ExportPage()
    qtbot.addWidget(page)
    page.set_context(source, 5_000, ExportSettings(), [cue], page.subtitle_style, [], [])
    page.start_seconds.setValue(1)
    page.end_seconds.setValue(4)
    page.preview_path = tmp_path / "preview.mp4"

    with qtbot.waitSignal(page.cues_changed, timeout=1_000) as changed:
        page.review_subtitles.item(0, 2).setText("Texto corrigido")

    assert changed.args[0][0].text == "Texto corrigido"
    assert changed.args[0][0].manually_edited
    assert page.preview_path is None
