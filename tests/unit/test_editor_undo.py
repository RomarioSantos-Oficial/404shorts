from cortaflow.domain.editing import AudioSettings, ReframeSettings, SubtitleStyle
from cortaflow.domain.project import ExportSettings
from cortaflow.services.editor_operations import create_initial_clips
from cortaflow.ui.pages.editor_page import EditorPage


def set_state(page: EditorPage, clips) -> None:
    page.set_project_editor_state(
        clips, None, [], [], [], ReframeSettings(), SubtitleStyle(), AudioSettings(), ExportSettings()
    )


def test_move_delete_undo_and_redo(qtbot) -> None:
    page = EditorPage()
    qtbot.addWidget(page)
    clips = create_initial_clips(10_000)
    set_state(page, clips)
    selected = clips[0]
    page._select_clip(selected.clip_id)

    page.move_selected_clip(selected.clip_id, 2_000)
    assert page.undo_stack.canUndo()
    assert next(item for item in page.timeline_clips if item.clip_id == selected.clip_id).timeline_start_ms == 2_000
    page.undo_stack.undo()
    assert next(item for item in page.timeline_clips if item.clip_id == selected.clip_id).timeline_start_ms == 0
    page.undo_stack.redo()
    assert next(item for item in page.timeline_clips if item.clip_id == selected.clip_id).timeline_start_ms == 2_000
    page.delete_selected_clip()
    assert all(item.clip_id != selected.clip_id for item in page.timeline_clips)
    page.undo_stack.undo()
    assert any(item.clip_id == selected.clip_id for item in page.timeline_clips)


def test_manual_keyframe_is_emitted(qtbot) -> None:
    page = EditorPage()
    qtbot.addWidget(page)
    page.source_width, page.source_height = 1920, 1080
    page.properties.crop_x.setValue(500)
    with qtbot.waitSignal(page.reframe_keyframes_changed, timeout=1_000) as emitted:
        page.add_manual_keyframe()
    keyframe = emitted.args[0][0]
    assert keyframe.manual
    assert keyframe.crop.width == 608
    assert keyframe.crop.x == 500
