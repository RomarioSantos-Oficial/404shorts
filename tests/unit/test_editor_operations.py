import pytest

from cortaflow.services.editor_operations import (
    create_initial_clips,
    delete_clip,
    move_clip,
    set_transition,
    split_at,
    trim_clip,
)


def test_split_move_trim_transition_and_delete() -> None:
    initial = create_initial_clips(10_000, "Fonte")
    assert {clip.track for clip in initial} == {"video", "audio"}
    split = split_at(initial, 4_000)
    assert len(split) == 4
    assert sorted(clip.duration_ms for clip in split) == [4000, 4000, 6000, 6000]

    selected = next(clip for clip in split if clip.track == "video" and clip.timeline_start_ms == 4000)
    moved = move_clip(split, selected.clip_id, 5_000)
    assert next(clip for clip in moved if clip.clip_id == selected.clip_id).timeline_start_ms == 5_000
    trimmed = trim_clip(moved, selected.clip_id, 5_000, 9_000)
    transitioned = set_transition(trimmed, selected.clip_id, 500)
    edited = next(clip for clip in transitioned if clip.clip_id == selected.clip_id)
    assert edited.duration_ms == 4_000
    assert edited.transition_ms == 500
    assert len(delete_clip(transitioned, selected.clip_id)) == 3


def test_invalid_trim_and_transition_are_rejected() -> None:
    clip = create_initial_clips(1000)[0]
    with pytest.raises(ValueError):
        trim_clip([clip], clip.clip_id, 900, 100)
    with pytest.raises(ValueError):
        set_transition([clip], clip.clip_id, 2000)
