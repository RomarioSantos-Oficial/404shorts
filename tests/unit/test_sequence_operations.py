from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.domain.editing import TimelineClip
from cortaflow.services.sequence_operations import (
    create_image_layer,
    create_sequence_from_suggestion,
    create_text_layer,
    delete_layer,
    ripple_delete,
    trim_clip_side,
    update_layer,
)


def test_suggestion_becomes_editable_audio_video_sequence() -> None:
    suggestion = ClipSuggestion(
        start_ms=12_000,
        end_ms=42_000,
        title="Gancho do jogo",
        transcript_excerpt="A análise começa aqui.",
        quality_score=0.9,
        reason="Ideia completa.",
    )

    sequence = create_sequence_from_suggestion(suggestion)

    assert sequence.duration_ms == 30_000
    assert {clip.track for clip in sequence.clips} == {"video", "audio"}
    assert sequence.clips[0].source_start_ms == 12_000
    assert sequence.clips[0].timeline_start_ms == 0
    assert sequence.suggested_start_ms == 12_000
    assert sequence.suggested_end_ms == 42_000


def test_trim_preserves_source_mapping_and_ripple_delete_closes_gap() -> None:
    clips = [
        TimelineClip(
            clip_id="a",
            track="video",
            source_start_ms=0,
            source_end_ms=10_000,
            timeline_start_ms=0,
        ),
        TimelineClip(
            clip_id="b",
            track="video",
            source_start_ms=20_000,
            source_end_ms=30_000,
            timeline_start_ms=10_000,
        ),
    ]

    left_trimmed = trim_clip_side(clips, "a", side="left", timeline_position_ms=2_000)
    assert left_trimmed[0].timeline_start_ms == 2_000
    assert left_trimmed[0].source_start_ms == 2_000
    right_trimmed = trim_clip_side(clips, "a", side="right", timeline_position_ms=7_000)
    assert right_trimmed[0].source_end_ms == 7_000

    compacted = ripple_delete(clips, "a")
    assert [clip.clip_id for clip in compacted] == ["b"]
    assert compacted[0].timeline_start_ms == 0


def test_text_and_image_layers_are_editable_and_persistable() -> None:
    suggestion = ClipSuggestion(
        start_ms=0,
        end_ms=15_000,
        title="Corte",
        transcript_excerpt="Texto",
        quality_score=0.8,
        reason="Contexto",
    )
    sequence = create_sequence_from_suggestion(suggestion)
    text = create_text_layer(sequence, "Palmeiras em debate")
    image = create_image_layer(sequence, "/tmp/logo.png")

    updated = update_layer(sequence, text.item_id, font_size=72, x_percent=50, opacity=0.9)
    assert updated.layers[0].font_size == 72
    assert any(layer.item_id == image.item_id for layer in updated.layers)

    removed = delete_layer(updated, image.item_id)
    assert all(layer.item_id != image.item_id for layer in removed.layers)
    assert removed.dirty
