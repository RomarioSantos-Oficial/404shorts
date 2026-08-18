from cortaflow.domain.editing import TimelineClip
from cortaflow.domain.subtitle import SubtitleCue, TranscriptWord
from cortaflow.services.export_service import _timeline_subtitle_track


def test_timeline_subtitles_follow_source_trims_and_timeline_positions() -> None:
    words = [
        TranscriptWord(text="primeiro", start_ms=100, end_ms=500),
        TranscriptWord(text="segundo", start_ms=1100, end_ms=1500),
    ]
    cues = [SubtitleCue(start_ms=100, end_ms=1500, text="primeiro segundo")]
    clips = [
        TimelineClip(clip_id="v2", track="video", source_start_ms=1000, source_end_ms=1800, timeline_start_ms=0),
        TimelineClip(clip_id="v1", track="video", source_start_ms=0, source_end_ms=800, timeline_start_ms=1200),
    ]

    output_cues, output_words = _timeline_subtitle_track(cues, words, clips, 7)

    assert [(word.text, word.start_ms) for word in output_words] == [
        ("segundo", 100),
        ("primeiro", 1300),
    ]
    assert [cue.text for cue in output_cues] == ["segundo", "primeiro"]
