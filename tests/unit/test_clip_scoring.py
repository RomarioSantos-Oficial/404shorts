from cortaflow.domain.subtitle import Transcript, TranscriptWord
from cortaflow.domain.analysis import ClipSelectionSettings, TimeRange
from cortaflow.services.clip_scoring import ClipRanker, suggest_clips


def test_suggestions_align_to_complete_sentences_and_do_not_duplicate() -> None:
    words = []
    for index in range(120):
        text = f"palavra{index}"
        if index % 20 == 19: text += "."
        words.append(TranscriptWord(text=text, start_ms=index * 500, end_ms=(index + 1) * 500))
    suggestions = suggest_clips(Transcript(language="pt", words=words), 60_000)
    assert suggestions
    assert all(item.transcript_excerpt.endswith(".") for item in suggestions)
    assert all(item.end_ms > item.start_ms for item in suggestions)
    for index, left in enumerate(suggestions):
        for right in suggestions[index + 1:]:
            overlap = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
            assert overlap == 0


def test_scene_alignment_improves_candidate_score() -> None:
    words = []
    for index in range(60):
        text = f"palavra{index}"
        if index % 20 == 19:
            text += "."
        words.append(TranscriptWord(text=text, start_ms=index * 500, end_ms=(index + 1) * 500))
    transcript = Transcript(language="pt", words=words)
    plain = suggest_clips(transcript, 30_000)
    aligned = suggest_clips(
        transcript,
        30_000,
        scenes=[TimeRange(start_ms=0, end_ms=10_000), TimeRange(start_ms=10_000, end_ms=30_000)],
    )
    assert plain and aligned
    assert aligned[0].quality_score > plain[0].quality_score
    assert "cena" in aligned[0].reason


def test_configurable_duration_range_is_respected_and_explained() -> None:
    words = [
        TranscriptWord(
            text=f"palavra{index}{'.' if index % 10 == 9 else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(200)
    ]
    settings = ClipSelectionSettings(
        min_seconds=20,
        preferred_seconds=25,
        max_seconds=30,
        max_results=5,
    )
    suggestions = suggest_clips(
        Transcript(language="pt", words=words),
        100_000,
        settings=settings,
    )
    assert suggestions
    assert len(suggestions) <= 5
    assert all(20_000 <= item.duration_ms <= 30_000 for item in suggestions)
    assert all(item.quality_score <= 0.83 for item in suggestions)
    assert all(
        {
            "fala", "densidade", "duração", "limites", "hook", "fluxo", "valor",
            "emoção", "energia", "cena", "rosto",
        }
        <= item.score_components.keys()
        for item in suggestions
    )


def test_selection_settings_enforce_global_five_to_179_second_limits() -> None:
    assert ClipSelectionSettings().min_seconds == 5
    assert ClipSelectionSettings().max_seconds == 179
    import pytest
    with pytest.raises(ValueError):
        ClipSelectionSettings(min_seconds=4)
    with pytest.raises(ValueError):
        ClipSelectionSettings(min_seconds=60, preferred_seconds=45, max_seconds=90)


def test_every_natural_end_is_generated_before_replaceable_ranking() -> None:
    words = [
        TranscriptWord(
            text=f"ideia{index}{'.' if index % 10 == 9 else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(120)
    ]
    updates: list[dict] = []

    class RecordingRanker:
        received = 0

        def rank(self, candidates, max_results, *, progress=None, cancelled=None):
            self.received = len(candidates)
            return candidates[:max_results]

    ranker: ClipRanker = RecordingRanker()
    suggest_clips(
        Transcript(language="pt", words=words),
        60_000,
        settings=ClipSelectionSettings(min_seconds=5, preferred_seconds=20, max_seconds=30),
        ranker=ranker,
        progress=updates.append,
    )
    generated = next(item["candidate_count"] for item in updates if "candidate_count" in item)
    assert generated > 20
    assert 1 <= ranker.received <= 24
    assert ranker.received < generated


def test_auto_accept_threshold_preserves_preview_status_on_lower_scores() -> None:
    words = [
        TranscriptWord(
            text=f"Como resolver problema {index}{'.' if index % 10 == 9 else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(80)
    ]
    suggestions = suggest_clips(
        Transcript(language="pt", words=words),
        40_000,
        settings=ClipSelectionSettings(
            min_seconds=5,
            preferred_seconds=15,
            max_seconds=30,
            auto_accept_threshold=0.6,
        ),
    )
    assert suggestions
    assert all(
        item.status == ("accepted" if item.quality_score >= 0.6 else "pending")
        for item in suggestions
    )
    assert all(item.editorial_status == "validated" for item in suggestions)


def test_visual_scene_boundary_never_authorizes_a_mid_speech_cut() -> None:
    words = [
        TranscriptWord(
            text=f"fala{index}{'.' if index == 79 else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(80)
    ]
    suggestions = suggest_clips(
        Transcript(language="pt", words=words),
        40_000,
        scenes=[TimeRange(start_ms=0, end_ms=10_000), TimeRange(start_ms=10_000, end_ms=40_000)],
        settings=ClipSelectionSettings(min_seconds=5, preferred_seconds=10, max_seconds=20),
    )
    assert suggestions == []


def test_oversized_discourse_is_resegmented_at_complete_word_boundaries() -> None:
    words = [
        TranscriptWord(
            text=f"ideia{index}{'.' if index % 30 == 29 else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(440)
    ]
    suggestions = suggest_clips(
        Transcript(language="pt", words=words),
        220_000,
        settings=ClipSelectionSettings(
            min_seconds=30,
            preferred_seconds=60,
            max_seconds=179,
            max_results=5,
        ),
    )
    assert suggestions
    assert all(item.duration_ms <= 179_000 for item in suggestions)
    assert all(item.transcript_excerpt.endswith((".", "!")) for item in suggestions)
    assert all(item.resegmented_from_long_unit for item in suggestions)
    assert all(item.editorial_status == "needs_review" for item in suggestions)
    assert all("acima de 179 s" in item.reason for item in suggestions)


def test_question_boundary_requires_review_and_is_never_auto_accepted() -> None:
    words = [
        TranscriptWord(
            text=f"palavra{index}{'?' if index % 20 == 19 else ''}",
            start_ms=index * 500,
            end_ms=(index + 1) * 500,
        )
        for index in range(80)
    ]
    suggestions = suggest_clips(
        Transcript(language="pt", words=words),
        40_000,
        settings=ClipSelectionSettings(
            min_seconds=5,
            preferred_seconds=15,
            max_seconds=30,
            auto_accept_threshold=0,
        ),
    )
    assert suggestions
    assert all(item.editorial_status == "needs_review" for item in suggestions)
    assert all(item.status == "pending" for item in suggestions)
    assert all("pergunta final" in item.reason for item in suggestions)
