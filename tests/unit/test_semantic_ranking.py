import json
from pathlib import Path

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.services.semantic_models import OllamaAssets, SemanticAssets
from cortaflow.services.semantic_ranking import OllamaClipRanker, QwenClipRanker


def _candidate(index: int, score: float) -> ClipSuggestion:
    start = index * 20_000
    topic = ("astronomia", "culinária", "programação")[index]
    return ClipSuggestion(
        start_ms=start,
        end_ms=start + 15_000,
        title=f"Heurística {index}",
        transcript_excerpt=f"Como entender {topic}? Esta explicação traz uma conclusão útil.",
        quality_score=score,
        reason="Heurística local.",
        score_components={"hook": 0.7, "fluxo": 0.8},
        editorial_status="validated",
        context_before="Contexto que veio antes.",
        context_after="Contexto que veio depois.",
    )


def _semantic(index: int, title: str, reason: str, **updates) -> dict:
    topic = ("astronomia", "culinária", "programação")[index % 3]
    payload = {
        "index": index,
        "topic_stated_in_clip": True,
        "opening_dependency": "none",
        "unresolved_references": [],
        "question_answer_complete": True,
        "ending_state": "complete",
        "after_continues_same_answer": False,
        "long_unit_importance": "important",
        "central_claim": "O trecho explica uma afirmação central completa.",
        "evidence_start": f"Como entender {topic}?",
        "evidence_end": "Esta explicação traz uma conclusão útil.",
        "completeness": 4,
        "relevance": 4,
        "hook": 3,
        "flow": 4,
        "value": 4,
        "emotion": 2,
        "shareability": 3,
        "novelty": 2,
        "title": title,
        "reason": reason,
    }
    payload.update(updates)
    return payload


def test_qwen_ranker_validates_json_combines_scores_and_safe_boundaries(monkeypatch) -> None:
    candidates = [_candidate(0, 0.7), _candidate(1, 0.8), _candidate(2, 0.6)]
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    response = {
        "rankings": [
            _semantic(2, "Melhor solução", "Tem hook, valor e conclusão."),
            _semantic(0, "Segunda opção", "Explica uma ideia completa.", relevance=3),
        ]
    }
    monkeypatch.setattr(ranker, "_execute", lambda prompt, schema, cancelled: json.dumps(response))
    result = ranker.rank(candidates, 2)
    assert len(result) == 2
    best = next(item for item in result if item.title == "Melhor solução")
    assert best.start_ms == 40_000
    assert best.end_ms == 55_000
    assert best.score_components["completude_semântica"] == 1.0
    assert best.editorial_status == "validated"
    assert best.relevance_score == 1.0
    assert best.confidence_score == 1.0
    unsafe = next(item for item in result if item.title == "Segunda opção")
    assert (unsafe.start_ms, unsafe.end_ms) == (0, 15_000)


def test_qwen_ranker_falls_back_to_heuristic_on_invalid_output(monkeypatch) -> None:
    candidates = [_candidate(0, 0.5), _candidate(1, 0.9)]
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    monkeypatch.setattr(ranker, "_execute", lambda prompt, schema, cancelled: "resposta inválida")
    updates: list[dict] = []
    result = ranker.rank(candidates, 1, progress=updates.append)
    assert [(item.start_ms, item.end_ms) for item in result] == [(20_000, 35_000)]
    assert result[0].editorial_status == "needs_review"
    assert result[0].confidence_score == 0
    assert ranker.last_error
    assert any(update["status"] == "semantic_fallback" for update in updates)


def test_unified_llama_executable_receives_cli_subcommand() -> None:
    unified = SemanticAssets(Path("llama.exe"), Path("model.gguf"))
    legacy = SemanticAssets(Path("llama-cli.exe"), Path("model.gguf"))
    assert unified.cli_command == ["llama.exe", "cli"]
    assert legacy.cli_command == ["llama-cli.exe"]


def test_ollama_ranker_uses_same_validated_semantic_contract(monkeypatch) -> None:
    candidates = [_candidate(0, 0.5), _candidate(1, 0.7)]
    ranker = OllamaClipRanker(OllamaAssets(Path("ollama.exe")))
    response = {
        "rankings": [
            _semantic(0, "Escolha do Ollama", "Hook forte e conclusão útil.")
        ]
    }
    monkeypatch.setattr(ranker, "_execute", lambda prompt, schema, cancelled: json.dumps(response))
    result = ranker.rank(candidates, 1)
    assert result[0].title == "Escolha do Ollama"
    assert result[0].score_components["semântica"] == result[0].score_components["potencial"]


def test_ollama_ranker_batches_large_candidate_sets(monkeypatch) -> None:
    candidates = [
        ClipSuggestion(
            start_ms=index * 20_000,
            end_ms=index * 20_000 + 15_000,
            title=f"Tema {index}",
            transcript_excerpt=f"Explicação específica tema{index} com conclusão própria.",
            quality_score=0.6 + index / 100,
            reason="Heurística.",
        )
        for index in range(12)
    ]
    ranker = OllamaClipRanker(OllamaAssets(Path("ollama.exe")))
    calls = 0

    def fake_execute(prompt, schema, cancelled):
        nonlocal calls
        starts = (
            (0, 40_000),
            (0, 40_000),
            (20_000, 60_000),
            (20_000, 60_000),
        )[calls]
        calls += 1
        return json.dumps(
            {
                "rankings": [
                    _semantic(
                        index,
                        f"Semântico {start}",
                        f"Evidência específica do trecho {start}.",
                    )
                    for index, start in enumerate(starts)
                ]
            }
        )

    monkeypatch.setattr(ranker, "_execute", fake_execute)
    result = ranker.rank(candidates, 4)
    assert calls == 4
    assert len(result) == 4
    assert all("semântica" in item.score_components for item in result)


def test_semantic_gate_rejects_unfinished_or_unimportant_long_excerpt(monkeypatch) -> None:
    normal = _candidate(0, 0.8)
    long_excerpt = _candidate(1, 0.9).model_copy(
        update={"resegmented_from_long_unit": True, "editorial_status": "needs_review"}
    )
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    response = {
        "rankings": [
            _semantic(
                0,
                "Fala incompleta",
                "O raciocínio continua depois.",
                ending_state="ongoing",
                after_continues_same_answer=True,
                completeness=2,
            ),
            _semantic(
                1,
                "Parte fraca da discussão",
                "É completa, mas não é a parte mais importante.",
                long_unit_importance="not_important",
            ),
        ]
    }
    monkeypatch.setattr(ranker, "_execute", lambda prompt, schema, cancelled: json.dumps(response))
    result = ranker.rank([normal, long_excerpt], 2)
    assert len(result) == 2
    assert all(item.editorial_status == "needs_review" for item in result)
    assert all(item.status == "pending" for item in result)


def test_semantic_prompt_includes_surrounding_context_and_never_changes_bounds(monkeypatch) -> None:
    candidate = _candidate(0, 0.7).model_copy(
        update={
            "selection_goal": "topic",
            "topic_prompt": "segurança em Python",
            "resegmented_from_long_unit": True,
        }
    )
    captured = ""
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))

    def fake_execute(prompt, schema, cancelled):
        nonlocal captured
        captured = prompt
        return json.dumps({"rankings": [_semantic(0, "Segurança", "Ideia completa e relevante.")]})

    monkeypatch.setattr(ranker, "_execute", fake_execute)
    result = ranker.rank([candidate], 1)
    assert result[0].start_ms == candidate.start_ms
    assert result[0].end_ms == candidate.end_ms
    assert "Contexto que veio antes" in captured
    assert "Contexto que veio depois" in captured
    assert "segurança em Python" in captured
    assert "editorial_valid" not in captured


def test_semantic_ranker_repairs_bounds_once_and_revalidates(monkeypatch) -> None:
    original = _candidate(1, 0.9)
    expanded = original.model_copy(
        update={
            "start_ms": 10_000,
            "transcript_excerpt": "Pergunta completa. " + original.transcript_excerpt,
            "quality_score": 0.8,
        }
    )
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    responses = [
        {
            "rankings": [
                _semantic(
                    0,
                    "Resposta sem pergunta",
                    "A abertura depende da pergunta anterior.",
                    opening_dependency="repairable",
                    question_answer_complete=False,
                )
            ]
        },
        {
            "rankings": [
                _semantic(
                    0,
                    "Resposta sem pergunta",
                    "A abertura depende da pergunta anterior.",
                    opening_dependency="repairable",
                    question_answer_complete=False,
                )
            ]
        },
        {
            "rankings": [
                _semantic(
                    0,
                    "Pergunta e resposta completas",
                    "A ampliação incluiu o assunto e a resposta completa.",
                )
            ]
        },
        {
            "rankings": [
                _semantic(
                    0,
                    "Pergunta e resposta completas",
                    "A ampliação incluiu o assunto e a resposta completa.",
                )
            ]
        },
    ]
    calls = 0

    def fake_execute(prompt, schema, cancelled):
        nonlocal calls
        response = responses[calls]
        calls += 1
        return json.dumps(response)

    monkeypatch.setattr(ranker, "_execute", fake_execute)
    result = ranker.rank([original, expanded], 1)
    assert calls == 4
    assert len(result) == 1
    assert (result[0].start_ms, result[0].end_ms) == (10_000, 35_000)
    assert result[0].editorial_status == "validated"


def test_dual_evaluation_reverses_order_and_calculates_confidence(monkeypatch) -> None:
    candidates = [_candidate(0, 0.7), _candidate(1, 0.8)]
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    prompts: list[str] = []
    response = {
        "rankings": [
            _semantic(0, "Astronomia", "Afirmação completa."),
            _semantic(1, "Culinária", "Afirmação completa."),
        ]
    }

    def fake_execute(prompt, schema, cancelled):
        prompts.append(prompt)
        return json.dumps(response)

    monkeypatch.setattr(ranker, "_execute", fake_execute)
    result = ranker.rank(candidates, 2)
    assert len(prompts) == 2
    assert prompts[0].index('"index":0') < prompts[0].index('"index":1')
    assert prompts[1].index('"index":1') < prompts[1].index('"index":0')
    assert all(item.confidence_score == 1 for item in result)


def test_dual_evaluation_disagreement_requires_review(monkeypatch) -> None:
    candidate = _candidate(0, 0.8)
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    responses = [
        {"rankings": [_semantic(0, "Astronomia", "Trecho completo.")]},
        {
            "rankings": [
                _semantic(
                    0,
                    "Astronomia",
                    "A abertura depende do contexto.",
                    topic_stated_in_clip=False,
                    opening_dependency="strong",
                    question_answer_complete=False,
                )
            ]
        },
    ]
    calls = 0

    def fake_execute(prompt, schema, cancelled):
        nonlocal calls
        response = responses[calls]
        calls += 1
        return json.dumps(response)

    monkeypatch.setattr(ranker, "_execute", fake_execute)
    result = ranker.rank([candidate], 1)
    assert result[0].editorial_status == "needs_review"
    assert result[0].confidence_score == 0.55


def test_explicit_topic_in_opening_corrects_small_model_false_negative(monkeypatch) -> None:
    candidate = ClipSuggestion(
        start_ms=0,
        end_ms=30_000,
        title="Flamengo",
        transcript_excerpt=(
            "Apareceu o Flamengo e ninguém pode dizer não ao Flamengo. "
            "O Flamengo é o maior clube da América do Sul e por isso eu aceitei voltar."
        ),
        quality_score=0.8,
        reason="Heurística.",
        editorial_score=1,
    )
    observation = _semantic(
        0,
        "A escolha pelo Flamengo",
        "A resposta explica por que ele aceitou voltar.",
        topic_stated_in_clip=False,
        opening_dependency="strong",
        question_answer_complete=False,
        central_claim="O Flamengo motivou a decisão de aceitar voltar ao Brasil.",
        evidence_start="Apareceu o Flamengo e ninguém pode dizer não ao Flamengo.",
        evidence_end="por isso eu aceitei voltar.",
    )
    ranker = QwenClipRanker(SemanticAssets(Path("llama.exe"), Path("model.gguf")))
    monkeypatch.setattr(
        ranker,
        "_execute",
        lambda prompt, schema, cancelled: json.dumps({"rankings": [observation]}),
    )
    result = ranker.rank([candidate], 1)
    assert result[0].editorial_status == "validated"
