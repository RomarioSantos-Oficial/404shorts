"""Optional local Qwen ranking through a verified llama.cpp runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
from threading import Event
import time
from typing import Any, Literal
import unicodedata
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field, ValidationError

from cortaflow.domain.analysis import ClipSuggestion
from cortaflow.services.clip_scoring import HeuristicClipRanker
from cortaflow.services.semantic_models import OllamaAssets, SemanticAssets


ProgressCallback = Callable[[dict[str, Any]], None]

_SEMANTIC_STOP_WORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos",
    "e", "ela", "ele", "em", "essa", "esse", "esta", "este", "eu", "foi",
    "isso", "mais", "mas", "me", "na", "nas", "no", "nos", "o", "os", "ou",
    "para", "pela", "pelo", "por", "pra", "que", "se", "sem", "seu", "sua",
    "um", "uma", "voce", "the", "and", "for", "that", "this", "with",
}


class SemanticRankingError(RuntimeError):
    """Raised when local semantic output cannot be safely used."""


class SemanticRankingCancelled(SemanticRankingError):
    """Raised after cooperative cancellation of llama.cpp."""


class SemanticCandidateScore(BaseModel):
    index: int = Field(ge=0)
    topic_stated_in_clip: bool
    opening_dependency: Literal["none", "repairable", "strong"]
    unresolved_references: list[str] = Field(default_factory=list, max_length=6)
    question_answer_complete: bool
    ending_state: Literal["complete", "repairable", "ongoing"]
    after_continues_same_answer: bool
    long_unit_importance: Literal["not_applicable", "important", "not_important"]
    central_claim: str = Field(min_length=1, max_length=120)
    evidence_start: str = Field(min_length=1, max_length=80)
    evidence_end: str = Field(min_length=1, max_length=80)
    completeness: int = Field(ge=0, le=4)
    relevance: int = Field(ge=0, le=4)
    hook: int = Field(ge=0, le=4)
    flow: int = Field(ge=0, le=4)
    value: int = Field(ge=0, le=4)
    emotion: int = Field(ge=0, le=4)
    shareability: int = Field(ge=0, le=4)
    novelty: int = Field(ge=0, le=4)
    title: str = Field(min_length=1, max_length=70)
    reason: str = Field(min_length=1, max_length=160)


class SemanticRankingResponse(BaseModel):
    rankings: list[SemanticCandidateScore] = Field(max_length=50)


@dataclass
class _SemanticEvaluation:
    accepted: list[ClipSuggestion]
    repairs: list[ClipSuggestion]
    reviews: list[ClipSuggestion]


class QwenClipRanker:
    """Rank a heuristic pre-filter locally and fall back without losing results."""

    def __init__(
        self,
        assets: SemanticAssets,
        timeout_seconds: int = 300,
        context_size: int = 8_192,
    ) -> None:
        self.assets = assets
        self.timeout_seconds = timeout_seconds
        self.context_size = context_size
        self.last_error: str | None = None

    def rank(
        self,
        candidates: list[ClipSuggestion],
        max_results: int,
        *,
        progress: ProgressCallback | None = None,
        cancelled: Event | None = None,
    ) -> list[ClipSuggestion]:
        if not candidates:
            return []
        fallback = HeuristicClipRanker()
        try:
            if progress:
                progress(
                    {
                        "status": "semantic_ranking",
                        "message": (
                            f"IA semântica local avaliando {len(candidates)} trechos "
                            f"em {self.assets.backend}…"
                        ),
                    }
                )
            schema = json.dumps(
                SemanticRankingResponse.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            response, confidences = self._evaluate_twice(
                candidates,
                max_results,
                schema,
                cancelled,
            )
            evaluation = _evaluate_semantic_scores(
                candidates,
                response,
                allow_repairs=True,
                confidences=confidences,
            )
            accepted = list(evaluation.accepted)
            reviews = list(evaluation.reviews)
            repairs = _unique_ranges(evaluation.repairs)
            if repairs:
                if progress:
                    progress(
                        {
                            "status": "semantic_repair",
                            "message": (
                                f"IA semântica reavaliando {len(repairs)} "
                                "limite(s) corrigido(s)…"
                            ),
                        }
                    )
                repaired_response, repaired_confidences = self._evaluate_twice(
                    repairs,
                    len(repairs),
                    schema,
                    cancelled,
                )
                repaired = _evaluate_semantic_scores(
                    repairs,
                    repaired_response,
                    allow_repairs=False,
                    confidences=repaired_confidences,
                )
                accepted.extend(repaired.accepted)
                reviews.extend(repaired.reviews)
            ranked = _finalize_semantic_result(
                candidates,
                accepted,
                reviews,
                max_results,
            )
            self.last_error = None
            if progress:
                progress(
                    {
                        "status": "semantic_complete",
                        "message": f"IA semântica ordenou {len(ranked)} cortes.",
                    }
                )
            return ranked
        except SemanticRankingCancelled:
            raise
        except (OSError, subprocess.SubprocessError, SemanticRankingError, ValidationError) as exc:
            self.last_error = str(exc)
            if progress:
                progress(
                    {
                        "status": "semantic_fallback",
                        "message": "IA semântica indisponível; mantendo ranking heurístico local.",
                        "detail": self.last_error,
                    }
                )
            return _technical_review_fallback(
                candidates,
                max_results,
                fallback,
                progress=progress,
                cancelled=cancelled,
                detail=self.last_error,
            )

    def _evaluate_twice(
        self,
        candidates: list[ClipSuggestion],
        max_results: int,
        schema: str,
        cancelled: Event | None,
    ) -> tuple[SemanticRankingResponse, dict[int, float]]:
        first = _parse_response(
            self._execute(
                _ranking_prompt(candidates, max_results, reverse_order=False),
                schema,
                cancelled,
            )
        )
        second = _parse_response(
            self._execute(
                _ranking_prompt(candidates, max_results, reverse_order=True),
                schema,
                cancelled,
            )
        )
        return _reconcile_semantic_responses(candidates, first, second)

    def _execute(self, prompt: str, schema: str, cancelled: Event | None) -> str:
        command = [
            *self.assets.cli_command,
            "-m",
            str(self.assets.model),
            "-ngl",
            "0" if self.assets.backend == "CPU" else "99",
            "-c",
            str(self.context_size),
            "-n",
            "4096",
            "--temp",
            "0",
            "--seed",
            "42",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--no-show-timings",
            "--color",
            "off",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--json-schema",
            schema,
            "--prompt",
            prompt,
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(  # noqa: S603 - fixed verified executable and argument list
            command,
            cwd=str(self.assets.llama_cli.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            creationflags=flags,
        )
        started = time.monotonic()
        while process.poll() is None:
            if cancelled and cancelled.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise SemanticRankingCancelled("Ranking semântico cancelado.")
            if time.monotonic() - started > self.timeout_seconds:
                process.kill()
                process.wait()
                raise SemanticRankingError("A IA local excedeu o tempo limite.")
            time.sleep(0.1)
        stdout, stderr = process.communicate()
        combined = f"{stdout}\n{stderr}"
        if process.returncode != 0:
            tail = " ".join(combined.strip().splitlines()[-8:])
            raise SemanticRankingError(
                f"llama.cpp terminou com código {process.returncode}: {tail[:600]}"
            )
        return combined


class OllamaClipRanker(QwenClipRanker):
    """Use the already-running local Ollama API with a registered GGUF model."""

    def __init__(
        self,
        assets: OllamaAssets,
        timeout_seconds: int = 300,
        context_size: int = 8_192,
        thinking_verification: bool = False,
    ) -> None:
        self.assets = assets
        self.timeout_seconds = timeout_seconds
        self.context_size = context_size
        self.thinking_verification = thinking_verification
        self.last_error: str | None = None

    def rank(
        self,
        candidates: list[ClipSuggestion],
        max_results: int,
        *,
        progress: ProgressCallback | None = None,
        cancelled: Event | None = None,
    ) -> list[ClipSuggestion]:
        if not candidates:
            return []
        fallback = HeuristicClipRanker()
        try:
            batch_count = max(1, math.ceil(len(candidates) / 6))
            batches = [candidates[offset::batch_count] for offset in range(batch_count)]
            quota = max(1, math.ceil(max_results / batch_count))
            semantic_pool: list[ClipSuggestion] = []
            review_pool: list[ClipSuggestion] = []
            if progress:
                progress(
                    {
                        "status": "semantic_ranking",
                        "message": (
                            f"IA semântica local avaliando {len(candidates)} trechos "
                            f"em {batch_count} lotes no Ollama…"
                        ),
                    }
                )
            schema = json.dumps(
                SemanticRankingResponse.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for index, batch in enumerate(batches, start=1):
                if cancelled and cancelled.is_set():
                    raise SemanticRankingCancelled("Ranking semântico cancelado.")
                if progress:
                    progress(
                        {
                            "status": "semantic_batch",
                            "message": f"IA semântica · lote {index} de {batch_count}…",
                            "batch": index,
                            "batch_count": batch_count,
                        }
                    )
                response, confidences = self._evaluate_twice(
                    batch,
                    quota,
                    schema,
                    cancelled,
                )
                evaluation = _evaluate_semantic_scores(
                    batch,
                    response,
                    allow_repairs=True,
                    repair_candidates=candidates,
                    confidences=confidences,
                )
                review_pool.extend(evaluation.reviews)
                confirmed = self._verify_accepted_with_thinking(
                    evaluation.accepted,
                    candidates,
                    schema,
                    cancelled,
                    progress,
                    allow_repairs=True,
                )
                semantic_pool.extend(confirmed.accepted)
                review_pool.extend(confirmed.reviews)
                repairs = _unique_ranges([*evaluation.repairs, *confirmed.repairs])
                if repairs:
                    repaired_response, repaired_confidences = self._evaluate_twice(
                        repairs,
                        len(repairs),
                        schema,
                        cancelled,
                    )
                    repaired = _evaluate_semantic_scores(
                        repairs,
                        repaired_response,
                        allow_repairs=False,
                        confidences=repaired_confidences,
                    )
                    review_pool.extend(repaired.reviews)
                    confirmed_repairs = self._verify_accepted_with_thinking(
                        repaired.accepted,
                        candidates,
                        schema,
                        cancelled,
                        progress,
                        allow_repairs=False,
                    )
                    semantic_pool.extend(confirmed_repairs.accepted)
                    review_pool.extend(confirmed_repairs.reviews)
            ranked = _finalize_semantic_result(
                candidates,
                semantic_pool,
                review_pool,
                max_results,
            )
            self.last_error = None
            if progress:
                progress(
                    {
                        "status": "semantic_complete",
                        "message": f"IA semântica ordenou {len(ranked)} cortes.",
                    }
                )
            return ranked
        except SemanticRankingCancelled:
            raise
        except (OSError, SemanticRankingError, ValidationError) as exc:
            self.last_error = str(exc)
            if progress:
                progress(
                    {
                        "status": "semantic_fallback",
                        "message": "IA semântica indisponível; mantendo ranking heurístico local.",
                        "detail": self.last_error,
                    }
                )
            return _technical_review_fallback(
                candidates,
                max_results,
                fallback,
                progress=progress,
                cancelled=cancelled,
                detail=self.last_error,
            )

    def _verify_accepted_with_thinking(
        self,
        accepted: list[ClipSuggestion],
        repair_candidates: list[ClipSuggestion],
        schema: str,
        cancelled: Event | None,
        progress: ProgressCallback | None,
        *,
        allow_repairs: bool,
    ) -> _SemanticEvaluation:
        if not self.thinking_verification or not accepted:
            return _SemanticEvaluation(accepted=list(accepted), repairs=[], reviews=[])
        confirmed = _SemanticEvaluation(accepted=[], repairs=[], reviews=[])
        for position, candidate in enumerate(accepted, start=1):
            if cancelled and cancelled.is_set():
                raise SemanticRankingCancelled("Ranking semântico cancelado.")
            if progress:
                progress(
                    {
                        "status": "semantic_thinking",
                        "message": (
                            "Verificação final do raciocínio · "
                            f"{position} de {len(accepted)}…"
                        ),
                    }
                )
            response = _parse_response(
                self._execute_thinking(
                    _ranking_prompt([candidate], 1),
                    schema,
                    cancelled,
                )
            )
            evaluated = _evaluate_semantic_scores(
                [candidate],
                response,
                allow_repairs=allow_repairs,
                repair_candidates=repair_candidates,
                confidences={0: candidate.confidence_score or 0.0},
            )
            confirmed.accepted.extend(evaluated.accepted)
            confirmed.repairs.extend(evaluated.repairs)
            confirmed.reviews.extend(evaluated.reviews)
        return confirmed

    def _execute(self, prompt: str, schema: str, cancelled: Event | None) -> str:
        return self._execute_ollama(prompt, schema, cancelled, thinking=False)

    def _execute_thinking(
        self,
        prompt: str,
        schema: str,
        cancelled: Event | None,
    ) -> str:
        return self._execute_ollama(prompt, schema, cancelled, thinking=True)

    def _execute_ollama(
        self,
        prompt: str,
        schema: str,
        cancelled: Event | None,
        *,
        thinking: bool,
    ) -> str:
        if cancelled and cancelled.is_set():
            raise SemanticRankingCancelled("Ranking semântico cancelado.")
        payload = json.dumps(
            {
                "model": self.assets.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": thinking,
                "format": json.loads(schema),
                "options": {
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": self.context_size,
                    "num_predict": 4_096,
                },
                "keep_alive": "5m",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.assets.host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - localhost only
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise SemanticRankingError(f"Falha na API local do Ollama: {exc}") from exc
        if cancelled and cancelled.is_set():
            raise SemanticRankingCancelled("Ranking semântico cancelado.")
        content = result.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise SemanticRankingError("O Ollama não retornou conteúdo semântico.")
        return content

    def release(self) -> None:
        """Unload only this model from GPU memory; keep Ollama and its files intact."""
        payload = json.dumps(
            {"model": self.assets.model_name, "keep_alive": 0},
        ).encode("utf-8")
        request = Request(
            f"{self.assets.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:  # noqa: S310 - localhost only
                response.read()
        except OSError:
            return


def _ranking_prompt(
    candidates: list[ClipSuggestion],
    max_results: int,
    *,
    reverse_order: bool = False,
) -> str:
    payload = [
        {
            "index": index,
            "start_ms": item.start_ms,
            "end_ms": item.end_ms,
            "signals": item.score_components,
            "selection_goal": item.selection_goal,
            "topic_prompt": item.topic_prompt,
            "audience": item.audience,
            "vocabulary": item.vocabulary,
            "resegmented_from_long_unit": item.resegmented_from_long_unit,
            "existing_editorial_score": item.editorial_score,
            "existing_production_quality": item.production_quality_score,
            "context_before": _semantic_excerpt(item.context_before, 450),
            "transcript": _semantic_excerpt(item.transcript_excerpt, 1_200),
            "context_after": _semantic_excerpt(item.context_after, 450),
        }
        for index, item in enumerate(candidates)
    ]
    if reverse_order:
        payload.reverse()
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "Você audita trechos falados para cortes verticais. Não dê uma decisão final de aprovação: "
        "registre somente observações factuais; o programa aplicará as regras. Compare sempre "
        "context_before, transcript e context_after. topic_stated_in_clip indica se o próprio "
        "transcript apresenta o assunto central. opening_dependency é none quando a abertura se "
        "entende sozinha, repairable quando algumas falas anteriores resolveriam a dependência, "
        "ou strong quando o trecho é apenas reação/resposta sem assunto identificável. Liste em "
        "unresolved_references pronomes ou referências sem antecedente. question_answer_complete "
        "só é true quando pergunta e resposta compreensível estão no trecho. ending_state é "
        "complete quando a última ideia fecha, repairable quando poucas falas seguintes fechariam, "
        "ou ongoing quando a discussão segue sem fechamento próximo. after_continues_same_answer "
        "indica se context_after continua diretamente a resposta ou raciocínio cortado. Para trecho "
        "marcado resegmented_from_long_unit, long_unit_importance é important apenas se for uma "
        "subideia autônoma e uma das partes realmente importantes; nos demais use not_applicable. "
        "central_claim resume a afirmação central. evidence_start e evidence_end devem copiar "
        "fragmentos de no máximo 80 caracteres do começo e do fim que sustentam as observações. "
        "Exemplo: 'Apareceu o Flamengo e ninguém pode dizer não ao Flamengo' já declara o assunto; "
        "marque topic_stated_in_clip=true e opening_dependency=none, mesmo que seja parte de uma "
        "resposta. Já 'Sim, por isso eu aceitei' sem nomear o assunto depende do contexto; marque "
        "topic_stated_in_clip=false e opening_dependency=strong. Uma pergunta seguida de resposta "
        "completa pode ter question_answer_complete=true. Se context_after continuar diretamente "
        "a mesma resposta, use ending_state=repairable e after_continues_same_answer=true, ainda que "
        "o transcript termine com pontuação. Nunca invente conteúdo "
        "ou tendência. Dê notas inteiras de 0 a 4 para completeness, relevance, hook, flow, value, "
        "emotion, shareability e novelty: 4 é excepcional, 3 é bom, 2 é mediano, 1 é fraco e 0 é "
        "ausente. Respeite selection_goal e, em topic, compare com topic_prompt. Avalie todos os "
        f"{len(candidates)} candidatos, uma vez cada, preservando seus índices. O destino comporta "
        f"até {max_results} cortes, mas não omita observações por causa desse limite. O título deve "
        "citar o assunto específico e o motivo deve apontar evidência concreta. Não altere start_ms "
        "ou end_ms: são limites físicos seguros. Responda somente com o JSON exigido pelo schema. "
        f"CANDIDATOS={serialized}"
    )


def _semantic_excerpt(text: str, limit: int = 420) -> str:
    if len(text) <= limit:
        return text
    head = round(limit * 0.62)
    tail = limit - head
    return f"{text[:head]} … [final] {text[-tail:]}"


def _parse_response(output: str) -> SemanticRankingResponse:
    decoder = json.JSONDecoder()
    for position, character in enumerate(output):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(output[position:])
            return SemanticRankingResponse.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            continue
    raise SemanticRankingError("A resposta da IA local não contém JSON válido.")


def _reconcile_semantic_responses(
    candidates: list[ClipSuggestion],
    first: SemanticRankingResponse,
    second: SemanticRankingResponse,
) -> tuple[SemanticRankingResponse, dict[int, float]]:
    first_by_index = {
        item.index: item
        for item in first.rankings
        if item.index < len(candidates)
    }
    second_by_index = {
        item.index: item
        for item in second.rankings
        if item.index < len(candidates)
    }
    reconciled: list[SemanticCandidateScore] = []
    confidences: dict[int, float] = {}
    for index in sorted(set(first_by_index) | set(second_by_index)):
        left = first_by_index.get(index)
        right = second_by_index.get(index)
        if left is None or right is None:
            available = left or right
            if available is None:
                continue
            reconciled.append(
                available.model_copy(
                    update={
                        "topic_stated_in_clip": False,
                        "opening_dependency": "strong",
                        "question_answer_complete": False,
                        "ending_state": "ongoing",
                        "after_continues_same_answer": True,
                        "completeness": min(2, available.completeness),
                        "reason": (
                            "A segunda avaliação não confirmou as observações; "
                            "revisão manual obrigatória."
                        ),
                    }
                )
            )
            confidences[index] = 0.0
            continue
        original = candidates[index]
        reconciled.append(_conservative_consensus(left, right))
        confidences[index] = _consensus_confidence(original, left, right)
    return SemanticRankingResponse(rankings=reconciled), confidences


def _conservative_consensus(
    left: SemanticCandidateScore,
    right: SemanticCandidateScore,
) -> SemanticCandidateScore:
    opening_order = {"none": 0, "repairable": 1, "strong": 2}
    ending_order = {"complete": 0, "repairable": 1, "ongoing": 2}
    opening = max(
        (left.opening_dependency, right.opening_dependency),
        key=opening_order.__getitem__,
    )
    ending = max(
        (left.ending_state, right.ending_state),
        key=ending_order.__getitem__,
    )
    if left.long_unit_importance == right.long_unit_importance:
        long_importance = left.long_unit_importance
    elif "not_important" in (left.long_unit_importance, right.long_unit_importance):
        long_importance = "not_important"
    else:
        long_importance = "not_important"
    unresolved = list(
        dict.fromkeys([*left.unresolved_references, *right.unresolved_references])
    )[:6]
    numeric_fields = (
        "completeness",
        "relevance",
        "hook",
        "flow",
        "value",
        "emotion",
        "shareability",
        "novelty",
    )
    averaged = {
        name: round((getattr(left, name) + getattr(right, name)) / 2)
        for name in numeric_fields
    }
    preferred = max(
        (left, right),
        key=lambda item: (item.relevance, item.completeness, item.flow),
    )
    return preferred.model_copy(
        update={
            "topic_stated_in_clip": (
                left.topic_stated_in_clip and right.topic_stated_in_clip
            ),
            "opening_dependency": opening,
            "unresolved_references": unresolved,
            "question_answer_complete": (
                left.question_answer_complete and right.question_answer_complete
            ),
            "ending_state": ending,
            "after_continues_same_answer": (
                left.after_continues_same_answer or right.after_continues_same_answer
            ),
            "long_unit_importance": long_importance,
            **averaged,
        }
    )


def _consensus_confidence(
    original: ClipSuggestion,
    left: SemanticCandidateScore,
    right: SemanticCandidateScore,
) -> float:
    """Score semantic confidence using the five report-defined dimensions."""
    left_decision = _semantic_decision(original, left)[0]
    right_decision = _semantic_decision(original, right)[0]
    if left_decision == right_decision:
        decision_agreement = 1.0
    elif {left_decision, right_decision} == {"valid", "repair"}:
        decision_agreement = 0.5
    else:
        decision_agreement = 0.0
    evidence_coverage = sum(
        _evidence_occurs(original.transcript_excerpt, evidence)
        for evidence in (
            left.evidence_start,
            left.evidence_end,
            right.evidence_start,
            right.evidence_end,
        )
    ) / 4
    completeness = (left.completeness + right.completeness) / 8
    boundary_stability = (
        1.0
        if (
            left.opening_dependency == right.opening_dependency
            and left.ending_state == right.ending_state
        )
        else 0.5
    )
    context_pairs = (
        (left.topic_stated_in_clip, right.topic_stated_in_clip),
        (left.question_answer_complete, right.question_answer_complete),
        (left.after_continues_same_answer, right.after_continues_same_answer),
        (left.long_unit_importance, right.long_unit_importance),
    )
    context_consistency = sum(a == b for a, b in context_pairs) / len(context_pairs)
    confidence = min(
        1.0,
        .30 * decision_agreement
        + .25 * evidence_coverage
        + .20 * completeness
        + .15 * boundary_stability
        + .10 * context_consistency,
    )
    # A hard disagreement is intentionally capped at the historical 0.55
    # review confidence, avoiding automatic approval by score arithmetic.
    if decision_agreement == 0.0:
        confidence = min(confidence, 0.55)
    return round(confidence, 3)


def _evidence_occurs(transcript: str, evidence: str) -> bool:
    normalized_transcript = _normalize_evidence(transcript)
    normalized_evidence = _normalize_evidence(evidence)
    if len(normalized_evidence) < 8:
        return False
    return normalized_evidence in normalized_transcript


def _normalize_evidence(value: str) -> str:
    ascii_value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value))


def _evaluate_semantic_scores(
    candidates: list[ClipSuggestion],
    response: SemanticRankingResponse,
    *,
    allow_repairs: bool,
    repair_candidates: list[ClipSuggestion] | None = None,
    confidences: dict[int, float] | None = None,
) -> _SemanticEvaluation:
    seen: set[int] = set()
    accepted: list[ClipSuggestion] = []
    repairs: list[ClipSuggestion] = []
    reviews: list[ClipSuggestion] = []
    for semantic in response.rankings:
        if semantic.index >= len(candidates) or semantic.index in seen:
            continue
        seen.add(semantic.index)
        original = candidates[semantic.index]
        decision, explanation = _semantic_decision(original, semantic)
        if decision == "repair" and allow_repairs:
            repaired = _find_repair_candidate(
                original,
                repair_candidates or candidates,
                semantic,
            )
            if repaired is not None:
                repairs.append(
                    repaired.model_copy(
                        update={
                            "editorial_status": "needs_review",
                            "reason": (
                                "Limites ampliados automaticamente para nova validação "
                                f"semântica: {explanation}."
                            )[:300],
                        }
                    )
                )
                continue
        if decision != "valid":
            reviews.append(
                _review_candidate(
                    original,
                    semantic,
                    explanation,
                    confidence=(confidences or {}).get(semantic.index),
                )
            )
            continue
        accepted.append(
            _validated_candidate(
                original,
                semantic,
                confidence=(confidences or {}).get(semantic.index),
            )
        )
    return _SemanticEvaluation(accepted=accepted, repairs=repairs, reviews=reviews)


def _semantic_decision(
    original: ClipSuggestion,
    semantic: SemanticCandidateScore,
) -> tuple[Literal["valid", "repair", "reject"], str]:
    hard_failures: list[str] = []
    topic_supported = semantic.topic_stated_in_clip or _topic_supported_by_text(
        original,
        semantic,
    )
    transcript_has_question_answer = _question_and_answer_present(
        original.transcript_excerpt
    )
    opening_names_subject = (
        topic_supported
        and not semantic.unresolved_references
        and _opening_supports_claim(original.transcript_excerpt, semantic.central_claim)
    )
    if not topic_supported:
        hard_failures.append("o assunto central não aparece dentro do trecho")
    if semantic.completeness < 3:
        hard_failures.append("completude abaixo de 3/4")
    if semantic.relevance < 2:
        hard_failures.append("relevância abaixo de 2/4")
    if (
        original.resegmented_from_long_unit
        and semantic.long_unit_importance != "important"
    ):
        hard_failures.append("não é uma parte importante da discussão longa")
    opening_passes = (
        semantic.opening_dependency == "none"
        or semantic.question_answer_complete
        or transcript_has_question_answer
        or opening_names_subject
    )
    if semantic.opening_dependency == "strong" and not opening_passes:
        hard_failures.append("a abertura depende fortemente do contexto anterior")
    if hard_failures:
        return "reject", "; ".join(hard_failures)

    repair_reasons: list[str] = []
    if not opening_passes:
        repair_reasons.append("a abertura precisa de falas anteriores")
    if semantic.ending_state != "complete" or semantic.after_continues_same_answer:
        repair_reasons.append("a resposta ou ideia continua depois do fim")
    if repair_reasons:
        return "repair", "; ".join(repair_reasons)
    if transcript_has_question_answer and semantic.opening_dependency != "none":
        return "valid", "a transcrição contém pergunta e resposta completas"
    if opening_names_subject and semantic.opening_dependency != "none":
        return "valid", "a abertura nomeia o assunto e a ideia termina completa"
    return "valid", "assunto, começo e conclusão estão completos no próprio trecho"


def _topic_supported_by_text(
    original: ClipSuggestion,
    semantic: SemanticCandidateScore,
) -> bool:
    claim_tokens = _content_tokens(semantic.central_claim)
    transcript_tokens = _content_tokens(original.transcript_excerpt)
    common = claim_tokens & transcript_tokens
    evidence_supported = (
        _evidence_occurs(original.transcript_excerpt, semantic.evidence_start)
        or _evidence_occurs(original.transcript_excerpt, semantic.evidence_end)
    )
    return bool(claim_tokens) and evidence_supported and len(common) >= min(2, len(claim_tokens))


def _opening_supports_claim(transcript: str, central_claim: str) -> bool:
    opening = " ".join(transcript.split()[:28])
    common = _content_tokens(opening) & _content_tokens(central_claim)
    return any(len(token) >= 5 for token in common)


def _question_and_answer_present(transcript: str) -> bool:
    question_end = transcript.find("?")
    if question_end < 0:
        return False
    question_words = transcript[: question_end + 1].split()
    answer_words = transcript[question_end + 1 :].split()
    return len(question_words) >= 5 and len(answer_words) >= 12


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in _normalize_evidence(value).split()
        if len(token) >= 3 and token not in _SEMANTIC_STOP_WORDS
    }


def _find_repair_candidate(
    original: ClipSuggestion,
    candidates: list[ClipSuggestion],
    semantic: SemanticCandidateScore,
) -> ClipSuggestion | None:
    needs_opening = (
        semantic.opening_dependency == "repairable"
        and not semantic.question_answer_complete
    )
    needs_ending = (
        semantic.ending_state != "complete"
        or semantic.after_continues_same_answer
    )
    compatible: list[ClipSuggestion] = []
    for candidate in candidates:
        if candidate.start_ms > original.start_ms or candidate.end_ms < original.end_ms:
            continue
        if needs_opening and candidate.start_ms >= original.start_ms:
            continue
        if needs_ending and candidate.end_ms <= original.end_ms:
            continue
        if candidate.duration_ms > 179_000:
            continue
        compatible.append(candidate)
    if not compatible:
        return None
    return min(
        compatible,
        key=lambda item: (
            item.duration_ms - original.duration_ms,
            -item.quality_score,
            item.start_ms,
        ),
    )


def _validated_candidate(
    original: ClipSuggestion,
    semantic: SemanticCandidateScore,
    confidence: float | None = None,
) -> ClipSuggestion:
    components = dict(original.score_components)
    factors = {
        "completude_semântica": semantic.completeness / 4,
        "relevância": semantic.relevance / 4,
        "hook_semântico": semantic.hook / 4,
        "fluxo_semântico": semantic.flow / 4,
        "valor_semântico": semantic.value / 4,
        "emoção_semântica": semantic.emotion / 4,
        "compartilhável": semantic.shareability / 4,
        "novidade": semantic.novelty / 4,
    }
    potential = _potential_score(original, semantic)
    components.update({name: round(value, 3) for name, value in factors.items()})
    components["potencial"] = potential
    components["semântica"] = potential
    title = " ".join(semantic.title.split())[:70] or original.title
    reason = " ".join(semantic.reason.split())[:240]
    return original.model_copy(
        update={
            "title": title,
            "reason": f"IA local: {reason}",
            "quality_score": potential,
            "potential_score": potential,
            "score_components": components,
            "editorial_status": "validated",
            "editorial_score": round(semantic.completeness / 4, 3),
            "relevance_score": round(semantic.relevance / 4, 3),
            "central_claim": semantic.central_claim,
            "payoff": semantic.reason,
            "evidence_start": semantic.evidence_start,
            "evidence_end": semantic.evidence_end,
            "opening_dependency": semantic.opening_dependency,
            "ending_state": semantic.ending_state,
            "after_continues_same_answer": semantic.after_continues_same_answer,
            "confidence_score": (
                confidence
                if confidence is not None
                else _phase_one_confidence(original, semantic)
            ),
        }
    )


def _phase_one_confidence(
    original: ClipSuggestion,
    semantic: SemanticCandidateScore,
) -> float:
    evidence = sum(
        bool(value.strip())
        for value in (
            semantic.central_claim,
            semantic.evidence_start,
            semantic.evidence_end,
        )
    ) / 3
    boundary = original.editorial_score if original.editorial_score is not None else 1.0
    unresolved = min(1.0, len(semantic.unresolved_references) / 3)
    return round(max(0.0, min(1.0, .55 * evidence + .4 * boundary - .15 * unresolved)), 3)


def _review_candidate(
    original: ClipSuggestion,
    semantic: SemanticCandidateScore,
    explanation: str,
    confidence: float | None = None,
) -> ClipSuggestion:
    components = dict(original.score_components)
    components["completude_semântica"] = round(semantic.completeness / 4, 3)
    components["relevância"] = round(semantic.relevance / 4, 3)
    return original.model_copy(
        update={
            "editorial_status": "needs_review",
            "status": "pending",
            "reason": f"Revisão necessária pela IA local: {explanation}."[:300],
            "score_components": components,
            "editorial_score": round(semantic.completeness / 4, 3),
            "relevance_score": round(semantic.relevance / 4, 3),
            "central_claim": semantic.central_claim,
            "payoff": semantic.reason,
            "evidence_start": semantic.evidence_start,
            "evidence_end": semantic.evidence_end,
            "opening_dependency": semantic.opening_dependency,
            "ending_state": semantic.ending_state,
            "after_continues_same_answer": semantic.after_continues_same_answer,
            "confidence_score": (
                confidence
                if confidence is not None
                else _phase_one_confidence(original, semantic)
            ),
        }
    )


def _finalize_semantic_result(
    candidates: list[ClipSuggestion],
    accepted: list[ClipSuggestion],
    reviews: list[ClipSuggestion],
    max_results: int,
) -> list[ClipSuggestion]:
    accepted = _unique_ranges(accepted)
    if accepted:
        return HeuristicClipRanker().rank(accepted, max_results)
    review_by_range = {
        (item.start_ms, item.end_ms): item
        for item in reviews
    }
    review_candidates = [
        review_by_range.get(
            (item.start_ms, item.end_ms),
            item.model_copy(
                update={
                    "editorial_status": "needs_review",
                    "status": "pending",
                    "reason": (
                        "Revisão necessária: a IA local não devolveu observações "
                        "suficientes para aprovação automática."
                    ),
                }
            ),
        )
        for item in candidates
    ]
    return HeuristicClipRanker().rank(review_candidates, min(3, max_results))


def _technical_review_fallback(
    candidates: list[ClipSuggestion],
    max_results: int,
    fallback: HeuristicClipRanker,
    *,
    progress: ProgressCallback | None,
    cancelled: Event | None,
    detail: str | None,
) -> list[ClipSuggestion]:
    selected = fallback.rank(
        candidates,
        min(3, max_results),
        progress=progress,
        cancelled=cancelled,
    )
    reason = "A IA semântica local falhou; confira este corte manualmente"
    if detail:
        reason = f"{reason}: {detail}"
    return [
        item.model_copy(
            update={
                "editorial_status": "needs_review",
                "status": "pending",
                "reason": reason[:300],
                "confidence_score": 0.0,
            }
        )
        for item in selected
    ]


def _unique_ranges(candidates: list[ClipSuggestion]) -> list[ClipSuggestion]:
    unique: dict[tuple[int, int], ClipSuggestion] = {}
    for item in candidates:
        key = (item.start_ms, item.end_ms)
        previous = unique.get(key)
        if previous is None or item.quality_score > previous.quality_score:
            unique[key] = item
    return list(unique.values())


def _merge_semantic_scores(
    candidates: list[ClipSuggestion],
    response: SemanticRankingResponse,
    max_results: int,
) -> list[ClipSuggestion]:
    """Compatibility helper used by focused tests and older callers."""
    evaluated = _evaluate_semantic_scores(candidates, response, allow_repairs=False)
    return _finalize_semantic_result(
        candidates,
        evaluated.accepted,
        evaluated.reviews,
        max_results,
    )


def _potential_score(
    original: ClipSuggestion,
    semantic: SemanticCandidateScore,
) -> float:
    """Calculate potential only after editorial validity has passed.

    The balanced baseline follows the report: hook 25%, flow/conclusion 25%,
    value 20%, relevance 15%, emotion/shareability 10% and production quality
    5%. Other modes are deliberate editorial variations, not hidden heuristics.
    """
    production = original.production_quality_score
    if production is None:
        production = original.score_components.get("qualidade_audiovisual", original.quality_score)
    factors = {
        "relevance": semantic.relevance / 4,
        "hook": semantic.hook / 4,
        "flow": semantic.flow / 4,
        "value": semantic.value / 4,
        "emotion": semantic.emotion / 4,
        "shareability": semantic.shareability / 4,
        "production": production,
        "novelty": semantic.novelty / 4,
    }
    weights = {
        "balanced": {
            "relevance": .15, "hook": .25, "flow": .25, "value": .20,
            "emotion": .05, "shareability": .05, "production": .05, "novelty": 0,
        },
        "faithful": {
            "relevance": .20, "hook": .15, "flow": .28, "value": .22,
            "emotion": .03, "shareability": .02, "production": .05, "novelty": .05,
        },
        "viral": {
            "relevance": .08, "hook": .28, "flow": .18, "value": .10,
            "emotion": .12, "shareability": .16, "production": .05, "novelty": .03,
        },
        "topic": {
            "relevance": .30, "hook": .16, "flow": .22, "value": .20,
            "emotion": .02, "shareability": .02, "production": .05, "novelty": .03,
        },
    }[original.selection_goal]
    return round(min(1.0, sum(factors[name] * weight for name, weight in weights.items())), 3)
