"""Local audio energy and voice-activity evidence extraction with PyAV."""

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import av
import numpy as np

from cortaflow.domain.analysis import TimeRange
from cortaflow.domain.subtitle import Transcript
from cortaflow.domain.tracking import AudioEvidence


class AudioAnalysisCancelled(RuntimeError):
    """Raised when audio analysis is cancelled."""


def extract_audio_evidence(
    media_path: Path,
    transcript: Transcript | None = None,
    silences: list[TimeRange] | None = None,
    window_ms: int = 200,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Event | None = None,
) -> list[AudioEvidence]:
    """Decode audio locally, calculate normalized RMS and combine it with VAD intervals."""
    media_path = media_path.resolve()
    if not media_path.is_file():
        raise FileNotFoundError(media_path)
    if window_ms <= 0:
        raise ValueError("A janela de áudio deve ser positiva.")
    silences = silences or []
    buckets: dict[int, list[float]] = {}
    with av.open(str(media_path)) as container:
        if not container.streams.audio:
            return []
        audio_stream = container.streams.audio[0]
        sample_clock_ms = 0.0
        for frame in container.decode(audio_stream):
            if cancelled and cancelled.is_set():
                raise AudioAnalysisCancelled("Análise de áudio cancelada.")
            samples = frame.to_ndarray()
            normalized = _normalize_audio(samples)
            energy = float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0
            start_ms = round(frame.time * 1000) if frame.time is not None else round(sample_clock_ms)
            duration_ms = frame.samples / max(1, frame.sample_rate) * 1000
            midpoint = start_ms + duration_ms / 2
            bucket = int(midpoint // window_ms)
            buckets.setdefault(bucket, []).append(energy)
            sample_clock_ms = start_ms + duration_ms
            if progress:
                progress({"status": "audio", "position_ms": round(sample_clock_ms)})
    if not buckets:
        return []
    raw = [sum(values) / len(values) for _, values in sorted(buckets.items())]
    reference = float(np.percentile(raw, 95)) if raw else 0.0
    reference = max(reference, 1e-9)
    evidence: list[AudioEvidence] = []
    words = transcript.words if transcript else []
    for bucket, values in sorted(buckets.items()):
        timestamp_ms = bucket * window_ms + window_ms // 2
        energy = min(1.0, (sum(values) / len(values)) / reference)
        in_silence = any(item.start_ms <= timestamp_ms < item.end_ms for item in silences)
        if words:
            voice_active = any(
                word.start_ms - 100 <= timestamp_ms <= word.end_ms + 100 for word in words
            ) and not in_silence
        else:
            voice_active = energy >= 0.08 and not in_silence
        evidence.append(
            AudioEvidence(
                timestamp_ms=timestamp_ms,
                energy=round(energy, 4),
                voice_active=voice_active,
            )
        )
    return evidence


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.integer):
        limit = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        return samples.astype(np.float32) / float(limit)
    return samples.astype(np.float32, copy=False)
