"""Temporary face IDs based only on geometry and time."""

import math

from cortaflow.domain.tracking import FaceBox, FaceObservation, FaceTrackPoint


def intersection_over_union(a: FaceBox, b: FaceBox) -> float:
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union else 0


def association_score(a: FaceBox, b: FaceBox) -> float:
    """Combine overlap, center proximity and size similarity without identity data."""
    iou = intersection_over_union(a, b)
    center_a = (a.x + a.width / 2, a.y + a.height / 2)
    center_b = (b.x + b.width / 2, b.y + b.height / 2)
    distance = math.dist(center_a, center_b)
    proximity = max(0.0, 1 - distance / 0.35)
    area_a, area_b = a.width * a.height, b.width * b.height
    size_similarity = min(area_a, area_b) / max(area_a, area_b)
    return 0.6 * iou + 0.25 * proximity + 0.15 * size_similarity


class FaceTracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        smoothing: float = 0.35,
        max_gap_ms: int = 1200,
        association_threshold: float = 0.35,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.smoothing = smoothing
        self.max_gap_ms = max_gap_ms
        self.association_threshold = association_threshold
        self._tracks: dict[int, FaceTrackPoint] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def update(
        self,
        timestamp_ms: int,
        detections: list[FaceBox | FaceObservation],
    ) -> list[FaceTrackPoint]:
        self._tracks = {
            key: value
            for key, value in self._tracks.items()
            if timestamp_ms - value.timestamp_ms <= self.max_gap_ms
        }
        available = set(self._tracks)
        results: list[FaceTrackPoint] = []
        for detection in detections:
            observation = detection if isinstance(detection, FaceObservation) else FaceObservation(box=detection)
            matches = [
                (association_score(observation.box, self._tracks[key].box), key)
                for key in available
            ]
            best_score, track_id = max(matches, default=(0.0, -1))
            best_iou = (
                intersection_over_union(observation.box, self._tracks[track_id].box)
                if track_id >= 0
                else 0.0
            )
            if best_iou < self.iou_threshold and best_score < self.association_threshold:
                track_id = self._next_id
                self._next_id += 1
            else:
                available.remove(track_id)
            previous = self._tracks.get(track_id)
            box = _smooth(previous.box, observation.box, self.smoothing) if previous else observation.box
            mouth = _smooth_value(
                previous.mouth_openness if previous else None,
                observation.mouth_openness,
                self.smoothing,
            )
            point = FaceTrackPoint(
                track_id=track_id,
                timestamp_ms=timestamp_ms,
                box=box,
                mouth_openness=mouth,
            )
            self._tracks[track_id] = point
            results.append(point)
        return results


def _smooth(old: FaceBox, new: FaceBox, alpha: float) -> FaceBox:
    blend = lambda a, b: a + alpha * (b - a)
    x = blend(old.x, new.x)
    y = blend(old.y, new.y)
    width = min(1 - x, blend(old.width, new.width))
    height = min(1 - y, blend(old.height, new.height))
    return FaceBox(
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=new.confidence,
    )


def _smooth_value(old: float | None, new: float | None, alpha: float) -> float | None:
    if new is None:
        return old
    if old is None:
        return new
    return old + alpha * (new - old)
