"""MediaPipe Tasks face adapters; no identity recognition."""

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from cortaflow.config import AppConfig
from cortaflow.domain.tracking import FaceBox, FaceObservation


def find_local_face_landmarker(preferred: Path | None = None) -> Path | None:
    """Locate an already-installed Face Landmarker model without downloading anything."""
    config = AppConfig()
    candidates = [
        preferred,
        config.cache_dir / "models" / "mediapipe" / "face_landmarker.task",
        config.data_dir / "Audit" / "face_landmarker.task",
        # Compatibility with the location used by early development versions.
        config.data_dir.parent / "Audit" / "face_landmarker.task",
    ]
    return next(
        (candidate.resolve() for candidate in candidates if candidate and candidate.is_file()),
        None,
    )


class MediaPipeFaceDetector:
    """Compatibility adapter for an official MediaPipe face detector task."""

    def __init__(self, model_path: Path, minimum_confidence: float = 0.5) -> None:
        if not model_path.is_file():
            raise FileNotFoundError("Selecione um modelo oficial do MediaPipe Face Detector.")
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=minimum_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list[FaceBox]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._detector.detect_for_video(image, timestamp_ms)
        width, height = rgb_frame.shape[1], rgb_frame.shape[0]
        boxes: list[FaceBox] = []
        for item in result.detections:
            x = max(0.0, item.bounding_box.origin_x / width)
            y = max(0.0, item.bounding_box.origin_y / height)
            box_width = min(1.0 - x, item.bounding_box.width / width)
            box_height = min(1.0 - y, item.bounding_box.height / height)
            if box_width <= 0 or box_height <= 0:
                continue
            boxes.append(
                FaceBox(
                    x=x,
                    y=y,
                    width=box_width,
                    height=box_height,
                    confidence=item.categories[0].score if item.categories else 0,
                )
            )
        return boxes

    def close(self) -> None:
        self._detector.close()


class MediaPipeFaceLandmarker:
    """Extract anonymous boxes and lip motion from face landmarks in video mode."""

    def __init__(self, model_path: Path, minimum_confidence: float = 0.5, max_faces: int = 8) -> None:
        if not model_path.is_file():
            raise FileNotFoundError("Selecione um modelo oficial do MediaPipe Face Landmarker.")
        video_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=minimum_confidence,
            min_face_presence_confidence=minimum_confidence,
            min_tracking_confidence=minimum_confidence,
        )
        tile_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=minimum_confidence,
            min_face_presence_confidence=minimum_confidence,
            min_tracking_confidence=minimum_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(video_options)
        self._tile_landmarker = vision.FaceLandmarker.create_from_options(tile_options)

    def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> list[FaceObservation]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        height, width = rgb_frame.shape[:2]
        observations = _landmark_observations(result, width, height, width, height)
        small_or_missing = not observations or max(item.box.width for item in observations) < 0.12
        if width / max(1, height) >= 1.4 and small_or_missing:
            for left, top, tile_width, tile_height in _wide_shot_tiles(width, height):
                tile = np.ascontiguousarray(
                    rgb_frame[top : top + tile_height, left : left + tile_width]
                )
                tile_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=tile)
                tile_result = self._tile_landmarker.detect(tile_image)
                tile_observations = _landmark_observations(
                    tile_result,
                    tile_width,
                    tile_height,
                    width,
                    height,
                    left,
                    top,
                )
                observations = _merge_observations(observations, tile_observations)
        return observations

    def close(self) -> None:
        self._landmarker.close()
        self._tile_landmarker.close()


def _wide_shot_tiles(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Cover a landscape conversation shot with overlapping, near-square regions."""
    tile_size = min(width, height, max(256, round(height * 5 / 9)))
    top = min(max(0, round(height * 5 / 36)), max(0, height - tile_size))
    available = max(0, width - tile_size)
    positions = [round(available * index / 5) for index in range(6)]
    return [(left, top, tile_size, tile_size) for left in dict.fromkeys(positions)]


def _landmark_observations(
    result: object,
    frame_width: int,
    frame_height: int,
    canvas_width: int,
    canvas_height: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[FaceObservation]:
    observations: list[FaceObservation] = []
    for landmarks in result.face_landmarks:
        xs = [max(0.0, min(1.0, float(point.x))) for point in landmarks]
        ys = [max(0.0, min(1.0, float(point.y))) for point in landmarks]
        if not xs or not ys:
            continue
        left = (offset_x + min(xs) * frame_width) / canvas_width
        right = (offset_x + max(xs) * frame_width) / canvas_width
        top = (offset_y + min(ys) * frame_height) / canvas_height
        bottom = (offset_y + max(ys) * frame_height) / canvas_height
        left, right = max(0.0, left), min(1.0, right)
        top, bottom = max(0.0, top), min(1.0, bottom)
        if right <= left or bottom <= top:
            continue
        observations.append(
            FaceObservation(
                box=FaceBox(x=left, y=top, width=right - left, height=bottom - top),
                mouth_openness=_mouth_openness(landmarks),
            )
        )
    return observations


def _merge_observations(
    existing: list[FaceObservation],
    candidates: list[FaceObservation],
) -> list[FaceObservation]:
    merged = list(existing)
    for candidate in candidates:
        if all(_box_iou(candidate.box, current.box) < 0.35 for current in merged):
            merged.append(candidate)
    return merged


def _box_iou(left: FaceBox, right: FaceBox) -> float:
    intersection_width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    intersection_height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    intersection = intersection_width * intersection_height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0 else 0.0


def _mouth_openness(landmarks: list[object]) -> float | None:
    """Normalize the lip gap by mouth width using standard mesh indices."""
    if len(landmarks) <= 308:
        return None
    upper, lower = landmarks[13], landmarks[14]
    left, right = landmarks[78], landmarks[308]
    vertical = abs(float(lower.y) - float(upper.y))
    horizontal = ((float(right.x) - float(left.x)) ** 2 + (float(right.y) - float(left.y)) ** 2) ** 0.5
    if horizontal <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, vertical / horizontal * 4.0))
