from cortaflow.domain.editing import ReframeSettings
from cortaflow.domain.project import ReframeKeyframe
from cortaflow.domain.tracking import CropFrame, FaceBox, FaceTrackPoint
from cortaflow.services.auto_reframe import apply_reframe_settings, calculate_crop, generate_reframe_keyframes, interpolate_crop, keep_face_in_safe_area, smooth_crop


def test_horizontal_to_vertical_crop_is_bounded() -> None:
    crop = calculate_crop(1920, 1080, focus=FaceBox(x=.75, y=.2, width=.1, height=.2))
    assert crop.width == 608
    assert crop.height == 1080
    assert 0 <= crop.x <= 1920 - crop.width


def test_detected_face_stays_inside_horizontal_safe_area_of_vertical_crop() -> None:
    face = FaceBox(x=.641, y=.244, width=.149, height=.293)

    crop = calculate_crop(1920, 1080, focus=face)
    face_left = face.x * 1920
    face_right = (face.x + face.width) * 1920
    relative_center = ((face_left + face_right) / 2 - crop.x) / crop.width

    assert crop.x <= face_left < face_right <= crop.x + crop.width
    assert .25 <= relative_center <= .75


def test_safe_area_recovers_a_face_after_tracking_lag() -> None:
    face = FaceBox(x=.68, y=.2, width=.14, height=.28)
    lagging = CropFrame(x=700, y=0, width=608, height=1080)

    corrected = keep_face_in_safe_area(lagging, face, 1920, 1080)
    face_left = face.x * 1920
    face_right = (face.x + face.width) * 1920

    assert corrected.x + corrected.width * .16 <= face_left
    assert face_right <= corrected.x + corrected.width * .84


def test_small_motion_stays_in_dead_zone_and_large_motion_is_limited() -> None:
    previous = CropFrame(x=100, y=20, width=600, height=1000)
    small = smooth_crop(previous, CropFrame(x=105, y=24, width=600, height=1000))
    assert (small.x, small.y) == (100, 20)
    large = smooth_crop(previous, CropFrame(x=900, y=500, width=600, height=1000), alpha=1, max_step_px=50)
    assert (large.x, large.y) == (150, 70)


def test_interpolation_and_manual_track_selection() -> None:
    start = CropFrame(x=0, y=0, width=600, height=1000)
    end = CropFrame(x=100, y=40, width=600, height=1000)
    assert interpolate_crop(start, end, .5).x == 50
    tracks = [
        FaceTrackPoint(track_id=1, timestamp_ms=0, box=FaceBox(x=.05, y=.2, width=.15, height=.2)),
        FaceTrackPoint(track_id=2, timestamp_ms=0, box=FaceBox(x=.7, y=.2, width=.15, height=.2)),
    ]
    keyframes = generate_reframe_keyframes(tracks, 1920, 1080, selected_track_id=2)
    assert keyframes[0].crop.x > 800


def test_scene_boundary_resets_crop_smoothing() -> None:
    tracks = [
        FaceTrackPoint(track_id=1, timestamp_ms=0, box=FaceBox(x=.05, y=.2, width=.15, height=.2)),
        FaceTrackPoint(track_id=2, timestamp_ms=1000, box=FaceBox(x=.75, y=.2, width=.15, height=.2)),
    ]
    smoothed = generate_reframe_keyframes(tracks, 1920, 1080)
    reset = generate_reframe_keyframes(tracks, 1920, 1080, scene_boundaries_ms=[1000])
    assert reset[-1].crop.x > smoothed[-1].crop.x


def test_applies_aspect_zoom_and_speed_limit_to_keyframes() -> None:
    keyframes = [
        ReframeKeyframe(timestamp_ms=0, crop=CropFrame(x=0, y=0, width=608, height=1080), face_safe=False),
        ReframeKeyframe(timestamp_ms=1000, crop=CropFrame(x=1000, y=0, width=608, height=1080), face_safe=False),
    ]
    result = apply_reframe_settings(
        keyframes,
        ReframeSettings(aspect_ratio="1:1", zoom=2, smoothing=1, max_speed_px=50),
        1920,
        1080,
    )
    assert (result[0].crop.width, result[0].crop.height) == (540, 540)
    assert result[1].crop.x - result[0].crop.x <= 50


def test_render_settings_do_not_smooth_across_a_scene_cut() -> None:
    keyframes = [
        ReframeKeyframe(
            timestamp_ms=0,
            crop=CropFrame(x=0, y=0, width=608, height=1080),
        ),
        ReframeKeyframe(
            timestamp_ms=1000,
            crop=CropFrame(x=1000, y=0, width=608, height=1080),
            scene_reset=True,
        ),
    ]

    result = apply_reframe_settings(
        keyframes,
        ReframeSettings(smoothing=1, max_speed_px=50),
        1920,
        1080,
    )

    assert result[1].scene_reset
    assert result[1].crop.x == 1000


def test_manual_reframe_uses_saved_position() -> None:
    result = apply_reframe_settings(
        [],
        ReframeSettings(automatic=False, aspect_ratio="4:5", x=123, y=45),
        1920,
        1080,
    )
    assert (result[0].crop.x, result[0].crop.y) == (123, 0)
