from cortaflow.domain.tracking import FaceBox, FaceObservation
from cortaflow.services.face_tracking import FaceTracker, association_score, intersection_over_union


def test_iou_and_stable_temporary_id() -> None:
    first = FaceBox(x=.1, y=.1, width=.2, height=.2)
    second = FaceBox(x=.11, y=.1, width=.2, height=.2)
    assert intersection_over_union(first, second) > .8
    tracker = FaceTracker(smoothing=.5)
    one = tracker.update(0, [first])[0]
    two = tracker.update(100, [second])[0]
    assert one.track_id == two.track_id
    assert one.box.x < two.box.x < second.x


def test_two_faces_receive_distinct_ids_and_scene_reset() -> None:
    tracker = FaceTracker()
    points = tracker.update(0, [FaceBox(x=.1, y=.1, width=.2, height=.2), FaceBox(x=.7, y=.1, width=.2, height=.2)])
    assert len({point.track_id for point in points}) == 2
    tracker.reset()
    assert tracker.update(100, [FaceBox(x=.1, y=.1, width=.2, height=.2)])[0].track_id not in {point.track_id for point in points}


def test_association_uses_proximity_and_preserves_mouth_motion() -> None:
    first = FaceObservation(box=FaceBox(x=.1, y=.1, width=.2, height=.2), mouth_openness=.2)
    moved = FaceObservation(box=FaceBox(x=.18, y=.1, width=.2, height=.2), mouth_openness=.8)
    assert association_score(first.box, moved.box) > 0
    tracker = FaceTracker(iou_threshold=.1, smoothing=.5)
    one = tracker.update(0, [first])[0]
    two = tracker.update(200, [moved])[0]
    assert one.track_id == two.track_id
    assert two.mouth_openness == .5
