"""Scene-level safeguards for drowning-model predictions.

The bundled model can mistake an ordinary webcam image for swimming.  A
prediction of swimming or drowning only makes sense when the frame contains a
substantial visible pool-water region, so this module corrects those labels in
dry, non-pool scenes.
"""

import cv2
import numpy as np


def _class_id(result, expected_name):
    """Return a model class id while tolerating hyphen/space differences."""
    expected = expected_name.replace(" ", "").replace("-", "").lower()
    for class_id, name in result.names.items():
        normalized = str(name).replace(" ", "").replace("-", "").lower()
        if normalized == expected:
            return int(class_id)
    return None


def has_visible_pool_water(frame_bgr, min_fraction=0.08):
    """Estimate whether a frame contains a substantial bright blue water area.

    This is intentionally conservative: small blue objects such as shirts or
    walls do not count as a pool.  It is a safety guard, not a water detector.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    water_mask = cv2.inRange(
        hsv,
        np.array((85, 80, 70), dtype=np.uint8),
        np.array((135, 255, 255), dtype=np.uint8),
    )
    water_fraction = float(np.count_nonzero(water_mask)) / water_mask.size
    return water_fraction >= min_fraction


def correct_dry_scene_predictions(result, frame_bgr, enabled=True, min_fraction=0.08):
    """Relabel detections as out-of-water when no pool water is visible.

    The result is updated before plotting and alert checks, so an indoor webcam
    frame cannot appear as a swimming/drowning event simply because it contains
    a person.
    """
    if not enabled or has_visible_pool_water(frame_bgr, min_fraction):
        return result

    out_of_water_id = _class_id(result, "Person out of water")
    if out_of_water_id is None or result.boxes is None or len(result.boxes) == 0:
        return result

    # Ultralytics inference tensors are read-only.  Replace the backing tensor
    # with a normal copy before changing its final (class-id) column.
    result.boxes.data = result.boxes.data.clone()
    result.boxes.data[:, -1] = out_of_water_id
    return result


def remove_conflicting_overlaps(result, iou_threshold=0.45):
    """Keep one label when several class boxes describe the same person.

    Ultralytics performs non-maximum suppression per class.  A model can
    therefore return overlapping Swimming, Drowning, and out-of-water boxes for
    one person.  This second, class-agnostic pass retains the most confident
    box while preserving boxes for separate people.
    """
    if result.boxes is None or len(result.boxes) < 2:
        return result

    boxes = result.boxes
    coordinates = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    order = scores.argsort()[::-1]
    keep = []

    for candidate in order:
        left, top, right, bottom = coordinates[candidate]
        candidate_area = max(0.0, right - left) * max(0.0, bottom - top)
        overlaps = False

        for accepted in keep:
            a_left, a_top, a_right, a_bottom = coordinates[accepted]
            intersection = max(0.0, min(right, a_right) - max(left, a_left)) * max(
                0.0, min(bottom, a_bottom) - max(top, a_top)
            )
            accepted_area = max(0.0, a_right - a_left) * max(0.0, a_bottom - a_top)
            union = candidate_area + accepted_area - intersection
            if union and intersection / union >= iou_threshold:
                overlaps = True
                break

        if not overlaps:
            keep.append(candidate)

    if len(keep) != len(boxes):
        boxes.data = boxes.data[keep].clone()

    return result


def suppress_weak_scene_conflicts(result, swimming_min_confidence=0.30):
    """Drop implausibly large, weaker labels when a swimmer is detected.

    Some Video 2 frames produce a tight Swimming box and a low-confidence
    Drowning/out-of-water box spanning most of the frame.  The large box is a
    scene artifact rather than another person.  Keep it only when it is at
    least as confident as the swimming detection.
    """
    if result.boxes is None or len(result.boxes) < 2:
        return result

    swimming_id = _class_id(result, "Swimming")
    if swimming_id is None:
        return result

    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    swimming_scores = scores[classes == swimming_id]
    if not len(swimming_scores):
        return result

    strongest_swimming = float(swimming_scores.max())
    if strongest_swimming < swimming_min_confidence:
        return result

    height, width = result.orig_shape
    frame_area = height * width
    coordinates = result.boxes.xyxy.cpu().numpy()
    keep = []

    for index, (class_id, score, (left, top, right, bottom)) in enumerate(
        zip(classes, scores, coordinates)
    ):
        area = max(0.0, right - left) * max(0.0, bottom - top)
        is_large_weaker_competitor = (
            class_id != swimming_id
            and area / frame_area >= 0.40
            and score < strongest_swimming
        )
        if not is_large_weaker_competitor:
            keep.append(index)

    if len(keep) != len(result.boxes):
        result.boxes.data = result.boxes.data[keep].clone()

    return result
