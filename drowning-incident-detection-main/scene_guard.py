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


_PERSON_MODEL = None


def get_person_model():
    """Lazily load the COCO person detector model for reliable multi-person detection."""
    global _PERSON_MODEL
    if _PERSON_MODEL is None:
        from pathlib import Path
        from ultralytics import YOLO
        import settings

        model_candidates = [
            getattr(settings, "PERSON_MODEL", None),
            settings.ROOT / "weights" / "yolo11n.pt",
            settings.ROOT / "public-drowning-dataset" / "Module" / "yolo11n.pt",
        ]

        for cand in model_candidates:
            if cand and Path(cand).exists():
                try:
                    _PERSON_MODEL = YOLO(str(cand))
                    break
                except Exception:
                    continue

    return _PERSON_MODEL


def correct_dry_scene_predictions(result, frame_bgr, enabled=True, min_fraction=0.08):
    """Relabel detections as out-of-water when no pool water is visible.

    In dry scenes (such as an indoor webcam), the generic person model is used
    to cleanly detect each individual person, preventing two people from being
    merged into a single inaccurate box.
    """
    if not enabled or has_visible_pool_water(frame_bgr, min_fraction):
        return result

    out_of_water_id = _class_id(result, "Person out of water")
    if out_of_water_id is None:
        out_of_water_id = 1

    person_model = get_person_model()
    if person_model is not None and frame_bgr is not None:
        try:
            person_res = person_model.predict(frame_bgr, classes=[0], conf=0.25, verbose=False)[0]
            if person_res.boxes is not None and len(person_res.boxes) > 0:
                data = person_res.boxes.data.clone()
                data[:, -1] = float(out_of_water_id)
                result.boxes.data = data
                return result
        except Exception:
            pass

    if result.boxes is not None and len(result.boxes) > 0:
        result.boxes.data = result.boxes.data.clone()
        result.boxes.data[:, -1] = float(out_of_water_id)

    return result


def combine_drowning_and_person_detections(result, frame_bgr):
    """Ensemble person detection to ensure every person in water/deck is detected."""
    person_model = get_person_model()
    if person_model is None or frame_bgr is None:
        return result

    try:
        person_res = person_model.predict(frame_bgr, classes=[0], conf=0.25, verbose=False)[0]
        if person_res.boxes is None or len(person_res.boxes) == 0:
            return result

        out_of_water_id = _class_id(result, "Person out of water")
        if out_of_water_id is None:
            out_of_water_id = 1

        existing_coords = (
            result.boxes.xyxy.cpu().numpy()
            if result.boxes is not None and len(result.boxes) > 0
            else np.empty((0, 4))
        )
        additional_boxes = []

        for p_box in person_res.boxes.data:
            p_coords = p_box[:4].cpu().numpy()
            p_area = max(0.0, p_coords[2] - p_coords[0]) * max(0.0, p_coords[3] - p_coords[1])
            overlaps = False

            for e_coords in existing_coords:
                inter_x = max(0.0, min(p_coords[2], e_coords[2]) - max(p_coords[0], e_coords[0]))
                inter_y = max(0.0, min(p_coords[3], e_coords[3]) - max(p_coords[1], e_coords[1]))
                inter_area = inter_x * inter_y
                if inter_area > 0 and p_area > 0:
                    e_area = max(0.0, e_coords[2] - e_coords[0]) * max(0.0, e_coords[3] - e_coords[1])
                    min_area = min(p_area, e_area)
                    io_min = inter_area / min_area if min_area > 0 else 0
                    iou = inter_area / (p_area + e_area - inter_area)
                    # If high IoU or one box is inside another, it describes the same person
                    if iou > 0.20 or io_min > 0.30:
                        overlaps = True
                        break

            if not overlaps:
                new_box = p_box.clone()
                new_box[-1] = float(out_of_water_id)
                additional_boxes.append(new_box)

        if additional_boxes:
            import torch

            if result.boxes is not None and len(result.boxes) > 0:
                all_boxes = torch.cat(
                    [result.boxes.data] + [b.unsqueeze(0) for b in additional_boxes], dim=0
                )
            else:
                all_boxes = torch.stack(additional_boxes, dim=0)

            result.boxes.data = all_boxes
    except Exception:
        pass

    return result


def remove_conflicting_overlaps(result, iou_threshold=0.65):
    """Keep one label when several class boxes describe the same person.

    Ultralytics performs non-maximum suppression per class. A model can
    therefore return overlapping Swimming, Drowning, and out-of-water boxes for
    one person. This second pass retains the most confident box when boxes
    have high overlap (same person), while preserving separate individuals.
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
            min_area = min(candidate_area, accepted_area)
            io_min = intersection / min_area if min_area > 0 else 0
            iou = intersection / union if union > 0 else 0
            if iou >= iou_threshold or io_min >= 0.40:
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
