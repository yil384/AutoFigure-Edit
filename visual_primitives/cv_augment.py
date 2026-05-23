"""Augment visual primitive programs with missing native CV primitives."""
from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any


CV_AUGMENT_VERSION = "cv-augment-0.1"


def augment_program_from_cv_evidence(
    program: dict[str, Any],
    evidence: dict[str, Any],
    *,
    add_text: bool = False,
    add_lines: bool = True,
    add_regions: bool = False,
    max_text: int = 45,
    max_lines: int = 70,
    max_regions: int = 18,
    min_text_confidence: float = 0.72,
    min_line_length: float = 26.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add high-confidence missing evidence as pure-native primitives."""
    updated = copy.deepcopy(program)
    primitives = updated.setdefault("primitives", [])
    operations: list[dict[str, Any]] = []

    if add_regions:
        operations.extend(_add_missing_regions(
            primitives, evidence.get("regions", []), max_regions=max_regions))
    if add_lines:
        operations.extend(_add_missing_lines(
            primitives, evidence, max_lines=max_lines,
            min_line_length=min_line_length))
    if add_text:
        text_evidence = evidence.get("text_lines") or evidence.get("text_regions", [])
        operations.extend(_add_missing_text(
            primitives, text_evidence, max_text=max_text,
            min_confidence=min_text_confidence))

    _refresh_counts(updated)
    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": CV_AUGMENT_VERSION,
        "source_evidence": evidence.get("source"),
        "evidence_counts": evidence.get("counts", {}),
        "config": {
            "add_text": add_text,
            "add_lines": add_lines,
            "add_regions": add_regions,
            "max_text": max_text,
            "max_lines": max_lines,
            "max_regions": max_regions,
            "min_text_confidence": min_text_confidence,
            "min_line_length": min_line_length,
        },
        "counts": {
            "operations": len(operations),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["cv_augment"] = {
        "version": CV_AUGMENT_VERSION,
        "counts": report["counts"],
        "source_evidence": evidence.get("source"),
    }
    return updated, report


def _add_missing_text(
    primitives: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    *,
    max_text: int,
    min_confidence: float,
) -> list[dict[str, Any]]:
    existing_boxes = [p["bbox"] for p in primitives if p.get("type") == "text" and p.get("bbox")]
    existing_norms = [_norm_text(p.get("text", "")) for p in primitives if p.get("type") == "text"]
    ops = []
    ranked = sorted(
        text_regions,
        key=lambda t: (-float(t.get("confidence", 0.0)), t.get("bbox", [0, 0])[1], t.get("bbox", [0, 0])[0]),
    )
    for region in ranked:
        if len(ops) >= max_text:
            break
        text = str(region.get("text", "")).strip()
        norm = _norm_text(text)
        if not _usable_augmented_text(region, norm, text, min_confidence):
            continue
        if float(region.get("confidence", 0.0)) < min_confidence:
            continue
        bbox = _round_bbox(region.get("bbox", [0, 0, 1, 1]))
        if _covered_by_text_boxes(bbox, existing_boxes):
            continue
        if norm and norm in existing_norms and _near_any_text_box(bbox, existing_boxes):
            continue
        primitive = {
            "id": _next_id(primitives, "cv_text"),
            "type": "text",
            "role": "label",
            "bbox": _pad_bbox(bbox, 1.0),
            "text": text,
            "style": {
                "font_size": _font_size_from_bbox(bbox),
                "bold": False,
                "align": "center",
            },
            "source": "cv_text_augment",
            "confidence": round(float(region.get("confidence", 0.0)), 3),
            "alignment": {"cv_text_id": region.get("id")},
        }
        primitives.append(primitive)
        existing_boxes.append(primitive["bbox"])
        existing_norms.append(norm)
        ops.append({
            "action": "add_cv_text",
            "primitive_id": primitive["id"],
            "cv_text_id": region.get("id"),
            "bbox": primitive["bbox"],
            "text": text,
            "confidence": primitive["confidence"],
        })
    return ops


def _add_missing_lines(
    primitives: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    max_lines: int,
    min_line_length: float,
) -> list[dict[str, Any]]:
    existing_paths = [
        _path(p) for p in primitives
        if p.get("type") == "edge"
    ]
    arrows = evidence.get("arrowheads", [])
    text_boxes = [
        item.get("bbox")
        for item in (evidence.get("text_lines") or evidence.get("text_regions") or [])
        if item.get("bbox")
    ]
    lines = [
        line for line in evidence.get("line_segments", [])
        if line.get("role_hint") not in {"text_stroke", "panel_border"}
        and float(line.get("length", 0.0)) >= min_line_length
        and not _line_crosses_text(line, text_boxes)
    ]
    lines.sort(key=lambda line: (
        0 if line.get("role_hint") == "connector_candidate" else 1,
        -float(line.get("confidence", 0.0)),
        -float(line.get("length", 0.0)),
    ))
    ops = []
    for line in lines:
        if len(ops) >= max_lines:
            break
        path = [_round_point(line["p0"]), _round_point(line["p1"])]
        if _line_is_covered(path, existing_paths):
            continue
        primitive = {
            "id": _next_id(primitives, "cv_edge"),
            "type": "edge",
            "role": "connector" if line.get("role_hint") == "connector_candidate" else "axis_or_connector",
            "bbox": _path_bbox(path),
            "path": path,
            "style": {
                "stroke": "#050505",
                "stroke_width": max(0.8, min(1.4, float(line.get("detector_width", 1.0)))),
                "arrow_start": False,
                "arrow_end": False,
            },
            "source": "cv_line_augment",
            "length": round(float(line.get("length", 0.0)), 3),
            "confidence": round(float(line.get("confidence", 0.0)), 3),
            "alignment": {"cv_line_id": line.get("id")},
        }
        _apply_arrow_evidence(primitive, line, arrows)
        primitives.append(primitive)
        existing_paths.append(path)
        ops.append({
            "action": "add_cv_line",
            "primitive_id": primitive["id"],
            "cv_line_id": line.get("id"),
            "role_hint": line.get("role_hint"),
            "path": path,
            "confidence": primitive["confidence"],
        })
    return ops


def _add_missing_regions(
    primitives: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    *,
    max_regions: int,
) -> list[dict[str, Any]]:
    existing_boxes = [p["bbox"] for p in primitives if p.get("type") == "region" and p.get("bbox")]
    ranked = sorted(
        regions,
        key=lambda r: (-float(r.get("confidence", 0.0)), -_bbox_area(r.get("bbox", [0, 0, 0, 0]))),
    )
    ops = []
    for region in ranked:
        if len(ops) >= max_regions:
            break
        bbox = _round_bbox(region.get("bbox", [0, 0, 1, 1]))
        area = _bbox_area(bbox)
        if area < 650 or float(region.get("confidence", 0.0)) < 0.78:
            continue
        if _covered_by_boxes(bbox, existing_boxes, iou_threshold=0.42):
            continue
        primitive = {
            "id": _next_id(primitives, "cv_region"),
            "type": "region",
            "role": "panel_or_container",
            "bbox": bbox,
            "style": {
                "fill": "#eaf2f7",
                "stroke": "#8f9cac",
                "rounded": True,
            },
            "source": "cv_region_augment",
            "confidence": round(float(region.get("confidence", 0.0)), 3),
            "alignment": {"cv_region_id": region.get("id")},
        }
        primitives.append(primitive)
        existing_boxes.append(bbox)
        ops.append({
            "action": "add_cv_region",
            "primitive_id": primitive["id"],
            "cv_region_id": region.get("id"),
            "bbox": bbox,
            "confidence": primitive["confidence"],
        })
    return ops


def _apply_arrow_evidence(
    primitive: dict[str, Any],
    line: dict[str, Any],
    arrows: list[dict[str, Any]],
) -> None:
    if line.get("role_hint") == "panel_border":
        return
    style = primitive.setdefault("style", {})
    path = primitive.get("path", [])
    for arrow in arrows:
        if arrow.get("nearest_line_id") != line.get("id"):
            continue
        centroid = arrow.get("centroid")
        if not centroid or len(path) < 2:
            continue
        if _point_dist(centroid, path[0]) <= 16:
            style["arrow_start"] = True
        if _point_dist(centroid, path[-1]) <= 16:
            style["arrow_end"] = True


def _line_is_covered(path: list[list[float]], existing_paths: list[list[list[float]]]) -> bool:
    orient = _line_orientation(path[0], path[-1])
    for other in existing_paths:
        if len(other) < 2 or _line_orientation(other[0], other[-1]) != orient:
            continue
        if orient == "horizontal":
            axis_dist = abs(((path[0][1] + path[-1][1]) / 2.0) - ((other[0][1] + other[-1][1]) / 2.0))
            overlap = _interval_overlap_fraction([path[0][0], path[-1][0]], [other[0][0], other[-1][0]])
        elif orient == "vertical":
            axis_dist = abs(((path[0][0] + path[-1][0]) / 2.0) - ((other[0][0] + other[-1][0]) / 2.0))
            overlap = _interval_overlap_fraction([path[0][1], path[-1][1]], [other[0][1], other[-1][1]])
        else:
            axis_dist = min(
                (_point_dist(path[0], other[0]) + _point_dist(path[-1], other[-1])) / 2.0,
                (_point_dist(path[0], other[-1]) + _point_dist(path[-1], other[0])) / 2.0,
            )
            overlap = _bbox_iou(_path_bbox(path, pad=5), _path_bbox(other, pad=5))
        if axis_dist <= 5.0 and overlap >= 0.58:
            return True
    return False


def _line_crosses_text(line: dict[str, Any], text_boxes: list[list[float]]) -> bool:
    p0 = line.get("p0")
    p1 = line.get("p1")
    if not p0 or not p1:
        return False
    path = [_round_point(p0), _round_point(p1)]
    orient = _line_orientation(path[0], path[-1])
    for box in text_boxes:
        x0, y0, x1, y1 = [float(v) for v in box]
        if orient == "horizontal":
            y = (path[0][1] + path[-1][1]) / 2.0
            interior_top = y0 + min(3.0, max(1.0, (y1 - y0) * 0.18))
            interior_bottom = y1 - min(3.0, max(1.0, (y1 - y0) * 0.18))
            if not (interior_top <= y <= interior_bottom):
                continue
            overlap = _interval_overlap_fraction(
                [path[0][0], path[-1][0]], [x0, x1])
            if overlap >= 0.22:
                return True
        elif orient == "vertical":
            x = (path[0][0] + path[-1][0]) / 2.0
            interior_left = x0 + min(3.0, max(1.0, (x1 - x0) * 0.08))
            interior_right = x1 - min(3.0, max(1.0, (x1 - x0) * 0.08))
            if not (interior_left <= x <= interior_right):
                continue
            overlap = _interval_overlap_fraction(
                [path[0][1], path[-1][1]], [y0, y1])
            if overlap >= 0.22:
                return True
        else:
            if _bbox_iou(_path_bbox(path, pad=2), [x0, y0, x1, y1]) >= 0.12:
                return True
    return False


def _path(primitive: dict[str, Any]) -> list[list[float]]:
    raw = primitive.get("path")
    if raw:
        return [[float(x), float(y)] for x, y in raw]
    bbox = primitive.get("bbox", [0, 0, 1, 1])
    return [[float(bbox[0]), float(bbox[1])], [float(bbox[2]), float(bbox[3])]]


def _line_orientation(p0: list[float], p1: list[float], tolerance: float = 8.0) -> str:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    angle = abs(((math.degrees(math.atan2(dy, dx)) + 90) % 180) - 90)
    if angle <= tolerance:
        return "horizontal"
    if abs(angle - 90) <= tolerance:
        return "vertical"
    return "diagonal"


def _covered_by_boxes(bbox: list[float], boxes: list[list[float]], *, iou_threshold: float) -> bool:
    return any(_bbox_iou(bbox, other) >= iou_threshold for other in boxes)


def _covered_by_text_boxes(bbox: list[float], boxes: list[list[float]]) -> bool:
    cx, cy = _bbox_center(bbox)
    for other in boxes:
        if _bbox_iou(bbox, other) >= 0.18:
            return True
        if _overlap_fraction(bbox, other) >= 0.42:
            return True
        ox0, oy0, ox1, oy1 = _pad_bbox(other, 4.0)
        if ox0 <= cx <= ox1 and oy0 <= cy <= oy1:
            return True
    return False


def _near_any_text_box(bbox: list[float], boxes: list[list[float]]) -> bool:
    cx, cy = _bbox_center(bbox)
    for other in boxes:
        ox, oy = _bbox_center(other)
        if math.hypot(cx - ox, cy - oy) <= max(12.0, _bbox_diag(bbox) * 0.7):
            return True
    return False


def _usable_text(norm: str, raw: str) -> bool:
    if len(norm) >= 2:
        return True
    return raw.strip() in {"X", "Z", "I", "V", "+", "-"}


def _usable_augmented_text(
    region: dict[str, Any],
    norm: str,
    raw: str,
    min_confidence: float,
) -> bool:
    text = raw.strip()
    if not _usable_text(norm, text):
        return False
    if text.startswith("(") and not text.endswith(")"):
        return False
    if text.endswith("-"):
        return False
    if text in {"+", "-"}:
        return False
    words = re.findall(r"[A-Za-z0-9]+", text)
    word_count = int(region.get("word_count") or len(words) or 1)
    conf = float(region.get("confidence", 0.0))
    if words and _looks_like_noise_tail(words):
        return False
    if word_count <= 1:
        allow_short = {
            "AI", "CV", "RL", "QFT", "ZX", "GNN", "LLM", "X", "Z", "2X", "2Y", "X/Z",
        }
        if len(norm) <= 3 and text not in allow_short:
            return False
        if text.islower():
            return False
        if conf < max(min_confidence, 0.72):
            return False
        if words and len(words[0]) <= 4 and any(ch.islower() for ch in words[0]):
            return False
    bbox = region.get("bbox", [0, 0, 1, 1])
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    if word_count <= 1 and len(norm) <= 5 and width / height > max(2.2, len(norm) * 0.8):
        return False
    return True


def _looks_like_noise_tail(words: list[str]) -> bool:
    if len(words) < 2:
        return False
    tail = words[-1]
    if tail in {"X", "Z", "AI", "CV", "RL", "ZX"}:
        return False
    return len(tail) <= 2 and tail.islower()


def _font_size_from_bbox(bbox: list[float]) -> int:
    height = max(5.0, float(bbox[3]) - float(bbox[1]))
    return int(max(6, min(13, round(height * 0.72))))


def _next_id(primitives: list[dict[str, Any]], prefix: str) -> str:
    existing = {p.get("id") for p in primitives}
    i = 1
    while f"{prefix}_{i:04d}" in existing:
        i += 1
    return f"{prefix}_{i:04d}"


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _point_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _bbox_diag(bbox: list[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return math.hypot(x1 - x0, y1 - y0)


def _bbox_area(bbox: list[float]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1.0, union)


def _overlap_fraction(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / max(1.0, _bbox_area(a))


def _interval_overlap_fraction(a, b) -> float:
    a0, a1 = sorted([float(a[0]), float(a[1])])
    b0, b1 = sorted([float(b[0]), float(b[1])])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1.0, min(a1 - a0, b1 - b0))


def _path_bbox(path: list[list[float]], pad: float = 0.0) -> list[float]:
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    return _round_bbox([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad])


def _pad_bbox(bbox: list[float], pad: float) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return _round_bbox([x0 - pad, y0 - pad, x1 + pad, y1 + pad])


def _round_bbox(bbox) -> list[float]:
    return [round(float(v), 3) for v in bbox]


def _round_point(point) -> list[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
