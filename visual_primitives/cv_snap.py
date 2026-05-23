"""Snap visual primitive programs to measured CV geometry evidence.

This pass is deliberately conservative: it only moves existing native
primitives when the CV evidence strongly agrees on the same region, line, text
box, or guide. It never introduces raster/image/stencil cells.
"""
from __future__ import annotations

import copy
import difflib
import math
import re
from collections import Counter
from typing import Any


CV_SNAP_VERSION = "cv-snap-0.1"


def snap_program_to_cv_evidence(
    program: dict[str, Any],
    evidence: dict[str, Any],
    *,
    snap_regions: bool = True,
    snap_text: bool = True,
    snap_edges: bool = True,
    region_min_score: float = 0.64,
    text_min_score: float = 0.62,
    edge_min_score: float = 0.68,
    guide_tolerance: float = 5.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a CV-snapped copy of a visual primitive program and a report."""
    updated = copy.deepcopy(program)
    primitives = updated.get("primitives", [])
    index = _EvidenceIndex(evidence)
    operations: list[dict[str, Any]] = []

    for primitive in primitives:
        ptype = primitive.get("type")
        if snap_regions and ptype == "region":
            op = _snap_region(primitive, index, region_min_score, guide_tolerance)
        elif snap_text and ptype == "text":
            op = _snap_text(primitive, index, text_min_score)
        elif snap_edges and ptype == "edge":
            op = _snap_edge(primitive, index, edge_min_score, guide_tolerance)
        else:
            op = None
        if op:
            operations.append(op)

    _refresh_counts(updated)
    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": CV_SNAP_VERSION,
        "source_evidence": evidence.get("source"),
        "evidence_counts": evidence.get("counts", {}),
        "config": {
            "snap_regions": snap_regions,
            "snap_text": snap_text,
            "snap_edges": snap_edges,
            "region_min_score": region_min_score,
            "text_min_score": text_min_score,
            "edge_min_score": edge_min_score,
            "guide_tolerance": guide_tolerance,
        },
        "counts": {
            "operations": len(operations),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["cv_snap"] = {
        "version": CV_SNAP_VERSION,
        "counts": report["counts"],
        "source_evidence": evidence.get("source"),
    }
    return updated, report


class _EvidenceIndex:
    def __init__(self, evidence: dict[str, Any]):
        self.regions = list(evidence.get("regions", []))
        self.text_regions = list(evidence.get("text_lines", [])) + list(evidence.get("text_regions", []))
        self.lines = [
            line for line in evidence.get("line_segments", [])
            if line.get("role_hint") != "text_stroke"
        ]
        self.arrows = list(evidence.get("arrowheads", []))
        self.junctions = list(evidence.get("junctions", []))
        guides = evidence.get("snap_guides", {})
        self.guides_x = list(guides.get("x", []))
        self.guides_y = list(guides.get("y", []))
        self.lines_by_id = {line.get("id"): line for line in self.lines}


def _snap_region(
    primitive: dict[str, Any],
    index: _EvidenceIndex,
    min_score: float,
    guide_tolerance: float,
) -> dict[str, Any] | None:
    bbox = primitive.get("bbox")
    if not bbox:
        return None
    match = _best_region_match(bbox, index.regions, min_score)
    if match:
        cv_region, score = match
        before = list(bbox)
        after = _round_bbox(cv_region["bbox"])
        if _bbox_delta(before, after) < 0.75:
            return None
        primitive["bbox"] = after
        primitive["source"] = _source_with_suffix(primitive, "cv_snap")
        primitive.setdefault("alignment", {})["cv_region_id"] = cv_region.get("id")
        return {
            "primitive_id": primitive.get("id"),
            "type": "region",
            "action": "snap_region_to_cv_region",
            "score": round(score, 4),
            "cv_region_id": cv_region.get("id"),
            "before": before,
            "after": after,
        }

    snapped, guide_ids = _snap_bbox_to_guides(bbox, index, guide_tolerance)
    if snapped and _bbox_delta(bbox, snapped) >= 0.75:
        before = list(bbox)
        primitive["bbox"] = snapped
        primitive["source"] = _source_with_suffix(primitive, "cv_guide_snap")
        primitive.setdefault("alignment", {})["cv_guide_ids"] = guide_ids
        return {
            "primitive_id": primitive.get("id"),
            "type": "region",
            "action": "snap_region_to_guides",
            "cv_guide_ids": guide_ids,
            "before": before,
            "after": snapped,
        }
    return None


def _snap_text(
    primitive: dict[str, Any],
    index: _EvidenceIndex,
    min_score: float,
) -> dict[str, Any] | None:
    bbox = primitive.get("bbox")
    if not bbox:
        return None
    match = _best_text_match(
        primitive.get("text", ""), bbox, index.text_regions, min_score)
    if not match:
        return None
    cv_text, score = match
    before = list(bbox)
    after = _pad_bbox(cv_text["bbox"], pad=1.0)
    before_text = str(primitive.get("text", ""))
    after_text = str(cv_text.get("text", "")).strip()
    text_changed = _should_correct_text(before_text, after_text, cv_text)
    if _bbox_delta(before, after) < 0.75 and not text_changed:
        return None
    if _bbox_delta(before, after) >= 0.75:
        primitive["bbox"] = after
    if text_changed:
        primitive["text"] = after_text
    primitive["source"] = _source_with_suffix(primitive, "cv_text_snap")
    primitive.setdefault("alignment", {})["cv_text_id"] = cv_text.get("id")
    return {
        "primitive_id": primitive.get("id"),
        "type": "text",
        "action": "snap_text_to_ocr_region" if not text_changed else "snap_and_correct_text",
        "score": round(score, 4),
        "cv_text_id": cv_text.get("id"),
        "before": before,
        "after": after,
        "before_text": before_text,
        "after_text": after_text if text_changed else before_text,
    }


def _snap_edge(
    primitive: dict[str, Any],
    index: _EvidenceIndex,
    min_score: float,
    guide_tolerance: float,
) -> dict[str, Any] | None:
    path = _path(primitive)
    if len(path) < 2 or _path_length(path) < 9:
        return None
    before = _round_path(path)

    if len(path) == 2:
        match = _best_line_match(path, index.lines, min_score)
        if match:
            cv_line, score, reversed_order = match
            snapped_path = [
                list(cv_line["p1"] if reversed_order else cv_line["p0"]),
                list(cv_line["p0"] if reversed_order else cv_line["p1"]),
            ]
            snapped_path = _round_path(snapped_path)
            _set_edge_path(primitive, snapped_path)
            _apply_arrow_evidence(primitive, cv_line, snapped_path, index)
            primitive["source"] = _source_with_suffix(primitive, "cv_line_snap")
            primitive.setdefault("alignment", {})["cv_line_id"] = cv_line.get("id")
            return {
                "primitive_id": primitive.get("id"),
                "type": "edge",
                "action": "snap_edge_to_cv_line",
                "score": round(score, 4),
                "cv_line_id": cv_line.get("id"),
                "role_hint": cv_line.get("role_hint"),
                "before": before,
                "after": snapped_path,
                "arrow_start": bool(primitive.get("style", {}).get("arrow_start")),
                "arrow_end": bool(primitive.get("style", {}).get("arrow_end")),
            }

    snapped_path, guide_ids = _snap_path_to_guides(path, index, guide_tolerance)
    if snapped_path and _path_delta(path, snapped_path) >= 0.75:
        _set_edge_path(primitive, snapped_path)
        primitive["source"] = _source_with_suffix(primitive, "cv_guide_snap")
        primitive.setdefault("alignment", {})["cv_guide_ids"] = guide_ids
        return {
            "primitive_id": primitive.get("id"),
            "type": "edge",
            "action": "snap_edge_to_guides",
            "cv_guide_ids": guide_ids,
            "before": before,
            "after": snapped_path,
        }
    return None


def _best_region_match(
    bbox: list[float],
    regions: list[dict[str, Any]],
    min_score: float,
) -> tuple[dict[str, Any], float] | None:
    best: tuple[dict[str, Any], float] | None = None
    for region in regions:
        rb = region.get("bbox")
        if not rb:
            continue
        iou = _bbox_iou(bbox, rb)
        center_score = _center_score(bbox, rb, scale=max(18.0, _bbox_diag(bbox) * 0.24))
        size_score = _size_score(bbox, rb)
        confidence = float(region.get("confidence", 0.5))
        area_ratio = max(_bbox_area(bbox), _bbox_area(rb)) / max(1.0, min(_bbox_area(bbox), _bbox_area(rb)))
        if area_ratio > 1.85 and iou < 0.62:
            continue
        if iou < 0.34 and not (center_score > 0.72 and size_score > 0.62):
            continue
        score = 0.52 * iou + 0.22 * center_score + 0.19 * size_score + 0.07 * confidence
        if score < min_score:
            continue
        if best is None or score > best[1]:
            best = (region, score)
    return best


def _best_text_match(
    text: str,
    bbox: list[float],
    text_regions: list[dict[str, Any]],
    min_score: float,
) -> tuple[dict[str, Any], float] | None:
    norm = _norm_text(text)
    best: tuple[dict[str, Any], float] | None = None
    for region in text_regions:
        rb = region.get("bbox")
        if not rb:
            continue
        tscore = _text_score(norm, _norm_text(region.get("text", "")))
        iou = _bbox_iou(_pad_bbox(bbox, 2.0), _pad_bbox(rb, 2.0))
        center_score = _center_score(bbox, rb, scale=max(12.0, _bbox_diag(bbox) * 0.45))
        size_score = _size_score(bbox, rb)
        if tscore <= 0 and iou < 0.20:
            continue
        score = 0.46 * tscore + 0.24 * iou + 0.20 * center_score + 0.10 * size_score
        if score < min_score:
            continue
        if best is None or score > best[1]:
            best = (region, score)
    return best


def _best_line_match(
    path: list[list[float]],
    lines: list[dict[str, Any]],
    min_score: float,
) -> tuple[dict[str, Any], float, bool] | None:
    orient = _line_orientation(path[0], path[-1])
    best: tuple[dict[str, Any], float, bool] | None = None
    for line in lines:
        if line.get("orientation") != orient:
            continue
        score, reversed_order = _line_match_score(path, line)
        if score < min_score:
            continue
        if best is None or score > best[1]:
            best = (line, score, reversed_order)
    return best


def _line_match_score(path: list[list[float]], line: dict[str, Any]) -> tuple[float, bool]:
    p0, p1 = path[0], path[-1]
    q0, q1 = line["p0"], line["p1"]
    orient = line.get("orientation")
    direct = (_point_dist(p0, q0) + _point_dist(p1, q1)) / 2.0
    reverse = (_point_dist(p0, q1) + _point_dist(p1, q0)) / 2.0
    endpoint_dist = min(direct, reverse)
    reversed_order = reverse < direct
    endpoint_score = max(0.0, 1.0 - endpoint_dist / 26.0)
    primitive_len = _point_dist(p0, p1)
    cv_len = float(line.get("length", 0.0))
    if primitive_len <= 0 or cv_len <= 0:
        return 0.0, reversed_order
    length_ratio = max(primitive_len, cv_len) / max(1.0, min(primitive_len, cv_len))
    if length_ratio > 3.0:
        return 0.0, reversed_order
    if primitive_len < 28.0 and cv_len > primitive_len * 2.15 and endpoint_dist > 7.0:
        return 0.0, reversed_order
    length_score = _length_score(primitive_len, cv_len)
    confidence = float(line.get("confidence", 0.5))

    if orient == "horizontal":
        axis_dist = abs(((p0[1] + p1[1]) / 2.0) - float(q0[1]))
        overlap = _interval_overlap_fraction([p0[0], p1[0]], [q0[0], q1[0]])
        if axis_dist > 8.0 or (overlap < 0.48 and endpoint_dist > 18.0):
            return 0.0, reversed_order
        axis_score = max(0.0, 1.0 - axis_dist / 8.0)
    elif orient == "vertical":
        axis_dist = abs(((p0[0] + p1[0]) / 2.0) - float(q0[0]))
        overlap = _interval_overlap_fraction([p0[1], p1[1]], [q0[1], q1[1]])
        if axis_dist > 8.0 or (overlap < 0.48 and endpoint_dist > 18.0):
            return 0.0, reversed_order
        axis_score = max(0.0, 1.0 - axis_dist / 8.0)
    else:
        expanded_iou = _bbox_iou(_path_bbox(path, pad=4), _line_bbox(line, pad=4))
        if endpoint_dist > 18.0 and expanded_iou < 0.22:
            return 0.0, reversed_order
        overlap = expanded_iou
        axis_score = endpoint_score

    score = (
        0.31 * overlap +
        0.27 * axis_score +
        0.22 * endpoint_score +
        0.13 * length_score +
        0.07 * confidence
    )
    if line.get("role_hint") == "panel_border":
        score += 0.025
    return min(1.0, score), reversed_order


def _apply_arrow_evidence(
    primitive: dict[str, Any],
    cv_line: dict[str, Any],
    path: list[list[float]],
    index: _EvidenceIndex,
) -> None:
    style = primitive.setdefault("style", {})
    if cv_line.get("role_hint") == "panel_border" and not (
        style.get("arrow_start") or style.get("arrow_end")
    ):
        return
    related = [
        arrow for arrow in index.arrows
        if arrow.get("nearest_line_id") == cv_line.get("id")
    ]
    if not related:
        return
    for arrow in related:
        centroid = arrow.get("centroid")
        if not centroid:
            continue
        if _point_dist(centroid, path[0]) <= 16:
            style["arrow_start"] = True
        if _point_dist(centroid, path[-1]) <= 16:
            style["arrow_end"] = True


def _snap_bbox_to_guides(
    bbox: list[float],
    index: _EvidenceIndex,
    tolerance: float,
) -> tuple[list[float] | None, list[str]]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    guide_ids: list[str] = []
    nx0, gx0 = _nearest_guide(x0, index.guides_x, tolerance, min_support=2)
    nx1, gx1 = _nearest_guide(x1, index.guides_x, tolerance, min_support=2)
    ny0, gy0 = _nearest_guide(y0, index.guides_y, tolerance, min_support=2)
    ny1, gy1 = _nearest_guide(y1, index.guides_y, tolerance, min_support=2)
    out = [nx0 if nx0 is not None else x0, ny0 if ny0 is not None else y0,
           nx1 if nx1 is not None else x1, ny1 if ny1 is not None else y1]
    for guide in (gx0, gx1, gy0, gy1):
        if guide:
            guide_ids.append(guide)
    if out[2] - out[0] < 4 or out[3] - out[1] < 4 or not guide_ids:
        return None, []
    return _round_bbox(out), sorted(set(guide_ids))


def _snap_path_to_guides(
    path: list[list[float]],
    index: _EvidenceIndex,
    tolerance: float,
) -> tuple[list[list[float]] | None, list[str]]:
    out = []
    guide_ids: list[str] = []
    for x, y in path:
        nx, gx = _nearest_guide(x, index.guides_x, tolerance, min_support=2)
        ny, gy = _nearest_guide(y, index.guides_y, tolerance, min_support=2)
        out.append([
            float(nx if nx is not None else x),
            float(ny if ny is not None else y),
        ])
        if gx:
            guide_ids.append(gx)
        if gy:
            guide_ids.append(gy)
    if not guide_ids:
        return None, []
    if _path_length(out) < 4:
        return None, []
    return _round_path(out), sorted(set(guide_ids))


def _nearest_guide(
    value: float,
    guides: list[dict[str, Any]],
    tolerance: float,
    *,
    min_support: int,
) -> tuple[float | None, str | None]:
    best = None
    best_dist = tolerance + 1.0
    for guide in guides:
        if int(guide.get("support", 0)) < min_support:
            continue
        dist = abs(float(guide.get("value", 0.0)) - float(value))
        if dist <= tolerance and dist < best_dist:
            best = guide
            best_dist = dist
    if best is None:
        return None, None
    return float(best["value"]), str(best.get("id"))


def _path(primitive: dict[str, Any]) -> list[list[float]]:
    raw = primitive.get("path")
    if raw:
        return [[float(x), float(y)] for x, y in raw]
    bbox = primitive.get("bbox", [0, 0, 1, 1])
    return [[float(bbox[0]), float(bbox[1])], [float(bbox[2]), float(bbox[3])]]


def _set_edge_path(primitive: dict[str, Any], path: list[list[float]]) -> None:
    primitive["path"] = path
    primitive["bbox"] = _path_bbox(path)


def _source_with_suffix(primitive: dict[str, Any], suffix: str) -> str:
    source = str(primitive.get("source", "unknown"))
    return source if suffix in source.split("+") else f"{source}+{suffix}"


def _line_orientation(p0: list[float], p1: list[float], tolerance: float = 8.0) -> str:
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    angle = abs(((math.degrees(math.atan2(dy, dx)) + 90) % 180) - 90)
    if angle <= tolerance:
        return "horizontal"
    if abs(angle - 90) <= tolerance:
        return "vertical"
    return "diagonal"


def _point_dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _path_length(path: list[list[float]]) -> float:
    return sum(_point_dist(a, b) for a, b in zip(path[:-1], path[1:]))


def _path_delta(a: list[list[float]], b: list[list[float]]) -> float:
    if len(a) != len(b):
        return 999.0
    return max((_point_dist(pa, pb) for pa, pb in zip(a, b)), default=0.0)


def _bbox_delta(a: list[float], b: list[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


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


def _center_score(a: list[float], b: list[float], *, scale: float) -> float:
    ac = _bbox_center(a)
    bc = _bbox_center(b)
    return max(0.0, 1.0 - math.hypot(ac[0] - bc[0], ac[1] - bc[1]) / scale)


def _size_score(a: list[float], b: list[float]) -> float:
    aa = max(1.0, _bbox_area(a))
    ba = max(1.0, _bbox_area(b))
    return max(0.0, 1.0 - abs(math.log(aa / ba)) / math.log(3.2))


def _length_score(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(math.log(a / b)) / math.log(3.0))


def _line_bbox(line: dict[str, Any], pad: float = 0.0) -> list[float]:
    return _path_bbox([line["p0"], line["p1"]], pad=pad)


def _path_bbox(path: list[list[float]], pad: float = 0.0) -> list[float]:
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    return _round_bbox([min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad])


def _interval_overlap_fraction(a, b) -> float:
    a0, a1 = sorted([float(a[0]), float(a[1])])
    b0, b1 = sorted([float(b[0]), float(b[1])])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1.0, min(a1 - a0, b1 - b0))


def _pad_bbox(bbox: list[float], pad: float) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return _round_bbox([x0 - pad, y0 - pad, x1 + pad, y1 + pad])


def _round_bbox(bbox) -> list[float]:
    return [round(float(v), 3) for v in bbox]


def _round_path(path) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in path]


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _text_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 3 and len(b) >= 3 and (a in b or b in a):
        return 0.72
    aset = _char_bigrams(a)
    bset = _char_bigrams(b)
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(1, len(aset | bset))


def _should_correct_text(
    current: str,
    candidate: str,
    cv_text: dict[str, Any],
) -> bool:
    current = str(current or "").strip()
    candidate = str(candidate or "").strip()
    if not current or not candidate or current == candidate:
        return False
    if current.startswith("Figure "):
        return False
    if not _plausible_replacement_text(candidate, cv_text):
        return False
    cur_norm = _norm_text(current)
    cand_norm = _norm_text(candidate)
    if len(cur_norm) < 4 or len(cand_norm) < 4:
        return False
    if float(cv_text.get("confidence", 0.0)) < 0.74:
        return False
    ratio = difflib.SequenceMatcher(None, cur_norm, cand_norm).ratio()
    if ratio >= 0.72:
        return True
    if cur_norm in cand_norm:
        return min(len(cur_norm), len(cand_norm)) / max(len(cur_norm), len(cand_norm)) >= 0.62
    return False


def _plausible_replacement_text(candidate: str, cv_text: dict[str, Any]) -> bool:
    text = candidate.strip()
    if any(ch in text for ch in "‘’`~<>"):
        return False
    if text.startswith("(") and not text.endswith(")"):
        return False
    if text.endswith("-"):
        return False
    norm = _norm_text(text)
    if len(norm) <= 3 and text not in {"AI", "CV", "RL", "QFT", "ZX", "GNN", "LLM", "X/Z"}:
        return False
    if text.islower():
        return False
    first_alpha = next((ch for ch in text if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False
    bbox = cv_text.get("bbox", [0, 0, 1, 1])
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    if len(norm) <= 5 and width / height > max(2.2, len(norm) * 0.8):
        return False
    return True


def _char_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text}
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
