"""Derive composition panels from CV evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def derive_panel_regions_from_evidence(
    evidence: dict[str, Any],
    *,
    max_leaf_panels: int = 12,
    min_leaf_area_fraction: float = 0.008,
) -> dict[str, Any]:
    """Return layout and leaf panels suitable for local variant composition.

    Layout panels are broad content clusters inferred from all CV support
    boxes. Leaf panels are high-confidence rectangular regions inside those
    clusters. Composition can then choose different reconstruction variants for
    different parts of the figure without relying on fixed quadrants.
    """
    canvas = evidence.get("canvas") or {}
    width = float(canvas.get("width") or 0)
    height = float(canvas.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("CV evidence has no valid canvas")

    layout_panels = _derive_layout_panels(evidence, width, height)
    leaf_panels = _derive_leaf_panels(
        evidence,
        width,
        height,
        max_leaf_panels=max_leaf_panels,
        min_leaf_area_fraction=min_leaf_area_fraction,
    )
    panels = _dedupe_panels(layout_panels + leaf_panels)
    return {
        "version": "panel-regions-0.1",
        "canvas": {"width": width, "height": height},
        "source_cv_evidence": evidence.get("source"),
        "panel_regions": panels,
        "counts": {
            "layout_panels": sum(
                1 for panel in panels if panel.get("kind") == "layout_cluster"),
            "leaf_panels": sum(
                1 for panel in panels if panel.get("kind") == "leaf_region"),
            "total": len(panels),
        },
    }


def save_panel_regions(report: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=True))


def load_panel_regions(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    return list(payload.get("panel_regions") or payload.get("regions") or [])


def _derive_layout_panels(
    evidence: dict[str, Any],
    width: float,
    height: float,
) -> list[dict[str, Any]]:
    support = _support_boxes(evidence, width, height)
    clusters: dict[str, list[list[float]]] = {
        "layout_top_left": [],
        "layout_top_right": [],
        "layout_bottom_left": [],
        "layout_bottom_right": [],
    }
    for box in support:
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        row = "top" if cy < height / 2.0 else "bottom"
        col = "left" if cx < width / 2.0 else "right"
        clusters[f"layout_{row}_{col}"].append(box)

    panels = []
    for name, boxes in clusters.items():
        if not boxes:
            continue
        box = _union_boxes(boxes, width, height, pad=2.0)
        if _area(box) < width * height * 0.015:
            continue
        panels.append({
            "id": name,
            "bbox": _round_bbox(box),
            "kind": "layout_cluster",
            "source": "cv_support_union",
            "support_count": len(boxes),
            "area": round(_area(box), 3),
        })
    return panels


def _derive_leaf_panels(
    evidence: dict[str, Any],
    width: float,
    height: float,
    *,
    max_leaf_panels: int,
    min_leaf_area_fraction: float,
) -> list[dict[str, Any]]:
    canvas_area = width * height
    candidates = []
    for region in evidence.get("regions", []):
        box = _valid_bbox(region.get("bbox"), width, height)
        if not box:
            continue
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        area = w * h
        confidence = float(region.get("confidence") or 0.0)
        if confidence < 0.45:
            continue
        if w < 50 or h < 35:
            continue
        if area < canvas_area * min_leaf_area_fraction:
            continue
        aspect = w / max(1.0, h)
        if aspect > 9.5 and h < 95:
            continue
        if aspect < 0.12:
            continue
        if y0 > height * 0.90 and h < height * 0.12:
            continue
        candidates.append({
            "id": region.get("id") or f"region_{len(candidates) + 1:04d}",
            "bbox": box,
            "kind": "leaf_region",
            "source": region.get("source", "cv_region"),
            "confidence": confidence,
            "area": area,
        })

    candidates.sort(key=lambda item: (-item["area"], item["bbox"][1], item["bbox"][0]))
    kept: list[dict[str, Any]] = []
    for item in candidates:
        box = item["bbox"]
        if any(_iou(box, other["bbox"]) >= 0.72 for other in kept):
            continue
        renamed = dict(item)
        renamed["id"] = f"leaf_{len(kept) + 1:02d}_{renamed['id']}"
        renamed["bbox"] = _round_bbox(box)
        renamed["area"] = round(float(renamed["area"]), 3)
        kept.append(renamed)
        if len(kept) >= max_leaf_panels:
            break
    return kept


def _support_boxes(
    evidence: dict[str, Any],
    width: float,
    height: float,
) -> list[list[float]]:
    boxes: list[list[float]] = []
    for collection in ("regions", "components"):
        for item in evidence.get(collection, []):
            box = _valid_bbox(item.get("bbox"), width, height)
            if not box:
                continue
            x0, y0, x1, y1 = box
            w = x1 - x0
            h = y1 - y0
            confidence = float(item.get("confidence") or 0.0)
            if confidence < 0.35 or w < 4 or h < 4:
                continue
            if _area(box) < 220:
                continue
            aspect = w / max(1.0, h)
            if aspect > 16 and h < 35:
                continue
            if y0 > height * 0.91 and h < 80:
                continue
            boxes.append(box)

    for item in evidence.get("text_lines", []):
        box = _valid_bbox(item.get("bbox"), width, height)
        if not box:
            continue
        text = str(item.get("text") or "").strip().lower()
        confidence = float(item.get("confidence") or 0.0)
        if confidence < 0.45 or _area(box) < 120:
            continue
        if text.startswith("figure") or box[1] > height * 0.91:
            continue
        boxes.append(box)

    for line in evidence.get("line_segments", []):
        p0 = line.get("p0")
        p1 = line.get("p1")
        if not p0 or not p1:
            continue
        x0 = min(float(p0[0]), float(p1[0]))
        y0 = min(float(p0[1]), float(p1[1]))
        x1 = max(float(p0[0]), float(p1[0]))
        y1 = max(float(p0[1]), float(p1[1]))
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        if length < 25 or y0 > height * 0.91:
            continue
        boxes.append(_clamp_bbox([x0, y0, x1, y1], width, height))
    return boxes


def _dedupe_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for panel in panels:
        box = panel["bbox"]
        if any(_iou(box, other["bbox"]) >= 0.88 for other in out):
            continue
        out.append(panel)
    return out


def _valid_bbox(raw: Any, width: float, height: float) -> list[float] | None:
    if not raw or len(raw) != 4:
        return None
    box = _clamp_bbox([float(v) for v in raw], width, height)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _clamp_bbox(box: list[float], width: float, height: float) -> list[float]:
    x0, y0, x1, y1 = box
    return [
        max(0.0, min(width, x0)),
        max(0.0, min(height, y0)),
        max(0.0, min(width, x1)),
        max(0.0, min(height, y1)),
    ]


def _union_boxes(
    boxes: list[list[float]],
    width: float,
    height: float,
    *,
    pad: float,
) -> list[float]:
    x0 = min(box[0] for box in boxes) - pad
    y0 = min(box[1] for box in boxes) - pad
    x1 = max(box[2] for box in boxes) + pad
    y1 = max(box[3] for box in boxes) + pad
    return _clamp_bbox([x0, y0, x1, y1], width, height)


def _round_bbox(box: list[float]) -> list[float]:
    return [round(float(v), 3) for v in box]


def _area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1]))


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / max(1.0, _area(a) + _area(b) - inter)
