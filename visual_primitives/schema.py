"""Schema adapters for visual-primitive draw.io reconstruction.

The project-level intermediate representation is intentionally explicit:
every primitive has a coordinate anchor, a type, a source detector, and enough
style metadata for a VLM or rule-based pass to edit it without referring to
ambiguous language such as "the left arrow".
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROGRAM_VERSION = "vp-drawio-0.1"


def load_ledger(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_program(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_program(program: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(program, indent=2, ensure_ascii=True))


def _bbox_area(bbox: list[float] | tuple[float, ...]) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max(1.0, _bbox_area(a))


def _path_bbox(path: list[list[float]] | list[tuple[float, float]]) -> list[float]:
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    return [min(xs), min(ys), max(xs), max(ys)]


def _path_length(path: list[list[float]] | list[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(path[:-1], path[1:]):
        total += math.hypot(float(p1[0]) - float(p0[0]),
                            float(p1[1]) - float(p0[1]))
    return total


def _points_from_line(line: dict[str, Any]) -> list[list[float]]:
    if "path" in line:
        return [[float(x), float(y)] for x, y in line["path"]]
    x0, y0, x1, y1 = [float(v) for v in line["points"]]
    return [[x0, y0], [x1, y1]]


def _round_bbox(bbox) -> list[float]:
    return [round(float(v), 3) for v in bbox]


def _round_path(path) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in path]


def _primitive_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:04d}"


def to_visual_primitive_program(ledger: dict[str, Any]) -> dict[str, Any]:
    primitives: list[dict[str, Any]] = []
    source = ledger.get("source")
    width = int(ledger.get("width", 0))
    height = int(ledger.get("height", 0))
    raw = ledger.get("primitives", {})

    for i, rect in enumerate(raw.get("rectangles", []), start=1):
        primitives.append({
            "id": _primitive_id("region", i),
            "type": "region",
            "role": "panel_or_container",
            "bbox": _round_bbox(rect["bbox"]),
            "style": {
                "fill": rect.get("fill"),
                "stroke": rect.get("stroke"),
                "rounded": True,
            },
            "source": rect.get("source", "cv_fill_or_border"),
            "confidence": rect.get("confidence"),
        })

    for i, item in enumerate(raw.get("text", []), start=1):
        primitives.append({
            "id": _primitive_id("text", i),
            "type": "text",
            "role": "label",
            "bbox": _round_bbox(item["bbox"]),
            "text": item.get("text", ""),
            "style": {
                "font_size": item.get("font_size"),
                "bold": bool(item.get("bold")),
                "align": item.get("align", "center"),
            },
            "source": "ocr",
            "confidence": item.get("conf"),
        })

    for i, line in enumerate(raw.get("lines", []), start=1):
        path = _points_from_line(line)
        primitives.append({
            "id": _primitive_id("edge", i),
            "type": "edge",
            "role": "connector" if line.get("orient") == "P" else "axis_or_connector",
            "bbox": _round_bbox(_path_bbox(path)),
            "path": _round_path(path),
            "style": {
                "stroke": line.get("stroke"),
                "stroke_width": line.get("width"),
                "arrow_start": bool(line.get("arrow_start")),
                "arrow_end": bool(line.get("arrow_end")),
            },
            "source": line.get("source", "cv_line"),
            "length": round(_path_length(path), 3),
            "confidence": line.get("confidence"),
        })

    for i, shape in enumerate(raw.get("native_shapes", []), start=1):
        primitives.append({
            "id": _primitive_id("shape", i),
            "type": "shape",
            "role": "symbol_or_mark",
            "shape": shape.get("shape", "rectangle"),
            "bbox": _round_bbox(shape["bbox"]),
            "style": {
                "fill": shape.get("fill"),
                "stroke": shape.get("stroke"),
                "direction": shape.get("direction"),
            },
            "source": shape.get("source", "cv_native_shape"),
            "confidence": shape.get("confidence"),
        })

    relations = _infer_containment_relations(primitives)
    return {
        "version": PROGRAM_VERSION,
        "source": source,
        "canvas": {"width": width, "height": height},
        "coordinate_space": {
            "type": "image_pixels",
            "normalized": False,
            "origin": "top_left",
        },
        "hard_constraints": {
            "drawio_native_only": True,
            "forbid_source_overlay": True,
            "forbid_raster_cells": True,
            "forbid_stencils": True,
        },
        "counts": {
            "regions": sum(1 for p in primitives if p["type"] == "region"),
            "texts": sum(1 for p in primitives if p["type"] == "text"),
            "edges": sum(1 for p in primitives if p["type"] == "edge"),
            "shapes": sum(1 for p in primitives if p["type"] == "shape"),
            "total": len(primitives),
        },
        "primitives": primitives,
        "relations": relations,
        "source_ledger_counts": ledger.get("counts", {}),
    }


def _infer_containment_relations(primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions = [p for p in primitives if p["type"] == "region"]
    children = [p for p in primitives if p["type"] in {"text", "shape"}]
    relations = []
    for region in regions:
        rb = region["bbox"]
        for child in children:
            cb = child["bbox"]
            if _bbox_overlap_fraction(cb, rb) >= 0.82:
                relations.append({
                    "type": "contains",
                    "from": region["id"],
                    "to": child["id"],
                })
    return relations
