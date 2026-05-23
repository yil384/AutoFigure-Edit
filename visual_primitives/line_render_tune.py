"""Generic line rendering adjustments for visual primitive programs."""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any


LINE_RENDER_TUNE_VERSION = "line-render-tune-0.1"


def tune_augmented_line_rendering(
    program: dict[str, Any],
    *,
    max_confidence: float = 0.58,
    stroke: str = "#4f5b66",
    width_scale: float = 0.75,
    min_width: float = 0.65,
    target_source: str = "cv_line_augment",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Soften lower-confidence augmented lines without removing geometry."""
    updated = copy.deepcopy(program)
    operations: list[dict[str, Any]] = []
    for primitive in updated.get("primitives", []):
        if primitive.get("type") != "edge":
            continue
        if primitive.get("source") != target_source:
            continue
        confidence = _confidence(primitive.get("confidence"))
        if confidence > max_confidence:
            continue
        style = primitive.setdefault("style", {})
        before_style = copy.deepcopy(style)
        before_width = float(style.get("stroke_width") or 1.0)
        style["stroke"] = stroke
        style["stroke_width"] = round(max(min_width, before_width * width_scale), 3)
        operations.append({
            "action": "soften_augmented_line",
            "primitive_id": primitive.get("id"),
            "confidence": confidence,
            "path": primitive.get("path"),
            "style_before": before_style,
            "style_after": copy.deepcopy(style),
        })

    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": LINE_RENDER_TUNE_VERSION,
        "config": {
            "max_confidence": max_confidence,
            "stroke": stroke,
            "width_scale": width_scale,
            "min_width": min_width,
            "target_source": target_source,
        },
        "counts": {
            "operations": len(operations),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["line_render_tune"] = report["config"]
    return updated, report


def _confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:
        return conf / 100.0
    return conf
