"""Generic text rendering adjustments for visual primitive programs."""
from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any


TEXT_RENDER_TUNE_VERSION = "text-render-tune-0.1"


def tune_text_rendering(
    program: dict[str, Any],
    *,
    font_scale: float = 1.0,
    bbox_pad: float = 0.0,
    y_pad: float | None = None,
    min_font_size: int = 6,
    max_font_size: int = 16,
    bold_mode: str = "none",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy with text font and box adjustments.

    This is deliberately content-agnostic: it only uses text geometry and local
    style, so it can be swept automatically without hardcoding figure labels.
    """
    updated = copy.deepcopy(program)
    canvas = updated.get("canvas", {})
    width = float(canvas.get("width") or 0)
    height = float(canvas.get("height") or 0)
    pad_y = bbox_pad if y_pad is None else y_pad
    operations: list[dict[str, Any]] = []
    for primitive in updated.get("primitives", []):
        if primitive.get("type") != "text" or not primitive.get("bbox"):
            continue
        style = primitive.setdefault("style", {})
        before_size = int(style.get("font_size") or 9)
        after_size = int(round(before_size * font_scale))
        after_size = max(min_font_size, min(max_font_size, after_size))
        if after_size != before_size:
            style["font_size"] = after_size

        before_bold = bool(style.get("bold"))
        after_bold = _should_bold(primitive, after_size, bold_mode, before_bold)
        if after_bold != before_bold:
            style["bold"] = after_bold

        before_bbox = list(primitive["bbox"])
        if bbox_pad or pad_y:
            primitive["bbox"] = _pad_bbox(
                before_bbox,
                bbox_pad,
                pad_y,
                width=width,
                height=height,
            )

        if (
            after_size != before_size
            or after_bold != before_bold
            or primitive["bbox"] != before_bbox
        ):
            operations.append({
                "action": "tune_text_rendering",
                "primitive_id": primitive.get("id"),
                "text": primitive.get("text", "")[:80],
                "font_size_before": before_size,
                "font_size_after": after_size,
                "bold_before": before_bold,
                "bold_after": after_bold,
                "bbox_before": before_bbox,
                "bbox_after": primitive["bbox"],
            })

    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": TEXT_RENDER_TUNE_VERSION,
        "config": {
            "font_scale": font_scale,
            "bbox_pad": bbox_pad,
            "y_pad": pad_y,
            "min_font_size": min_font_size,
            "max_font_size": max_font_size,
            "bold_mode": bold_mode,
        },
        "counts": {
            "operations": len(operations),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["text_render_tune"] = report["config"]
    return updated, report


def _should_bold(
    primitive: dict[str, Any],
    font_size: int,
    mode: str,
    current: bool,
) -> bool:
    if mode == "none":
        return current
    if mode == "all":
        return True
    if mode != "headers":
        raise ValueError(f"unknown bold mode: {mode}")
    text = str(primitive.get("text", "")).strip()
    if not text:
        return current
    words = re.findall(r"[A-Za-z0-9]+", text)
    if font_size >= 11 and len(words) <= 6:
        return True
    if len(text) >= 18 and text == text.title():
        return True
    return current


def _pad_bbox(
    bbox: list[float],
    pad_x: float,
    pad_y: float,
    *,
    width: float,
    height: float,
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    if width > 0:
        x0 = max(0.0, x0 - pad_x)
        x1 = min(width, x1 + pad_x)
    else:
        x0 -= pad_x
        x1 += pad_x
    if height > 0:
        y0 = max(0.0, y0 - pad_y)
        y1 = min(height, y1 + pad_y)
    else:
        y0 -= pad_y
        y1 += pad_y
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]
