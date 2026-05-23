"""Text cleanup passes for visual primitive programs."""
from __future__ import annotations

import copy
import re
from typing import Any


def remove_contained_duplicate_text(
    program: dict[str, Any],
    *,
    min_overlap_fraction: float = 0.15,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove overlapping short text when a nearby longer text contains it."""
    updated = copy.deepcopy(program)
    primitives = updated.get("primitives", [])
    texts = [
        primitive for primitive in primitives
        if primitive.get("type") == "text"
        and primitive.get("bbox")
        and _norm_text(primitive.get("text", ""))
    ]
    remove_ids: set[str] = set()
    operations = []
    for long_text in texts:
        long_norm = _norm_text(long_text.get("text", ""))
        if len(_tokens(long_text.get("text", ""))) < 2:
            continue
        for short_text in texts:
            if short_text is long_text:
                continue
            short_norm = _norm_text(short_text.get("text", ""))
            if not short_norm or short_norm == long_norm:
                continue
            if short_text.get("id") in remove_ids:
                continue
            if short_norm not in long_norm:
                continue
            if _overlap_fraction(short_text["bbox"], long_text["bbox"]) < min_overlap_fraction:
                continue
            if not _centers_near(short_text["bbox"], long_text["bbox"]):
                continue
            remove_ids.add(str(short_text["id"]))
            operations.append({
                "action": "remove_contained_duplicate_text",
                "removed_id": short_text.get("id"),
                "kept_id": long_text.get("id"),
                "removed_text": short_text.get("text"),
                "kept_text": long_text.get("text"),
            })

    if not remove_ids:
        return updated, {
            "version": "text-cleanup-0.1",
            "counts": {"operations": 0},
            "operations": [],
        }

    updated["primitives"] = [
        primitive for primitive in primitives
        if str(primitive.get("id")) not in remove_ids
    ]
    _refresh_counts(updated)
    report = {
        "version": "text-cleanup-0.1",
        "counts": {
            "operations": len(operations),
            "removed_texts": len(remove_ids),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["text_cleanup"] = report["counts"]
    return updated, report


def remove_noise_text_fragments(
    program: dict[str, Any],
    *,
    max_font_size: int = 6,
    min_confidence: float = 85.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove tiny standalone OCR fragments that read like noise."""
    updated = copy.deepcopy(program)
    primitives = updated.get("primitives", [])
    remove_ids: set[str] = set()
    operations = []
    for primitive in primitives:
        if primitive.get("type") != "text":
            continue
        if str(primitive.get("source", "")) != "ocr":
            continue
        text = str(primitive.get("text", "")).strip()
        if not text:
            continue
        font_size = int(primitive.get("style", {}).get("font_size") or 9)
        if font_size > max_font_size:
            continue
        norm = _norm_text(text)
        if not norm or _allowed_short_text(text):
            continue
        confidence = _confidence_0_100(primitive.get("confidence"))
        is_tiny_lower = len(norm) <= 3 and text.islower()
        is_tiny_number = len(norm) <= 2 and norm.isdigit()
        is_weak_short = len(norm) <= 4 and confidence < min_confidence
        if not (is_tiny_lower or is_tiny_number or is_weak_short):
            continue
        remove_ids.add(str(primitive.get("id")))
        operations.append({
            "action": "remove_noise_text_fragment",
            "removed_id": primitive.get("id"),
            "removed_text": text,
            "font_size": font_size,
            "confidence": confidence,
            "bbox": primitive.get("bbox"),
        })

    if not remove_ids:
        return updated, {
            "version": "text-cleanup-0.1",
            "counts": {"operations": 0},
            "operations": [],
        }

    updated["primitives"] = [
        primitive for primitive in primitives
        if str(primitive.get("id")) not in remove_ids
    ]
    _refresh_counts(updated)
    report = {
        "version": "text-cleanup-0.1",
        "counts": {
            "operations": len(operations),
            "removed_texts": len(remove_ids),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["noise_text_cleanup"] = report["counts"]
    return updated, report


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _allowed_short_text(text: str) -> bool:
    stripped = text.strip()
    return stripped in {
        "AI", "CV", "RL", "QFT", "ZX", "GNN", "LLM",
        "X", "Z", "2X", "2Y", "X/Z",
        "1-3", "4-6", "7-9",
    }


def _confidence_0_100(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if 0.0 <= conf <= 1.0:
        return conf * 100.0
    return conf


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text))


def _centers_near(a: list[float], b: list[float]) -> bool:
    ax, ay = _center(a)
    bx, by = _center(b)
    return abs(ax - bx) <= max(_width(a), _width(b)) * 0.8 and abs(ay - by) <= max(_height(a), _height(b)) * 0.8


def _center(bbox: list[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _width(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0]))


def _height(bbox: list[float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _area(bbox: list[float]) -> float:
    return _width(bbox) * _height(bbox)


def _overlap_fraction(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max(1.0, _area(a))


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
