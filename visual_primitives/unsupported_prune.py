"""Prune weakly supported native primitives using source-image evidence."""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


UNSUPPORTED_PRUNE_VERSION = "unsupported-prune-0.1"


def prune_unsupported_edges(
    program: dict[str, Any],
    source_image: str | Path,
    *,
    support_threshold: float = 0.70,
    mode: str = "all_cv",
    mask_dilate: int = 5,
    stroke_pad: float = 2.0,
    min_length: float = 12.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Delete edge primitives that do not align with source foreground.

    This is a detector-consistency cleanup: it never introduces raster content
    and only removes native edges that the source image does not support.
    """
    if mode not in {"augment", "all_cv", "all_edges"}:
        raise ValueError("mode must be one of: augment, all_cv, all_edges")
    mask = _source_foreground_mask(source_image, dilate=mask_dilate)
    updated = copy.deepcopy(program)
    kept: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    for primitive in updated.get("primitives", []):
        if primitive.get("type") != "edge":
            kept.append(primitive)
            continue
        if not _mode_allows_edge(primitive, mode):
            kept.append(primitive)
            continue
        if float(primitive.get("length") or _path_length(primitive.get("path") or [])) < min_length:
            kept.append(primitive)
            continue
        support = edge_source_support(
            primitive,
            mask,
            stroke_pad=stroke_pad,
        )
        if support < support_threshold:
            operations.append({
                "action": "delete_unsupported_edge",
                "primitive_id": primitive.get("id"),
                "source": primitive.get("source"),
                "bbox": primitive.get("bbox"),
                "length": primitive.get("length"),
                "support": round(float(support), 6),
            })
            continue
        kept.append(primitive)
    updated["primitives"] = kept
    _refresh_counts(updated)
    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": UNSUPPORTED_PRUNE_VERSION,
        "source_image": str(source_image),
        "config": {
            "support_threshold": support_threshold,
            "mode": mode,
            "mask_dilate": mask_dilate,
            "stroke_pad": stroke_pad,
            "min_length": min_length,
        },
        "counts": {
            "operations": len(operations),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["unsupported_prune"] = {
        "version": UNSUPPORTED_PRUNE_VERSION,
        "counts": report["counts"],
        "config": report["config"],
    }
    return updated, report


def edge_source_support(
    primitive: dict[str, Any],
    source_foreground_mask: np.ndarray,
    *,
    stroke_pad: float = 2.0,
) -> float:
    """Return fraction of rendered edge pixels supported by source foreground."""
    path = primitive.get("path") or []
    if len(path) < 2:
        return 1.0
    canvas = np.zeros(source_foreground_mask.shape, dtype=np.uint8)
    stroke_width = float(primitive.get("style", {}).get("stroke_width") or 1.0)
    width = max(1, int(round(stroke_width + stroke_pad)))
    points = [(int(round(float(x))), int(round(float(y)))) for x, y in path]
    for p0, p1 in zip(points[:-1], points[1:]):
        cv2.line(canvas, p0, p1, 255, width)
    pixels = canvas > 0
    if int(pixels.sum()) == 0:
        return 1.0
    return float((source_foreground_mask[pixels] > 0).mean())


def _source_foreground_mask(path: str | Path, *, dilate: int) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    non_white = np.max(255.0 - rgb, axis=2)
    dark = (luma < 170) & (non_white > 15)
    colored = (chroma > 25) & (luma < 245) & (non_white > 10)
    mask = (dark | colored).astype(np.uint8) * 255
    if dilate > 0:
        k = int(max(1, dilate))
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
    return mask


def _mode_allows_edge(primitive: dict[str, Any], mode: str) -> bool:
    if mode == "all_edges":
        return True
    source = str(primitive.get("source") or "")
    if mode == "augment":
        return source == "cv_line_augment"
    return "cv_line" in source or source == "cv_line_augment"


def _path_length(path: list[list[float]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for p0, p1 in zip(path[:-1], path[1:]):
        total += float(np.hypot(float(p1[0]) - float(p0[0]),
                                float(p1[1]) - float(p0[1])))
    return total


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
