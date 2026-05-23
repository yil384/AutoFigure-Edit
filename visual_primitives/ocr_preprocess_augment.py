"""Add missing text from OCR preprocessing passes as native draw.io text."""
from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from png_to_drawio import _detect_ocr_text_single, _merge_ocr_blocks


OCR_PREPROCESS_AUGMENT_VERSION = "ocr-preprocess-augment-0.1"


def augment_program_with_preprocessed_ocr(
    program: dict[str, Any],
    image_path: str | Path,
    *,
    modes: tuple[str, ...] = ("sharp",),
    psm_values: tuple[int, ...] = (6, 11, 12),
    scale: float = 3.0,
    conf_threshold: float = 25.0,
    min_confidence: float = 70.0,
    max_additions: int = 40,
    allow_standalone_numbers: bool = True,
    similar_text_radius: float = 48.0,
    merge_stacked_labels: bool = False,
    min_support_count: int = 1,
    merged_candidates: list[dict[str, Any]] | None = None,
    candidate_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover high-confidence missing OCR blocks from enhanced source views.

    The image is detector evidence only. The output remains pure-native text
    primitives with no image cells or raster overlays.
    """
    if merged_candidates is None:
        ocr = collect_preprocessed_ocr_candidates(
            image_path,
            modes=modes,
            psm_values=psm_values,
            scale=scale,
            conf_threshold=conf_threshold,
        )
        merged = ocr["merged"]
        raw_candidate_count = len(ocr["candidates"])
    else:
        merged = merged_candidates
        raw_candidate_count = int(candidate_count or len(merged_candidates))

    updated = copy.deepcopy(program)
    primitives = updated.setdefault("primitives", [])
    existing_texts = [
        {
            "text": str(p.get("text", "")),
            "bbox": bbox,
            "norm": _norm_text_key(p.get("text", "")),
        }
        for p in primitives
        if p.get("type") == "text" and p.get("bbox")
        for bbox in [_bbox_tuple(p.get("bbox"))]
        if bbox is not None
    ]
    existing_text_boxes = [item["bbox"] for item in existing_texts]
    operations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for candidate in _rank_text_candidates(merged):
        if len(operations) >= max_additions:
            break
        if float(candidate.get("conf", 0.0)) < min_confidence:
            continue
        if not _is_good_missing_text(
            candidate,
            allow_standalone_numbers=allow_standalone_numbers,
            min_support_count=min_support_count,
        ):
            skipped.append(_skip(candidate, "failed_text_quality_filter"))
            continue
        bbox = _bbox_tuple(candidate.get("bbox"))
        if bbox is None:
            continue
        if _covered_by_text(bbox, existing_text_boxes):
            skipped.append(_skip(candidate, "covered_by_existing_text"))
            continue
        if _similar_to_nearby_text(candidate, existing_texts, similar_text_radius):
            skipped.append(_skip(candidate, "similar_to_nearby_text"))
            continue
        primitive = {
            "id": _next_id(primitives, "ocr_text"),
            "type": "text",
            "role": "ocr_preprocess_text",
            "text": _normalize_text(candidate.get("text", "")),
            "bbox": [round(v, 3) for v in bbox],
            "style": {
                "font_size": int(candidate.get("font_size") or 8),
                "bold": bool(candidate.get("bold")),
                "align": candidate.get("align", "center"),
            },
            "source": "ocr_preprocess_augment",
            "confidence": round(float(candidate.get("conf", 0.0)) / 100.0, 3),
            "ocr_mode": candidate.get("ocr_mode"),
            "ocr_psm": candidate.get("ocr_psm"),
        }
        primitives.append(primitive)
        existing_text_boxes.append(bbox)
        existing_texts.append({
            "text": primitive["text"],
            "bbox": bbox,
            "norm": _norm_text_key(primitive["text"]),
        })
        operations.append({
            "action": "add_text",
            "primitive_id": primitive["id"],
            "text": primitive["text"],
            "bbox": primitive["bbox"],
            "confidence": primitive["confidence"],
            "ocr_mode": primitive["ocr_mode"],
            "ocr_psm": primitive["ocr_psm"],
        })

    if merge_stacked_labels:
        merge_operations = _merge_stacked_text_labels(primitives)
        operations.extend(merge_operations)

    _refresh_counts(updated)
    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": OCR_PREPROCESS_AUGMENT_VERSION,
        "source_image": str(image_path),
        "config": {
            "modes": list(modes),
            "psm_values": list(psm_values),
            "scale": scale,
            "conf_threshold": conf_threshold,
            "min_confidence": min_confidence,
            "max_additions": max_additions,
            "allow_standalone_numbers": allow_standalone_numbers,
            "similar_text_radius": similar_text_radius,
            "merge_stacked_labels": merge_stacked_labels,
            "min_support_count": min_support_count,
        },
        "counts": {
            "candidates": raw_candidate_count,
            "merged": len(merged),
            "operations": len(operations),
            "skipped": len(skipped),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
        "skipped_sample": skipped[:80],
    }
    updated.setdefault("metadata", {})["ocr_preprocess_augment"] = {
        "version": OCR_PREPROCESS_AUGMENT_VERSION,
        "counts": report["counts"],
    }
    return updated, report


def collect_preprocessed_ocr_candidates(
    image_path: str | Path,
    *,
    modes: tuple[str, ...] = ("sharp",),
    psm_values: tuple[int, ...] = (6, 11, 12),
    scale: float = 3.0,
    conf_threshold: float = 25.0,
) -> dict[str, list[dict[str, Any]]]:
    image = Image.open(image_path).convert("RGB")
    candidates: list[dict[str, Any]] = []
    for mode in modes:
        view = _preprocess_image(image, mode)
        for psm in psm_values:
            for item in _detect_ocr_text_single(
                view,
                conf_threshold=conf_threshold,
                scale=scale,
                psm=psm,
            ):
                normalized = dict(item)
                normalized["bbox"] = [float(v) for v in normalized["bbox"]]
                normalized["ocr_mode"] = mode
                normalized["ocr_psm"] = psm
                candidates.append(normalized)
    merged = []
    for item in _merge_ocr_blocks(candidates):
        normalized = dict(item)
        normalized["bbox"] = [float(v) for v in normalized["bbox"]]
        merged.append(normalized)
    _annotate_candidate_support(merged, candidates)
    return {
        "candidates": candidates,
        "merged": merged,
    }


def _preprocess_image(image: Image.Image, mode: str) -> Image.Image:
    if mode == "raw":
        return image
    if mode == "gray_autocontrast":
        return ImageOps.autocontrast(image.convert("L")).convert("RGB")
    if mode == "sharp":
        return image.filter(ImageFilter.SHARPEN)
    if mode == "unsharp":
        return image.filter(ImageFilter.UnsharpMask(
            radius=1.0, percent=160, threshold=3))
    raise ValueError(f"unknown OCR preprocess mode: {mode}")


def _rank_text_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (
        -float(item.get("conf", 0.0)),
        item["bbox"][1],
        item["bbox"][0],
    ))


def _is_good_missing_text(
    item: dict[str, Any],
    *,
    allow_standalone_numbers: bool,
    min_support_count: int,
) -> bool:
    text = _normalize_text(item.get("text", ""))
    if not text:
        return False
    if any(ch in text for ch in "\\‘’`~<>"):
        return False
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return False
    standalone_number = bool(re.fullmatch(r"\d+(?:-\d+)?", cleaned))
    if len(cleaned) <= 2 and not standalone_number:
        return False
    if standalone_number and not allow_standalone_numbers:
        return False
    if int(item.get("support_count") or 1) < min_support_count:
        return False
    alnum = sum(ch.isalnum() for ch in cleaned)
    if alnum / max(1, len(cleaned)) < 0.45:
        return False
    bbox = _bbox_tuple(item.get("bbox"))
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    if width < 4 or height < 3 or width > 330 or height > 42:
        return False
    if len(cleaned) <= 4 and height > 24:
        return False
    if re.fullmatch(r"[A-Z][a-z]?", cleaned) and float(item.get("conf", 0.0)) < 88:
        return False
    return True


def _annotate_candidate_support(
    merged: list[dict[str, Any]],
    raw_candidates: list[dict[str, Any]],
) -> None:
    for item in merged:
        bbox = _bbox_tuple(item.get("bbox"))
        text_key = _norm_text_key(item.get("text", ""))
        support = []
        if bbox is None or not text_key:
            item["support_count"] = 1
            continue
        for raw in raw_candidates:
            raw_bbox = _bbox_tuple(raw.get("bbox"))
            raw_key = _norm_text_key(raw.get("text", ""))
            if raw_bbox is None or not raw_key:
                continue
            if not _similar_text_key(text_key, raw_key):
                continue
            if (
                _iou(bbox, raw_bbox) > 0.30 or
                _overlap_fraction(bbox, raw_bbox) > 0.58 or
                _overlap_fraction(raw_bbox, bbox) > 0.58
            ):
                support.append({
                    "mode": raw.get("ocr_mode"),
                    "psm": raw.get("ocr_psm"),
                    "conf": raw.get("conf"),
                })
        item["support_count"] = max(1, len(support))
        item["support"] = support[:8]


def _similar_text_key(left: str, right: str) -> bool:
    if left == right:
        return True
    short, long = sorted((left, right), key=len)
    if len(short) >= 5 and short in long:
        return True
    return _jaccard_ngrams(left, right) >= 0.80


def _normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip("|_")


def _norm_text_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _similar_to_nearby_text(
    candidate: dict[str, Any],
    existing_texts: list[dict[str, Any]],
    radius: float,
) -> bool:
    text_key = _norm_text_key(candidate.get("text", ""))
    bbox = _bbox_tuple(candidate.get("bbox"))
    if not text_key or bbox is None:
        return False
    cx, cy = _center(bbox)
    for existing in existing_texts:
        other_key = existing.get("norm", "")
        if not other_key:
            continue
        ox, oy = _center(existing["bbox"])
        if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 > radius:
            continue
        if text_key == other_key:
            return True
        short, long = sorted((text_key, other_key), key=len)
        if len(short) >= 4 and short in long:
            return True
        if _jaccard_ngrams(text_key, other_key) >= 0.78:
            return True
    return False


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _jaccard_ngrams(left: str, right: str, n: int = 3) -> float:
    if len(left) < n or len(right) < n:
        return 0.0
    a = {left[i:i + n] for i in range(len(left) - n + 1)}
    b = {right[i:i + n] for i in range(len(right) - n + 1)}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _skip(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "reason": reason,
        "text": _normalize_text(item.get("text", "")),
        "bbox": item.get("bbox"),
        "conf": item.get("conf"),
    }


def _covered_by_text(
    bbox: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
) -> bool:
    for other in boxes:
        if _iou(bbox, other) > 0.18:
            return True
        if _overlap_fraction(bbox, other) > 0.35:
            return True
        if _overlap_fraction(other, bbox) > 0.55:
            return True
    return False


def _merge_stacked_text_labels(primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        pair = _find_stacked_text_pair(primitives)
        if pair is None:
            break
        upper_index, lower_index = pair
        upper = primitives[upper_index]
        lower = primitives[lower_index]
        upper_text = _normalize_text(upper.get("text", ""))
        lower_text = _normalize_text(lower.get("text", ""))
        if not upper_text or not lower_text:
            break
        upper_bbox = _bbox_tuple(upper.get("bbox"))
        lower_bbox = _bbox_tuple(lower.get("bbox"))
        if upper_bbox is None or lower_bbox is None:
            break
        merged_bbox = [
            round(min(upper_bbox[0], lower_bbox[0]), 3),
            round(min(upper_bbox[1], lower_bbox[1]), 3),
            round(max(upper_bbox[2], lower_bbox[2]), 3),
            round(max(upper_bbox[3], lower_bbox[3]), 3),
        ]
        upper["text"] = f"{upper_text}\n{lower_text}"
        upper["bbox"] = merged_bbox
        upper.setdefault("style", {})["font_size"] = min(
            int(upper.get("style", {}).get("font_size") or 8),
            int(lower.get("style", {}).get("font_size") or 8),
        )
        upper.setdefault("style", {})["align"] = "center"
        upper["source"] = "ocr_stack_merge"
        operations.append({
            "action": "merge_stacked_text",
            "primitive_id": upper.get("id"),
            "removed_id": lower.get("id"),
            "text": upper["text"],
            "bbox": merged_bbox,
        })
        del primitives[lower_index]
        changed = True
    return operations


def _find_stacked_text_pair(
    primitives: list[dict[str, Any]],
) -> tuple[int, int] | None:
    texts = [
        (index, primitive, _bbox_tuple(primitive.get("bbox")))
        for index, primitive in enumerate(primitives)
        if primitive.get("type") == "text"
    ]
    texts = [(i, p, b) for i, p, b in texts if b is not None]
    for upper_index, upper, upper_bbox in texts:
        if "\n" in str(upper.get("text", "")):
            continue
        for lower_index, lower, lower_bbox in texts:
            if lower_index == upper_index:
                continue
            if "\n" in str(lower.get("text", "")):
                continue
            if lower_bbox[1] < upper_bbox[1]:
                continue
            if not _stack_merge_allowed(upper, lower):
                continue
            if _stack_geometry_ok(upper_bbox, lower_bbox):
                return (upper_index, lower_index)
    return None


def _stack_merge_allowed(upper: dict[str, Any], lower: dict[str, Any]) -> bool:
    sources = {upper.get("source", ""), lower.get("source", "")}
    if "ocr_preprocess_augment" not in sources:
        return False
    for primitive in (upper, lower):
        text = _normalize_text(primitive.get("text", ""))
        if not text or len(text) > 18 or text.startswith("Figure "):
            return False
        bbox = _bbox_tuple(primitive.get("bbox"))
        if bbox is None:
            return False
        if bbox[3] - bbox[1] > 18:
            return False
    return True


def _stack_geometry_ok(
    upper_bbox: tuple[float, float, float, float],
    lower_bbox: tuple[float, float, float, float],
) -> bool:
    ux0, uy0, ux1, uy1 = upper_bbox
    lx0, ly0, lx1, ly1 = lower_bbox
    gap = ly0 - uy1
    if gap < -1.5 or gap > 9.0:
        return False
    uw = ux1 - ux0
    lw = lx1 - lx0
    if max(uw, lw) > 90:
        return False
    ucx, _ = _center(upper_bbox)
    lcx, _ = _center(lower_bbox)
    if abs(ucx - lcx) > max(8.0, 0.36 * max(uw, lw)):
        return False
    ix0 = max(ux0, lx0)
    ix1 = min(ux1, lx1)
    overlap = max(0.0, ix1 - ix0)
    if overlap / max(1.0, min(uw, lw)) < 0.25:
        return False
    return True


def _bbox_tuple(raw: Any) -> tuple[float, float, float, float] | None:
    if not raw or len(raw) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in raw]
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = _area(a) + _area(b) - inter
    return inter / max(1.0, area)


def _overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max(1.0, _area(a))


def _area(bbox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _next_id(primitives: list[dict[str, Any]], prefix: str) -> str:
    existing = {p.get("id") for p in primitives}
    index = 1
    while f"{prefix}_{index:04d}" in existing:
        index += 1
    return f"{prefix}_{index:04d}"


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
