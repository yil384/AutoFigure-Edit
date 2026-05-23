"""Computer-vision evidence extraction for visual-primitive reconstruction.

The VLM should not estimate geometry from memory. This module gives it a
measurable evidence layer: candidate panel boxes, line segments, junctions,
arrowheads, connected components, and snap guides in source-image pixels.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CV_EVIDENCE_VERSION = "cv-evidence-0.1"


def extract_cv_primitives(
    image_path: str | Path,
    *,
    max_lines: int = 1200,
    max_regions: int = 240,
    max_components: int = 420,
    min_line_length: float = 12.0,
    axis_angle_tolerance: float = 8.0,
    merge_axis_lines: bool = True,
) -> dict[str, Any]:
    """Extract geometry evidence from a raster figure.

    The returned JSON is intentionally detector-facing, not final draw.io IR:
    each object is a candidate with source, confidence, and pixel geometry.
    """
    image_path = Path(image_path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 5, 24, 24)

    text_regions = _detect_text_regions(bgr)
    text_lines = _group_text_lines(text_regions)
    regions = _detect_regions(bgr, denoised, max_regions=max_regions)
    raw_lines = _detect_line_segments(
        denoised,
        min_line_length=min_line_length,
        axis_angle_tolerance=axis_angle_tolerance,
    )
    if merge_axis_lines:
        raw_lines = _merge_axis_aligned_lines(raw_lines)
    line_segments = _rank_lines(raw_lines, max_lines=max_lines,
                                regions=regions, text_regions=text_regions)
    junctions = _cluster_junctions(line_segments)
    arrowheads = _detect_arrowheads(denoised, line_segments, text_regions=text_regions)
    components = _detect_components(bgr, denoised, max_components=max_components)
    snap_guides = _make_snap_guides(regions, line_segments, junctions)

    evidence = {
        "version": CV_EVIDENCE_VERSION,
        "source": str(image_path),
        "canvas": {"width": width, "height": height},
        "coordinate_space": {
            "type": "image_pixels",
            "origin": "top_left",
            "normalized": False,
        },
        "counts": {
            "regions": len(regions),
            "line_segments": len(line_segments),
            "junctions": len(junctions),
            "arrowheads": len(arrowheads),
            "components": len(components),
            "text_regions": len(text_regions),
            "text_lines": len(text_lines),
            "snap_x": len(snap_guides["x"]),
            "snap_y": len(snap_guides["y"]),
        },
        "regions": regions,
        "line_segments": line_segments,
        "junctions": junctions,
        "arrowheads": arrowheads,
        "components": components,
        "text_regions": text_regions,
        "text_lines": text_lines,
        "snap_guides": snap_guides,
        "extraction_config": {
            "max_lines": max_lines,
            "max_regions": max_regions,
            "max_components": max_components,
            "min_line_length": min_line_length,
            "axis_angle_tolerance": axis_angle_tolerance,
            "merge_axis_lines": merge_axis_lines,
        },
    }
    return evidence


def save_cv_evidence(evidence: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=True))


def draw_cv_overlay(
    image_path: str | Path,
    evidence: dict[str, Any],
    output_path: str | Path,
    *,
    label_limit: int = 160,
) -> None:
    """Render a transparent diagnostic overlay over the source image."""
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font()

    for region in evidence.get("regions", []):
        x0, y0, x1, y1 = region["bbox"]
        color = (52, 120, 246, 72) if region.get("source") == "fill_component" else (52, 120, 246, 38)
        draw.rectangle([x0, y0, x1, y1], outline=(30, 96, 210, 210), fill=color, width=2)

    for component in evidence.get("components", [])[:max(0, label_limit // 2)]:
        x0, y0, x1, y1 = component["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(160, 96, 210, 125), width=1)

    for text_region in evidence.get("text_regions", [])[:240]:
        x0, y0, x1, y1 = text_region["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(210, 40, 150, 62), width=1)

    for text_line in evidence.get("text_lines", [])[:220]:
        x0, y0, x1, y1 = text_line["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(70, 30, 190, 150), width=2)

    for line in evidence.get("line_segments", []):
        x0, y0 = line["p0"]
        x1, y1 = line["p1"]
        orient = line.get("orientation")
        role = line.get("role_hint")
        if role == "text_stroke":
            color = (105, 105, 105, 95)
        elif role == "panel_border":
            color = (30, 96, 210, 235)
        elif orient == "horizontal":
            color = (34, 160, 70, 235)
        elif orient == "vertical":
            color = (245, 140, 32, 235)
        else:
            color = (28, 170, 190, 220)
        draw.line([x0, y0, x1, y1], fill=color, width=2)

    for arrow in evidence.get("arrowheads", []):
        points = [tuple(p) for p in arrow.get("points", [])]
        if len(points) >= 3:
            draw.polygon(points, outline=(220, 48, 54, 235), fill=(220, 48, 54, 95))

    for node in evidence.get("junctions", []):
        x, y = node["point"]
        r = 3 if node.get("degree", 1) <= 2 else 5
        draw.ellipse([x - r, y - r, x + r, y + r],
                     outline=(255, 214, 42, 240),
                     fill=(255, 214, 42, 175))

    label_count = 0
    for collection_name in ("regions", "line_segments", "junctions"):
        for item in evidence.get(collection_name, []):
            if label_count >= label_limit:
                break
            label = item.get("id")
            if not label:
                continue
            if "bbox" in item:
                x, y = item["bbox"][:2]
            else:
                x, y = item.get("point", [0, 0])
            draw.text((float(x) + 2, float(y) + 2), label,
                      fill=(10, 38, 72, 220), font=font)
            label_count += 1

    out = Image.alpha_composite(image, overlay).convert("RGB")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def _detect_regions(bgr: np.ndarray, gray: np.ndarray, *, max_regions: int) -> list[dict[str, Any]]:
    h, w = gray.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    non_white = np.max(255 - bgr, axis=2)

    fill_mask = ((sat < 70) & (val > 162) & (non_white > 7)).astype(np.uint8) * 255
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    edges = cv2.Canny(gray, 55, 155)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    candidates: list[dict[str, Any]] = []
    candidates.extend(_region_candidates_from_mask(fill_mask, "fill_component", h, w))
    candidates.extend(_region_candidates_from_contours(closed_edges, "edge_contour", h, w))
    candidates = _dedupe_boxes(candidates, iou_threshold=0.72)
    candidates.sort(key=lambda item: (-_area(item["bbox"]), item["bbox"][1], item["bbox"][0]))

    out = []
    for idx, item in enumerate(candidates[:max_regions], start=1):
        x0, y0, x1, y1 = item["bbox"]
        area = _area(item["bbox"])
        out.append({
            "id": f"cv_region_{idx:04d}",
            "bbox": _round_bbox([x0, y0, x1, y1]),
            "width": round(float(x1 - x0), 3),
            "height": round(float(y1 - y0), 3),
            "area": round(area, 3),
            "source": item["source"],
            "confidence": round(float(item.get("confidence", 0.5)), 3),
        })
    return out


def _detect_text_regions(bgr: np.ndarray) -> list[dict[str, Any]]:
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return []

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    try:
        data = pytesseract.image_to_data(
            pil,
            config="--psm 11",
            output_type=Output.DICT,
        )
    except Exception:
        return []

    items = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < 18:
            continue
        x = float(data["left"][i])
        y = float(data["top"][i])
        w = float(data["width"][i])
        h = float(data["height"][i])
        if w < 2 or h < 4:
            continue
        items.append({
            "bbox": [x - 2, y - 2, x + w + 2, y + h + 2],
            "text": text,
            "confidence": conf / 100.0,
            "source": "tesseract_word_box",
            "block_num": int(data.get("block_num", [0] * n)[i]),
            "par_num": int(data.get("par_num", [0] * n)[i]),
            "line_num": int(data.get("line_num", [0] * n)[i]),
            "word_num": int(data.get("word_num", [0] * n)[i]),
        })

    items = _dedupe_boxes(items, iou_threshold=0.65)
    items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    out = []
    for idx, item in enumerate(items[:900], start=1):
        out.append({
            "id": f"cv_text_{idx:04d}",
            "bbox": _round_bbox(item["bbox"]),
            "text": item["text"],
            "source": item["source"],
            "confidence": round(float(item["confidence"]), 3),
            "block_num": item.get("block_num"),
            "par_num": item.get("par_num"),
            "line_num": item.get("line_num"),
            "word_num": item.get("word_num"),
        })
    return out


def _group_text_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group OCR word boxes into line-level text evidence."""
    usable = [
        word for word in words
        if word.get("bbox") and _usable_text_for_line(word.get("text", ""))
    ]
    keyed = [word for word in usable if word.get("block_num") and word.get("line_num")]
    if len(keyed) >= max(6, int(0.4 * len(usable))):
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for word in keyed:
            key = (
                int(word.get("block_num") or 0),
                int(word.get("par_num") or 0),
                int(word.get("line_num") or 0),
            )
            grouped.setdefault(key, []).append(word)
        line_items = []
        for group_words in grouped.values():
            line_items.extend(_split_text_line_words(group_words))
        return _finalize_text_lines(line_items)

    usable.sort(key=lambda w: (_bbox_center_y(w["bbox"]), w["bbox"][0]))
    rows: list[dict[str, Any]] = []
    for word in usable:
        bbox = word["bbox"]
        cy = _bbox_center_y(bbox)
        h = max(1.0, bbox[3] - bbox[1])
        best = None
        best_delta = 1e9
        for row in rows:
            tolerance = max(5.0, 0.62 * max(h, row["median_h"]))
            delta = abs(cy - row["center_y"])
            if delta <= tolerance and delta < best_delta:
                best = row
                best_delta = delta
        if best is None:
            rows.append({
                "words": [word],
                "center_y": cy,
                "median_h": h,
            })
            continue
        best["words"].append(word)
        centers = [_bbox_center_y(item["bbox"]) for item in best["words"]]
        heights = [max(1.0, item["bbox"][3] - item["bbox"][1]) for item in best["words"]]
        best["center_y"] = float(np.median(centers))
        best["median_h"] = float(np.median(heights))

    line_items: list[dict[str, Any]] = []
    for row in rows:
        line_items.extend(_split_text_line_words(row["words"]))

    return _finalize_text_lines(line_items)


def _split_text_line_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_words = sorted(words, key=lambda w: (int(w.get("word_num") or 0), w["bbox"][0]))
    heights = [max(1.0, w["bbox"][3] - w["bbox"][1]) for w in row_words]
    median_h = float(np.median(heights)) if heights else 8.0
    line_items: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    last_x1: float | None = None
    for word in row_words:
        x0, _, x1, _ = word["bbox"]
        gap = 0.0 if last_x1 is None else x0 - last_x1
        split_gap = max(28.0, 3.4 * median_h)
        if current and gap > split_gap:
            line_items.append(_make_text_line(current))
            current = []
        current.append(word)
        last_x1 = x1
    if current:
        line_items.append(_make_text_line(current))
    return line_items


def _finalize_text_lines(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    line_items = [
        item for item in line_items
        if _usable_text_for_line(item["text"]) and item["bbox"][2] - item["bbox"][0] >= 3
    ]
    line_items = _dedupe_boxes(line_items, iou_threshold=0.78)
    line_items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    out = []
    for idx, item in enumerate(line_items[:420], start=1):
        out.append({
            "id": f"cv_textline_{idx:04d}",
            "bbox": _round_bbox(item["bbox"]),
            "text": item["text"],
            "word_count": item["word_count"],
            "word_ids": item["word_ids"][:32],
            "source": "tesseract_line_group",
            "confidence": round(float(item["confidence"]), 3),
        })
    return out


def _make_text_line(words: list[dict[str, Any]]) -> dict[str, Any]:
    x0 = min(float(w["bbox"][0]) for w in words)
    y0 = min(float(w["bbox"][1]) for w in words)
    x1 = max(float(w["bbox"][2]) for w in words)
    y1 = max(float(w["bbox"][3]) for w in words)
    text = _join_ocr_words([str(w.get("text", "")) for w in words])
    weights = [
        max(1.0, _area(w["bbox"]))
        for w in words
    ]
    conf = sum(float(w.get("confidence", 0.0)) * weight for w, weight in zip(words, weights)) / max(1.0, sum(weights))
    if len(words) >= 2:
        conf = min(0.99, conf + 0.05)
    return {
        "bbox": [x0 - 1.0, y0 - 1.0, x1 + 1.0, y1 + 1.0],
        "text": text,
        "word_count": len(words),
        "word_ids": [str(w.get("id")) for w in words if w.get("id")],
        "confidence": conf,
    }


def _join_ocr_words(words: list[str]) -> str:
    out = ""
    for raw in words:
        word = raw.strip()
        if not word:
            continue
        if not out:
            out = word
        elif word in {".", ",", ":", ";", ")", "]", "}"}:
            out += word
        elif out[-1:] in {"(", "[", "{", "/", "-"}:
            out += word
        elif word.startswith(("/", "-")):
            out += word
        else:
            out += " " + word
    return out


def _usable_text_for_line(text: str) -> bool:
    stripped = str(text).strip()
    if not stripped:
        return False
    alnum = sum(ch.isalnum() for ch in stripped)
    if alnum >= 2:
        return True
    return stripped in {"X", "Z", "I", "V", "+", "-", "&", "/", "...", "2X", "2Y", "ZX"}


def _region_candidates_from_mask(mask: np.ndarray, source: str, h: int, w: int) -> list[dict[str, Any]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if not _valid_region_box(x, y, bw, bh, area, h, w):
            continue
        fill_fraction = float(area) / max(1.0, bw * bh)
        out.append({
            "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
            "source": source,
            "confidence": min(0.98, 0.45 + fill_fraction),
        })
    return out


def _region_candidates_from_contours(mask: np.ndarray, source: str, h: int, w: int) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not _valid_region_box(x, y, bw, bh, area, h, w):
            continue
        rect_area = max(1.0, bw * bh)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        rectangularity = area / rect_area
        confidence = 0.45 + 0.35 * min(1.0, rectangularity)
        if 4 <= len(approx) <= 10:
            confidence += 0.12
        out.append({
            "bbox": [float(x), float(y), float(x + bw), float(y + bh)],
            "source": source,
            "confidence": min(0.98, confidence),
        })
    return out


def _valid_region_box(x: int, y: int, w: int, h: int, area: float, image_h: int, image_w: int) -> bool:
    if w < 18 or h < 14:
        return False
    if area < 260:
        return False
    if w > image_w * 0.98 and h > image_h * 0.98:
        return False
    aspect = w / max(1.0, h)
    if aspect > 22 or aspect < 0.045:
        return False
    return True


def _detect_line_segments(
    gray: np.ndarray,
    *,
    min_line_length: float,
    axis_angle_tolerance: float,
) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    if hasattr(cv2, "createLineSegmentDetector"):
        detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        detected = detector.detect(gray)
        lsd_lines = detected[0] if detected and detected[0] is not None else []
        widths = detected[1] if len(detected) > 1 and detected[1] is not None else None
        for i, raw in enumerate(lsd_lines):
            x0, y0, x1, y1 = [float(v) for v in raw[0]]
            length = math.hypot(x1 - x0, y1 - y0)
            if length < min_line_length:
                continue
            line_width = float(widths[i][0]) if widths is not None and i < len(widths) else 1.0
            lines.append(_line_record(
                [x0, y0], [x1, y1],
                source="lsd",
                confidence=min(0.99, 0.45 + length / 260.0),
                detector_width=line_width,
                axis_angle_tolerance=axis_angle_tolerance,
            ))

    edges = cv2.Canny(gray, 45, 140)
    hough = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=28,
                            minLineLength=int(max(10, min_line_length)),
                            maxLineGap=6)
    if hough is not None:
        for raw in hough[:, 0, :]:
            x0, y0, x1, y1 = [float(v) for v in raw]
            length = math.hypot(x1 - x0, y1 - y0)
            if length < min_line_length:
                continue
            lines.append(_line_record(
                [x0, y0], [x1, y1],
                source="hough",
                confidence=min(0.92, 0.42 + length / 300.0),
                detector_width=1.0,
                axis_angle_tolerance=axis_angle_tolerance,
            ))

    return _dedupe_lines(lines)


def _line_record(p0, p1, *, source: str, confidence: float, detector_width: float,
                 axis_angle_tolerance: float) -> dict[str, Any]:
    x0, y0 = p0
    x1, y1 = p1
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    normalized_angle = abs(((angle + 90) % 180) - 90)
    if normalized_angle <= axis_angle_tolerance:
        orientation = "horizontal"
        avg_y = (y0 + y1) / 2
        p0 = [min(x0, x1), avg_y]
        p1 = [max(x0, x1), avg_y]
    elif abs(normalized_angle - 90) <= axis_angle_tolerance:
        orientation = "vertical"
        avg_x = (x0 + x1) / 2
        p0 = [avg_x, min(y0, y1)]
        p1 = [avg_x, max(y0, y1)]
    else:
        orientation = "diagonal"
        p0 = [x0, y0]
        p1 = [x1, y1]

    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return {
        "p0": [float(p0[0]), float(p0[1])],
        "p1": [float(p1[0]), float(p1[1])],
        "bbox": _round_bbox([
            min(p0[0], p1[0]), min(p0[1], p1[1]),
            max(p0[0], p1[0]), max(p0[1], p1[1]),
        ]),
        "length": float(length),
        "angle_deg": float(angle),
        "orientation": orientation,
        "source": source,
        "detector_width": detector_width,
        "confidence": float(confidence),
    }


def _dedupe_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = sorted(lines, key=lambda line: (-line["length"], line["source"]))
    kept: list[dict[str, Any]] = []
    for line in lines:
        if any(_same_line(line, other) for other in kept):
            continue
        kept.append(line)
    return kept


def _same_line(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["orientation"] != b["orientation"]:
        return False
    if a["orientation"] == "horizontal":
        if abs(a["p0"][1] - b["p0"][1]) > 3:
            return False
        return _interval_overlap_fraction(
            [a["p0"][0], a["p1"][0]], [b["p0"][0], b["p1"][0]]) > 0.86
    if a["orientation"] == "vertical":
        if abs(a["p0"][0] - b["p0"][0]) > 3:
            return False
        return _interval_overlap_fraction(
            [a["p0"][1], a["p1"][1]], [b["p0"][1], b["p1"][1]]) > 0.86
    return _endpoint_distance(a, b) < 4.5 or _endpoint_distance_reversed(a, b) < 4.5


def _merge_axis_aligned_lines(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axis = [line for line in lines if line["orientation"] in {"horizontal", "vertical"}]
    diag = [line for line in lines if line["orientation"] == "diagonal"]
    merged: list[dict[str, Any]] = []

    for orient in ("horizontal", "vertical"):
        group = [line for line in axis if line["orientation"] == orient]
        group.sort(key=lambda line: (round(line["p0"][1] if orient == "horizontal" else line["p0"][0]),
                                     line["p0"][0] if orient == "horizontal" else line["p0"][1]))
        consumed = [False] * len(group)
        for i, line in enumerate(group):
            if consumed[i]:
                continue
            consumed[i] = True
            cluster = [line]
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(group):
                    if consumed[j]:
                        continue
                    if _mergeable_axis_cluster(cluster, other, orient):
                        consumed[j] = True
                        cluster.append(other)
                        changed = True
            merged.append(_merge_axis_cluster(cluster, orient))

    return merged + diag


def _mergeable_axis_cluster(cluster: list[dict[str, Any]], line: dict[str, Any], orient: str) -> bool:
    c = _merge_axis_cluster(cluster, orient)
    if orient == "horizontal":
        if abs(c["p0"][1] - line["p0"][1]) > 4.5:
            return False
        return _interval_gap([c["p0"][0], c["p1"][0]], [line["p0"][0], line["p1"][0]]) <= 9
    if abs(c["p0"][0] - line["p0"][0]) > 4.5:
        return False
    return _interval_gap([c["p0"][1], c["p1"][1]], [line["p0"][1], line["p1"][1]]) <= 9


def _merge_axis_cluster(cluster: list[dict[str, Any]], orient: str) -> dict[str, Any]:
    if orient == "horizontal":
        y = float(np.median([line["p0"][1] for line in cluster]))
        x0 = min(min(line["p0"][0], line["p1"][0]) for line in cluster)
        x1 = max(max(line["p0"][0], line["p1"][0]) for line in cluster)
        p0, p1 = [x0, y], [x1, y]
    else:
        x = float(np.median([line["p0"][0] for line in cluster]))
        y0 = min(min(line["p0"][1], line["p1"][1]) for line in cluster)
        y1 = max(max(line["p0"][1], line["p1"][1]) for line in cluster)
        p0, p1 = [x, y0], [x, y1]
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return {
        "p0": p0,
        "p1": p1,
        "bbox": _round_bbox([min(p0[0], p1[0]), min(p0[1], p1[1]),
                             max(p0[0], p1[0]), max(p0[1], p1[1])]),
        "length": length,
        "angle_deg": 0.0 if orient == "horizontal" else 90.0,
        "orientation": orient,
        "source": "merged_axis_line",
        "detector_width": max(line.get("detector_width", 1.0) for line in cluster),
        "confidence": min(0.99, max(line["confidence"] for line in cluster) + 0.04),
        "merged_count": len(cluster),
    }


def _rank_lines(lines: list[dict[str, Any]], *, max_lines: int,
                regions: list[dict[str, Any]],
                text_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = sorted(lines, key=lambda line: (-line["length"], line["bbox"][1], line["bbox"][0]))
    out = []
    for idx, line in enumerate(lines[:max_lines], start=1):
        item = dict(line)
        item["id"] = f"cv_line_{idx:04d}"
        item["p0"] = _round_point(item["p0"])
        item["p1"] = _round_point(item["p1"])
        item["length"] = round(float(item["length"]), 3)
        item["angle_deg"] = round(float(item["angle_deg"]), 3)
        item["confidence"] = round(float(item["confidence"]), 3)
        item["detector_width"] = round(float(item.get("detector_width", 1.0)), 3)
        item["role_hint"] = _line_role_hint(item, regions, text_regions)
        out.append(item)
    return out


def _cluster_junctions(lines: list[dict[str, Any]], threshold: float = 7.0) -> list[dict[str, Any]]:
    points: list[tuple[float, float, str]] = []
    for line in lines:
        points.append((float(line["p0"][0]), float(line["p0"][1]), line["id"]))
        points.append((float(line["p1"][0]), float(line["p1"][1]), line["id"]))

    clusters: list[dict[str, Any]] = []
    for x, y, line_id in points:
        assigned = False
        for cluster in clusters:
            cx, cy = cluster["sum_x"] / cluster["count"], cluster["sum_y"] / cluster["count"]
            if math.hypot(x - cx, y - cy) <= threshold:
                cluster["sum_x"] += x
                cluster["sum_y"] += y
                cluster["count"] += 1
                cluster["line_ids"].add(line_id)
                assigned = True
                break
        if not assigned:
            clusters.append({"sum_x": x, "sum_y": y, "count": 1, "line_ids": {line_id}})

    out = []
    for idx, cluster in enumerate(
        sorted(clusters, key=lambda c: (-len(c["line_ids"]), c["sum_y"] / c["count"], c["sum_x"] / c["count"])),
        start=1,
    ):
        x = cluster["sum_x"] / cluster["count"]
        y = cluster["sum_y"] / cluster["count"]
        out.append({
            "id": f"cv_node_{idx:04d}",
            "point": _round_point([x, y]),
            "degree": len(cluster["line_ids"]),
            "line_ids": sorted(cluster["line_ids"])[:16],
            "confidence": round(min(0.98, 0.36 + 0.14 * len(cluster["line_ids"])), 3),
        })
    return out


def _detect_arrowheads(gray: np.ndarray, lines: list[dict[str, Any]],
                       *, text_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _, dark = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    endpoints = []
    for line in lines:
        endpoints.append((line["id"], line["p0"]))
        endpoints.append((line["id"], line["p1"]))

    arrows = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 9 or area > 900:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.08 * perimeter, True)
        if not 3 <= len(approx) <= 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) < 5 or max(w, h) > 42:
            continue
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        if _max_overlap_fraction(bbox, [t["bbox"] for t in text_regions]) > 0.42:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area / hull_area < 0.55:
            continue
        m = cv2.moments(contour)
        if not m["m00"]:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        nearest_line = None
        nearest_dist = 1e9
        for line_id, pt in endpoints:
            dist = math.hypot(cx - pt[0], cy - pt[1])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_line = line_id
        if nearest_dist > 16:
            continue
        nearest_line_record = next((line for line in lines if line["id"] == nearest_line), None)
        if nearest_line_record and nearest_line_record.get("role_hint") == "text_stroke":
            continue
        points = [[float(p[0][0]), float(p[0][1])] for p in approx]
        arrows.append({
            "bbox": _round_bbox(bbox),
            "centroid": _round_point([cx, cy]),
            "points": [_round_point(p) for p in points],
            "nearest_line_id": nearest_line,
            "nearest_endpoint_distance": round(float(nearest_dist), 3),
            "source": "dark_triangular_contour",
            "confidence": round(min(0.94, 0.42 + area / 220.0), 3),
        })

    arrows = _dedupe_boxes(arrows, iou_threshold=0.55)
    for idx, arrow in enumerate(arrows, start=1):
        arrow["id"] = f"cv_arrow_{idx:04d}"
    return arrows


def _detect_components(bgr: np.ndarray, gray: np.ndarray, *, max_components: int) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    non_white = np.max(255 - bgr, axis=2)
    dark_or_colored = ((gray < 214) | ((sat > 38) & (non_white > 15))).astype(np.uint8) * 255
    dark_or_colored = cv2.morphologyEx(dark_or_colored, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_or_colored, connectivity=8)
    items = []
    image_area = gray.shape[0] * gray.shape[1]
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 12 or area > image_area * 0.18:
            continue
        if w < 2 or h < 2:
            continue
        aspect = w / max(1.0, h)
        if aspect > 80 or aspect < 0.012:
            continue
        cx, cy = centroids[i]
        items.append({
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "centroid": _round_point([cx, cy]),
            "area": int(area),
            "aspect": float(aspect),
            "source": "foreground_component",
            "confidence": min(0.9, 0.25 + math.sqrt(area) / 90.0),
        })
    items = _dedupe_boxes(items, iou_threshold=0.86)
    items.sort(key=lambda item: (-item["area"], item["bbox"][1], item["bbox"][0]))
    out = []
    for idx, item in enumerate(items[:max_components], start=1):
        out.append({
            "id": f"cv_comp_{idx:04d}",
            "bbox": _round_bbox(item["bbox"]),
            "centroid": item["centroid"],
            "area": item["area"],
            "aspect": round(item["aspect"], 3),
            "source": item["source"],
            "confidence": round(item["confidence"], 3),
        })
    return out


def _line_role_hint(line: dict[str, Any],
                    regions: list[dict[str, Any]],
                    text_regions: list[dict[str, Any]]) -> str:
    expanded = _expanded_line_bbox(line, pad=3.5)
    text_overlap = _max_overlap_fraction(expanded, [t["bbox"] for t in text_regions])
    if text_overlap > 0.32 and line.get("length", 0) < 90:
        return "text_stroke"

    if line.get("orientation") in {"horizontal", "vertical"}:
        for region in regions:
            rb = region["bbox"]
            if _line_near_region_border(line, rb, tolerance=5.5):
                return "panel_border"
    return "connector_candidate"


def _expanded_line_bbox(line: dict[str, Any], pad: float = 3.0) -> list[float]:
    x0, y0 = line["p0"]
    x1, y1 = line["p1"]
    return [
        min(x0, x1) - pad,
        min(y0, y1) - pad,
        max(x0, x1) + pad,
        max(y0, y1) + pad,
    ]


def _line_near_region_border(line: dict[str, Any], rb: list[float], tolerance: float) -> bool:
    x0, y0, x1, y1 = [float(v) for v in rb]
    if line["orientation"] == "horizontal":
        y = line["p0"][1]
        lx0, lx1 = sorted([line["p0"][0], line["p1"][0]])
        if min(abs(y - y0), abs(y - y1)) > tolerance:
            return False
        return _interval_overlap_fraction([lx0, lx1], [x0, x1]) > 0.35
    if line["orientation"] == "vertical":
        x = line["p0"][0]
        ly0, ly1 = sorted([line["p0"][1], line["p1"][1]])
        if min(abs(x - x0), abs(x - x1)) > tolerance:
            return False
        return _interval_overlap_fraction([ly0, ly1], [y0, y1]) > 0.35
    return False


def _max_overlap_fraction(bbox: list[float], others: list[list[float]]) -> float:
    if not others:
        return 0.0
    return max((_overlap_fraction(bbox, other) for other in others), default=0.0)


def _overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / max(1.0, _area(a))


def _make_snap_guides(
    regions: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    xs: list[tuple[float, float, str]] = []
    ys: list[tuple[float, float, str]] = []
    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        conf = float(region.get("confidence", 0.5))
        xs.extend([(x0, conf, region["id"]), (x1, conf, region["id"]), ((x0 + x1) / 2, conf * 0.7, region["id"])])
        ys.extend([(y0, conf, region["id"]), (y1, conf, region["id"]), ((y0 + y1) / 2, conf * 0.7, region["id"])])
    for line in lines:
        conf = float(line.get("confidence", 0.5))
        if line["orientation"] == "vertical":
            xs.append((line["p0"][0], conf, line["id"]))
        elif line["orientation"] == "horizontal":
            ys.append((line["p0"][1], conf, line["id"]))
    for node in junctions:
        conf = float(node.get("confidence", 0.5))
        xs.append((node["point"][0], conf, node["id"]))
        ys.append((node["point"][1], conf, node["id"]))
    return {
        "x": _cluster_guides(xs),
        "y": _cluster_guides(ys),
    }


def _cluster_guides(values: list[tuple[float, float, str]], tolerance: float = 4.0) -> list[dict[str, Any]]:
    values = sorted(values, key=lambda item: item[0])
    clusters: list[dict[str, Any]] = []
    for value, confidence, source_id in values:
        if not clusters or abs(value - clusters[-1]["mean"]) > tolerance:
            clusters.append({
                "values": [value],
                "confidence_sum": confidence,
                "source_ids": {source_id},
                "mean": value,
            })
        else:
            cluster = clusters[-1]
            cluster["values"].append(value)
            cluster["confidence_sum"] += confidence
            cluster["source_ids"].add(source_id)
            cluster["mean"] = sum(cluster["values"]) / len(cluster["values"])
    out = []
    for idx, cluster in enumerate(clusters, start=1):
        support = len(cluster["source_ids"])
        out.append({
            "id": f"guide_{idx:04d}",
            "value": round(float(cluster["mean"]), 3),
            "support": support,
            "confidence": round(min(0.99, cluster["confidence_sum"] / max(1.0, support)), 3),
            "source_ids": sorted(cluster["source_ids"])[:24],
        })
    out.sort(key=lambda item: (-item["support"], -item["confidence"], item["value"]))
    return out[:240]


def _dedupe_boxes(items: list[dict[str, Any]], *, iou_threshold: float) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: (-float(item.get("confidence", 0.5)), -_area(item["bbox"])))
    kept: list[dict[str, Any]] = []
    for item in ranked:
        if any(_iou(item["bbox"], other["bbox"]) >= iou_threshold for other in kept):
            continue
        kept.append(item)
    return kept


def _interval_gap(a, b) -> float:
    a0, a1 = sorted([float(a[0]), float(a[1])])
    b0, b1 = sorted([float(b[0]), float(b[1])])
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _interval_overlap_fraction(a, b) -> float:
    a0, a1 = sorted([float(a[0]), float(a[1])])
    b0, b1 = sorted([float(b[0]), float(b[1])])
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1.0, min(a1 - a0, b1 - b0))


def _endpoint_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(a["p0"][0] - b["p0"][0], a["p0"][1] - b["p0"][1]) + \
        math.hypot(a["p1"][0] - b["p1"][0], a["p1"][1] - b["p1"][1])


def _endpoint_distance_reversed(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(a["p0"][0] - b["p1"][0], a["p0"][1] - b["p1"][1]) + \
        math.hypot(a["p1"][0] - b["p0"][0], a["p1"][1] - b["p0"][1])


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _area(a) + _area(b) - inter
    return inter / max(1.0, union)


def _area(bbox) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_center_y(bbox) -> float:
    return (float(bbox[1]) + float(bbox[3])) / 2.0


def _round_bbox(bbox) -> list[float]:
    return [round(float(v), 3) for v in bbox]


def _round_point(point) -> list[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _font():
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, 9)
            except OSError:
                pass
    return ImageFont.load_default()
