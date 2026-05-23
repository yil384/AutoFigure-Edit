#!/usr/bin/env python3
"""Generate/evaluate native chart rebuild variants for residual regions."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.panel_regions import load_panel_regions  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import compact_score_row, evaluate_drawio_variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rebuild chart-like residual regions with native axes/bars/lines")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("regions_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="chart_rebuild_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--region-ids", required=True,
                    help="comma-separated chart region ids")
    ap.add_argument("--delete-mode", choices=["intersect", "contained", "center"],
                    default="intersect")
    ap.add_argument("--replace-policy", choices=["replace", "add"],
                    default="replace",
                    help="replace deletes existing chart primitives; add only appends detected native primitives")
    ap.add_argument("--include-combo", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    source = cv2.cvtColor(cv2.imread(args.source_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    program = load_program(args.program_json)
    wanted = {item.strip() for item in args.region_ids.split(",") if item.strip()}
    regions = [
        region for region in load_panel_regions(args.regions_json)
        if str(region.get("id")) in wanted
    ]

    region_payloads = []
    variants = []
    for region in regions:
        new_primitives, detect_report = detect_chart_primitives(source, region)
        region_payloads.append({
            "region": region,
            "new_primitives": new_primitives,
            "detect_report": detect_report,
        })
        if not new_primitives:
            continue
        updated, deleted = _apply_region_primitives(
            program,
            region,
            new_primitives,
            mode=args.delete_mode,
            replace_policy=args.replace_policy,
        )
        variants.append(_write_variant(
            updated,
            base,
            _safe(region["id"]),
            args.font_family,
            {
                "operation": "chart_region_rebuild",
                "replace_policy": args.replace_policy,
                "region": region,
                "deleted_ids": deleted,
                "added": len(new_primitives),
                "detect_report": detect_report,
            },
        ))

    if args.include_combo and region_payloads:
        updated = copy.deepcopy(program)
        all_deleted: list[str] = []
        for payload in region_payloads:
            if not payload["new_primitives"]:
                continue
            updated, deleted = _apply_region_primitives(
                updated,
                payload["region"],
                payload["new_primitives"],
                mode=args.delete_mode,
                replace_policy=args.replace_policy,
            )
            all_deleted.extend(deleted)
        variants.append(_write_variant(
            updated,
            base,
            "combo",
            args.font_family,
            {
                "operation": "chart_region_rebuild_combo",
                "replace_policy": args.replace_policy,
                "regions": regions,
                "deleted_ids": all_deleted,
                "added": sum(len(p["new_primitives"]) for p in region_payloads),
                "region_reports": region_payloads,
            },
        ))

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "regions_json": args.regions_json,
        "delete_mode": args.delete_mode,
        "replace_policy": args.replace_policy,
        "region_payloads": region_payloads,
        "variants": variants,
    }, indent=2, ensure_ascii=True))
    if args.generate_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "variant_count": len(variants),
            "variants": [item["drawio"] for item in variants],
        }, indent=2))
        return

    drawios = [Path(item["drawio"]) for item in variants]
    if args.baseline_drawio:
        drawios.insert(0, Path(args.baseline_drawio))
    rows = evaluate_drawio_variants(
        args.source_image,
        drawios,
        drawio_cli=args.drawio_cli,
        export=True,
    )
    ranking_path = Path(f"{base}.ranking.json")
    ranking_path.write_text(json.dumps({
        "source_image": args.source_image,
        "winner": rows[0]["drawio"] if rows else None,
        "variants": rows,
        "manifest": str(manifest_path),
    }, indent=2, ensure_ascii=True))
    print(json.dumps({
        "manifest": str(manifest_path),
        "ranking": str(ranking_path),
        "winner": rows[0]["drawio"] if rows else None,
        "scores": [compact_score_row(row) for row in rows[:24]],
    }, indent=2))


def detect_chart_primitives(source: np.ndarray, region: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in region["bbox"]]
    crop = source[y0:y1, x0:x1]
    if crop.size == 0:
        return [], {"error": "empty_crop"}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=18,
        maxLineGap=4,
    )
    segments = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            x_a, y_a, x_b, y_b = [float(v) for v in raw]
            length = math.hypot(x_b - x_a, y_b - y_a)
            if length >= 12:
                segments.append((x_a, y_a, x_b, y_b, length))
    axis_x, axis_y, top_y, right_x = _infer_axes(crop, segments)
    primitives = []
    primitives.extend(_axis_primitives(region, axis_x, axis_y, top_y, right_x))
    primitives.extend(_bar_primitives(crop, region, axis_x, axis_y, top_y, right_x))
    primitives.extend(_line_primitives(crop, region, segments, axis_x, axis_y, top_y, right_x))
    primitives = _dedupe_primitives(primitives)
    return primitives, {
        "axis": {
            "x": round(axis_x, 3),
            "y": round(axis_y, 3),
            "top_y": round(top_y, 3),
            "right_x": round(right_x, 3),
        },
        "segments": len(segments),
        "added": len(primitives),
        "counts": {
            "edges": sum(1 for p in primitives if p["type"] == "edge"),
            "shapes": sum(1 for p in primitives if p["type"] == "shape"),
        },
    }


def _infer_axes(crop: np.ndarray, segments: list[tuple[float, float, float, float, float]]) -> tuple[float, float, float, float]:
    h, w = crop.shape[:2]
    horizontals = []
    verticals = []
    for x0, y0, x1, y1, length in segments:
        if abs(y1 - y0) <= 3 and length >= max(24, w * 0.25):
            y = (y0 + y1) / 2
            horizontals.append((x0, x1, y, length))
        if abs(x1 - x0) <= 3 and length >= max(24, h * 0.28):
            x = (x0 + x1) / 2
            verticals.append((x, y0, y1, length))
    if horizontals:
        hx0, hx1, axis_y, _ = max(
            horizontals,
            key=lambda item: item[3] + item[2] * 0.7,
        )
    else:
        hx0, hx1, axis_y = w * 0.25, w * 0.92, h * 0.82
    if verticals:
        axis_x, vy0, vy1, _ = min(
            verticals,
            key=lambda item: abs(item[0] - hx0) - item[3] * 0.03,
        )
        top_y = min(vy0, vy1)
    else:
        axis_x = hx0
        top_y = h * 0.16
    right_x = max(hx1, w * 0.88)
    return axis_x, axis_y, top_y, right_x


def _axis_primitives(region: dict[str, Any], axis_x: float, axis_y: float,
                     top_y: float, right_x: float) -> list[dict[str, Any]]:
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    return [
        _edge(
            region,
            "axis_x",
            [[rx0 + axis_x, ry0 + axis_y], [rx0 + right_x, ry0 + axis_y]],
            "#050505",
            1.15,
            arrow_end=True,
        ),
        _edge(
            region,
            "axis_y",
            [[rx0 + axis_x, ry0 + axis_y], [rx0 + axis_x, ry0 + top_y]],
            "#050505",
            1.15,
            arrow_end=True,
        ),
    ]


def _bar_primitives(crop: np.ndarray, region: dict[str, Any], axis_x: float,
                    axis_y: float, top_y: float, right_x: float) -> list[dict[str, Any]]:
    h, w = crop.shape[:2]
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    rgb = crop.astype(np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    non_white = np.max(255.0 - rgb, axis=2)
    plot = np.zeros((h, w), dtype=np.uint8)
    px0 = int(max(0, axis_x + 3))
    px1 = int(min(w, right_x - 1))
    py0 = int(max(0, top_y - 4))
    py1 = int(min(h, axis_y + 1))
    colored = ((non_white > 20) & (luma > 82) & (luma < 245) & (chroma < 175)).astype(np.uint8) * 255
    plot[py0:py1, px0:px1] = colored[py0:py1, px0:px1]
    plot = cv2.morphologyEx(plot, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(plot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if bw < 4 or bw > 24 or bh < 4 or area < 28:
            continue
        if y + bh < top_y or y > axis_y + 3:
            continue
        fill = _median_hex(crop, x, y, bw, bh)
        out.append({
            "id": f"chart_{region['id']}_bar_{len(out)+1:02d}",
            "type": "shape",
            "role": "chart_bar",
            "shape": "rectangle",
            "bbox": [_r(rx0 + x), _r(ry0 + y), _r(rx0 + x + bw), _r(ry0 + y + bh)],
            "style": {"fill": fill, "stroke": _darken(fill)},
            "source": "chart_rebuild_cv_bar",
        })
    out.extend(_bar_primitives_from_columns(
        crop,
        region,
        plot,
        px0,
        px1,
        py0,
        py1,
        axis_y,
        rx0,
        ry0,
    ))
    out.sort(key=lambda p: (p["bbox"][0], p["bbox"][1]))
    return out[:8]


def _bar_primitives_from_columns(
    crop: np.ndarray,
    region: dict[str, Any],
    plot: np.ndarray,
    px0: int,
    px1: int,
    py0: int,
    py1: int,
    axis_y: float,
    rx0: float,
    ry0: float,
) -> list[dict[str, Any]]:
    window = plot[py0:py1, px0:px1] > 0
    if window.size == 0:
        return []
    min_count = max(4, int(window.shape[0] * 0.12))
    active = np.where(window.sum(axis=0) >= min_count)[0]
    groups = _runs(active.tolist())
    out = []
    for start, end in groups:
        width = end - start + 1
        if width < 4 or width > 24:
            continue
        cols = window[:, start:end + 1]
        rows = np.where(cols.any(axis=1))[0]
        if rows.size == 0:
            continue
        y_top = py0 + int(rows.min())
        y_bot = py0 + int(rows.max()) + 1
        x_left = px0 + start
        x_right = px0 + end + 1
        height = y_bot - y_top
        if height < 6:
            continue
        if abs(y_bot - axis_y) > 12:
            continue
        fill = _median_hex(crop, x_left, y_top, x_right - x_left, y_bot - y_top)
        out.append({
            "id": f"chart_{region['id']}_bar_col_{len(out)+1:02d}",
            "type": "shape",
            "role": "chart_bar",
            "shape": "rectangle",
            "bbox": [_r(rx0 + x_left), _r(ry0 + y_top), _r(rx0 + x_right), _r(ry0 + y_bot)],
            "style": {"fill": fill, "stroke": _darken(fill)},
            "source": "chart_rebuild_cv_bar",
        })
    return out


def _line_primitives(crop: np.ndarray, region: dict[str, Any],
                     segments: list[tuple[float, float, float, float, float]],
                     axis_x: float, axis_y: float, top_y: float,
                     right_x: float) -> list[dict[str, Any]]:
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    out = []
    for x0, y0, x1, y1, length in segments:
        if length < 22:
            continue
        if not _inside_plot(x0, y0, axis_x, axis_y, top_y, right_x):
            continue
        if not _inside_plot(x1, y1, axis_x, axis_y, top_y, right_x):
            continue
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx <= 3 or dy <= 3:
            continue
        color = _line_color(crop, x0, y0, x1, y1)
        if color == "#ffffff":
            continue
        out.append(_edge(
            region,
            f"trend_{len(out)+1:02d}",
            [[rx0 + x0, ry0 + y0], [rx0 + x1, ry0 + y1]],
            color,
            1.35,
            arrow_end=False,
        ))
    out.sort(key=lambda p: -_path_len(p["path"]))
    return out[:7]


def _inside_plot(x: float, y: float, axis_x: float, axis_y: float,
                 top_y: float, right_x: float) -> bool:
    return axis_x + 2 <= x <= right_x + 3 and top_y - 4 <= y <= axis_y - 2


def _edge(region: dict[str, Any], suffix: str, path: list[list[float]],
          stroke: str, width: float, *, arrow_end: bool) -> dict[str, Any]:
    return {
        "id": f"chart_{region['id']}_{suffix}",
        "type": "edge",
        "role": "chart_axis_or_trend",
        "bbox": _path_bbox(path),
        "path": [[_r(x), _r(y)] for x, y in path],
        "style": {
            "stroke": stroke,
            "stroke_width": width,
            "arrow_start": False,
            "arrow_end": arrow_end,
        },
        "source": "chart_rebuild_cv_edge",
    }


def _apply_region_primitives(
    program: dict[str, Any],
    region: dict[str, Any],
    new_primitives: list[dict[str, Any]],
    *,
    mode: str,
    replace_policy: str,
) -> tuple[dict[str, Any], list[str]]:
    updated = copy.deepcopy(program)
    box = tuple(float(v) for v in region["bbox"])
    deleted = []
    kept = []
    for primitive in updated.get("primitives", []):
        if replace_policy == "replace" and primitive.get("type") in {"edge", "shape"}:
            bbox = _primitive_bbox(primitive)
            if bbox and _bbox_matches(bbox, box, mode=mode):
                deleted.append(str(primitive.get("id")))
                continue
        kept.append(primitive)
    used = {p.get("id") for p in kept}
    for primitive in new_primitives:
        copied = copy.deepcopy(primitive)
        base_id = str(copied.get("id") or "chart")
        if base_id in used:
            i = 1
            while f"{base_id}_{i}" in used:
                i += 1
            copied["id"] = f"{base_id}_{i}"
        used.add(copied["id"])
        kept.append(copied)
    updated["primitives"] = kept
    _refresh_counts(updated)
    return updated, deleted


def _write_variant(program: dict[str, Any], base: Path, tag: str,
                   font_family: str, report: dict[str, Any]) -> dict[str, Any]:
    drawio = Path(f"{base}.{_safe(tag)}.drawio")
    program_path = Path(f"{base}.{_safe(tag)}.vp_program.json")
    report_path = Path(f"{base}.{_safe(tag)}.report.json")
    save_program(program, program_path)
    compile_program_to_drawio(program, drawio, font_family=font_family)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    return {
        "name": _safe(tag),
        "drawio": str(drawio),
        "program": str(program_path),
        "report": str(report_path),
    }


def _dedupe_primitives(primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for primitive in primitives:
        bbox = _primitive_bbox(primitive)
        if not bbox:
            continue
        if any(_iou(bbox, _primitive_bbox(other) or bbox) > 0.82 for other in out if other["type"] == primitive["type"]):
            continue
        out.append(primitive)
    return out


def _line_color(crop: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> str:
    points = []
    steps = max(2, int(math.hypot(x1 - x0, y1 - y0)))
    h, w = crop.shape[:2]
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 * (1 - t) + x1 * t))
        y = int(round(y0 * (1 - t) + y1 * t))
        if 0 <= x < w and 0 <= y < h:
            pixel = crop[y, x]
            if int(pixel.max()) < 245:
                points.append(pixel)
    if not points:
        return "#050505"
    arr = np.asarray(points)
    med = np.median(arr, axis=0).astype(int)
    return f"#{med[0]:02x}{med[1]:02x}{med[2]:02x}"


def _runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    runs = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = value
        prev = value
    runs.append((start, prev))
    return runs


def _median_hex(crop: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    patch = crop[y:y + h, x:x + w].reshape(-1, 3)
    if patch.size == 0:
        return "#d6e3f2"
    med = np.median(patch, axis=0).astype(int)
    return f"#{med[0]:02x}{med[1]:02x}{med[2]:02x}"


def _darken(hex_color: str) -> str:
    vals = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    vals = [max(0, int(v * 0.7)) for v in vals]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _primitive_bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = primitive.get("bbox")
    if bbox and len(bbox) == 4:
        return tuple(float(v) for v in bbox)
    path = primitive.get("path") or []
    if not path:
        return None
    return _path_bbox(path)


def _path_bbox(path: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    return [_r(min(xs)), _r(min(ys)), _r(max(xs)), _r(max(ys))]


def _path_len(path: list[list[float]]) -> float:
    return sum(math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) for i in range(1, len(path)))


def _bbox_matches(
    bbox: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    *,
    mode: str,
) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = box
    if mode == "intersect":
        return not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1)
    if mode == "contained":
        return x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _iou(a: tuple[float, float, float, float] | list[float],
         b: tuple[float, float, float, float] | list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(1.0, union)


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }


def _safe(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()[:64] or "variant"


def _r(value: float) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    main()
