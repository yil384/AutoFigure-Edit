#!/usr/bin/env python3
"""Generate/evaluate native semantic chart grammar variants for residual regions."""
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
        description="Sweep semantic chart grammar variants as native draw.io primitives")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("regions_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="chart_semantic_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--region-ids", required=True)
    ap.add_argument("--modes", default="bars,lineplot,auto")
    ap.add_argument("--delete-policies", default="plot,region,add")
    ap.add_argument("--include-combo", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    source = cv2.cvtColor(cv2.imread(args.source_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if source is None:
        raise FileNotFoundError(args.source_image)
    program = load_program(args.program_json)
    wanted = {item.strip() for item in args.region_ids.split(",") if item.strip()}
    regions = [
        region for region in load_panel_regions(args.regions_json)
        if str(region.get("id")) in wanted
    ]
    modes = _csv(args.modes)
    delete_policies = _csv(args.delete_policies)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name

    variants = []
    payloads = []
    for region in regions:
        for mode in modes:
            primitives, report = detect_semantic_chart_primitives(source, region, mode)
            payloads.append({
                "region": region,
                "mode": mode,
                "new_primitives": primitives,
                "detect_report": report,
            })
            if not primitives:
                continue
            for delete_policy in delete_policies:
                updated, deleted = _apply_chart_primitives(
                    program,
                    region,
                    primitives,
                    report,
                    delete_policy=delete_policy,
                )
                variants.append(_write_variant(
                    updated,
                    base,
                    f"{region['id']}_{mode}_{delete_policy}",
                    args.font_family,
                    {
                        "operation": "semantic_chart_grammar",
                        "region": region,
                        "mode": mode,
                        "delete_policy": delete_policy,
                        "deleted_ids": deleted,
                        "added": len(primitives),
                        "detect_report": report,
                    },
                ))

    if args.include_combo:
        for delete_policy in delete_policies:
            updated = copy.deepcopy(program)
            all_deleted: list[str] = []
            applied = []
            for payload in payloads:
                if payload["mode"] != "auto" or not payload["new_primitives"]:
                    continue
                updated, deleted = _apply_chart_primitives(
                    updated,
                    payload["region"],
                    payload["new_primitives"],
                    payload["detect_report"],
                    delete_policy=delete_policy,
                )
                all_deleted.extend(deleted)
                applied.append({
                    "region": payload["region"],
                    "mode": payload["mode"],
                    "added": len(payload["new_primitives"]),
                    "detect_report": payload["detect_report"],
                })
            if applied:
                variants.append(_write_variant(
                    updated,
                    base,
                    f"combo_auto_{delete_policy}",
                    args.font_family,
                    {
                        "operation": "semantic_chart_grammar_combo",
                        "delete_policy": delete_policy,
                        "deleted_ids": all_deleted,
                        "applied": applied,
                    },
                ))

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "regions_json": args.regions_json,
        "modes": modes,
        "delete_policies": delete_policies,
        "payloads": payloads,
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


def detect_semantic_chart_primitives(
    source: np.ndarray,
    region: dict[str, Any],
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in region["bbox"]]
    crop = source[y0:y1, x0:x1]
    if crop.size == 0:
        return [], {"error": "empty_crop"}
    axis = _infer_axes(crop)
    auto_mode = _auto_mode(crop, axis)
    selected_mode = auto_mode if mode == "auto" else mode
    primitives = _axis_primitives(region, axis)
    if selected_mode == "bars":
        primitives.extend(_bar_primitives(crop, region, axis))
        primitives.extend(_single_trend_primitives(crop, region, axis))
    elif selected_mode == "lineplot":
        primitives.extend(_lineplot_primitives(crop, region, axis))
    else:
        primitives.extend(_bar_primitives(crop, region, axis))
        primitives.extend(_single_trend_primitives(crop, region, axis))
        primitives.extend(_lineplot_primitives(crop, region, axis))
    primitives = _dedupe_primitives(primitives)
    return primitives, {
        "requested_mode": mode,
        "selected_mode": selected_mode,
        "auto_mode": auto_mode,
        "axis": axis,
        "plot_bbox": _plot_bbox(region, axis),
        "counts": {
            "edges": sum(1 for p in primitives if p["type"] == "edge"),
            "shapes": sum(1 for p in primitives if p["type"] == "shape"),
        },
        "added": len(primitives),
    }


def _infer_axes(crop: np.ndarray) -> dict[str, float]:
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 55, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18,
                            minLineLength=24, maxLineGap=5)
    horizontals = []
    verticals = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            xa, ya, xb, yb = [float(v) for v in raw]
            length = math.hypot(xb - xa, yb - ya)
            if abs(yb - ya) <= 3 and length >= max(24, w * 0.35):
                horizontals.append((min(xa, xb), max(xa, xb), (ya + yb) / 2.0, length))
            if abs(xb - xa) <= 3 and length >= max(24, h * 0.35):
                verticals.append(((xa + xb) / 2.0, min(ya, yb), max(ya, yb), length))
    if horizontals:
        hx0, hx1, axis_y, _ = max(horizontals, key=lambda item: item[3] + item[2] * 0.6)
    else:
        hx0, hx1, axis_y = w * 0.36, w * 0.92, h * 0.82
    if verticals:
        axis_x, vy0, vy1, _ = min(verticals, key=lambda item: abs(item[0] - hx0) - item[3] * 0.03)
        top_y = min(vy0, vy1)
    else:
        axis_x, top_y = hx0, h * 0.12
    right_x = max(hx1, w * 0.88)
    return {
        "x": _r(axis_x),
        "y": _r(axis_y),
        "top_y": _r(top_y),
        "right_x": _r(right_x),
    }


def _auto_mode(crop: np.ndarray, axis: dict[str, float]) -> str:
    plot = _plot_crop(crop, axis)
    if plot.size == 0:
        return "bars"
    hsv = cv2.cvtColor(plot, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    colored = ((sat > 35) & (val > 60) & (val < 248)).astype(np.uint8)
    vertical_mass = colored.sum(axis=0)
    if int((vertical_mass > max(3, colored.shape[0] * 0.18)).sum()) >= 14:
        return "bars"
    return "lineplot"


def _axis_primitives(region: dict[str, Any], axis: dict[str, float]) -> list[dict[str, Any]]:
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    return [
        _edge(
            region, "axis_x",
            [[rx0 + axis["x"], ry0 + axis["y"]], [rx0 + axis["right_x"], ry0 + axis["y"]]],
            "#050505", 1.25, arrow_end=True,
        ),
        _edge(
            region, "axis_y",
            [[rx0 + axis["x"], ry0 + axis["y"]], [rx0 + axis["x"], ry0 + axis["top_y"]]],
            "#050505", 1.25, arrow_end=True,
        ),
    ]


def _bar_primitives(crop: np.ndarray, region: dict[str, Any], axis: dict[str, float]) -> list[dict[str, Any]]:
    h, w = crop.shape[:2]
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    rgb = crop.astype(np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    non_white = np.max(255.0 - rgb, axis=2)
    px0 = int(max(0, axis["x"] + 3))
    px1 = int(min(w, axis["right_x"] - 1))
    py0 = int(max(0, axis["top_y"] - 4))
    py1 = int(min(h, axis["y"] + 2))
    mask = ((non_white > 18) & (luma > 80) & (luma < 246) & (chroma < 145)).astype(np.uint8)
    window = mask[py0:py1, px0:px1]
    if window.size == 0:
        return []
    min_count = max(4, int(window.shape[0] * 0.11))
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
        if abs(y_bot - axis["y"]) > 14:
            continue
        x_left = px0 + start
        x_right = px0 + end + 1
        if y_bot - y_top < 5:
            continue
        fill = _median_hex(crop, x_left, y_top, x_right - x_left, y_bot - y_top)
        out.append({
            "id": f"semchart_{region['id']}_bar_{len(out)+1:02d}",
            "type": "shape",
            "role": "semantic_chart_bar",
            "shape": "rectangle",
            "bbox": [_r(rx0 + x_left), _r(ry0 + y_top), _r(rx0 + x_right), _r(ry0 + y_bot)],
            "style": {"fill": fill, "stroke": _darken(fill)},
            "source": "semantic_chart_bar",
        })
    return out[:6]


def _single_trend_primitives(crop: np.ndarray, region: dict[str, Any], axis: dict[str, float]) -> list[dict[str, Any]]:
    plot_box = _local_plot_box(axis)
    x0, y0, x1, y1 = plot_box
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=14,
                            minLineLength=28, maxLineGap=6)
    candidates = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            xa, ya, xb, yb = [float(v) for v in raw]
            if not (_inside(xa, ya, plot_box) and _inside(xb, yb, plot_box)):
                continue
            dx = xb - xa
            dy = yb - ya
            length = math.hypot(dx, dy)
            if length < 34 or abs(dx) < 16 or abs(dy) < 10:
                continue
            # Image y decreases for an upward trend.
            if dx * dy >= 0:
                continue
            candidates.append((length, xa, ya, xb, yb))
    if not candidates:
        return []
    _, xa, ya, xb, yb = max(candidates, key=lambda item: item[0])
    if xa > xb:
        xa, ya, xb, yb = xb, yb, xa, ya
    color = _saturated_line_color(crop, xa, ya, xb, yb)
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    return [_edge(
        region, "trend",
        [[rx0 + xa, ry0 + ya], [rx0 + xb, ry0 + yb]],
        color, 1.55, arrow_end=True,
    )]


def _lineplot_primitives(crop: np.ndarray, region: dict[str, Any], axis: dict[str, float]) -> list[dict[str, Any]]:
    plot_box = _local_plot_box(axis)
    x0, y0, x1, y1 = [int(round(v)) for v in plot_box]
    plot = crop[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if plot.size == 0:
        return []
    hsv = cv2.cvtColor(plot, cv2.COLOR_RGB2HSV)
    rgb = plot.astype(np.int16)
    masks = [
        ("green", ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 28) & (hsv[:, :, 2] > 65)), "#5f8f4e", False),
        ("blue", ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 24) & (hsv[:, :, 2] > 65)), "#4a82a8", False),
        ("red", ((rgb[:, :, 0] > rgb[:, :, 2] + 8) & (rgb[:, :, 0] > rgb[:, :, 1] + 2) & (hsv[:, :, 1] > 18) & (hsv[:, :, 2] > 65)), "#a26b75", True),
    ]
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    out = []
    for name, mask, color, dashed in masks:
        pts = _fit_polyline(mask.astype(np.uint8), x0, y0)
        if len(pts) < 2:
            continue
        global_pts = [[_r(rx0 + x), _r(ry0 + y)] for x, y in pts]
        out.append(_edge(
            region, f"line_{name}",
            global_pts,
            color,
            1.45,
            arrow_end=False,
            dashed=dashed,
        ))
    return out[:4]


def _fit_polyline(mask: np.ndarray, ox: int, oy: int) -> list[list[float]]:
    ys, xs = np.where(mask > 0)
    if xs.size < 16:
        return []
    width = mask.shape[1]
    bins = np.linspace(0, width - 1, 7)
    pts = []
    for left, right in zip(bins[:-1], bins[1:]):
        keep = (xs >= left) & (xs <= right)
        if int(keep.sum()) < 3:
            continue
        x = float(np.median(xs[keep]))
        y = float(np.percentile(ys[keep], 45))
        pts.append([ox + x, oy + y])
    if len(pts) < 2:
        return []
    pts.sort(key=lambda p: p[0])
    return _rdp(pts, epsilon=1.6)


def _apply_chart_primitives(
    program: dict[str, Any],
    region: dict[str, Any],
    new_primitives: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    delete_policy: str,
) -> tuple[dict[str, Any], list[str]]:
    updated = copy.deepcopy(program)
    deleted = []
    if delete_policy == "add":
        keep = list(updated.get("primitives", []))
    else:
        delete_box = tuple(report.get("plot_bbox") or region["bbox"])
        if delete_policy == "region":
            delete_box = tuple(float(v) for v in region["bbox"])
        keep = []
        for primitive in updated.get("primitives", []):
            if primitive.get("type") in {"edge", "shape"}:
                bbox = _primitive_bbox(primitive)
                if bbox and _center_in(bbox, delete_box, margin=3.0):
                    deleted.append(str(primitive.get("id")))
                    continue
            keep.append(primitive)
    used = {p.get("id") for p in keep}
    for primitive in new_primitives:
        item = copy.deepcopy(primitive)
        base_id = str(item.get("id") or "semchart")
        if base_id in used:
            suffix = 1
            while f"{base_id}_{suffix}" in used:
                suffix += 1
            item["id"] = f"{base_id}_{suffix}"
        used.add(item["id"])
        keep.append(item)
    updated["primitives"] = keep
    _refresh_counts(updated)
    return updated, deleted


def _write_variant(
    program: dict[str, Any],
    base: Path,
    tag: str,
    font_family: str,
    report: dict[str, Any],
) -> dict[str, str]:
    tag = _safe(tag)
    drawio = Path(f"{base}.{tag}.drawio")
    program_path = Path(f"{base}.{tag}.vp_program.json")
    report_path = Path(f"{base}.{tag}.report.json")
    save_program(program, program_path)
    compile_program_to_drawio(program, drawio, font_family=font_family)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    return {
        "name": tag,
        "drawio": str(drawio),
        "program": str(program_path),
        "report": str(report_path),
    }


def _plot_crop(crop: np.ndarray, axis: dict[str, float]) -> np.ndarray:
    x0, y0, x1, y1 = [int(round(v)) for v in _local_plot_box(axis)]
    return crop[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]


def _local_plot_box(axis: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(axis["x"]) + 3.0,
        float(axis["top_y"]) - 2.0,
        float(axis["right_x"]) + 2.0,
        float(axis["y"]) - 2.0,
    )


def _plot_bbox(region: dict[str, Any], axis: dict[str, float]) -> list[float]:
    rx0, ry0 = float(region["bbox"][0]), float(region["bbox"][1])
    x0, y0, x1, y1 = _local_plot_box(axis)
    return [_r(rx0 + x0), _r(ry0 + y0), _r(rx0 + x1), _r(ry0 + y1)]


def _edge(
    region: dict[str, Any],
    suffix: str,
    path: list[list[float]],
    stroke: str,
    width: float,
    *,
    arrow_end: bool,
    dashed: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"semchart_{region['id']}_{suffix}",
        "type": "edge",
        "role": "semantic_chart_edge",
        "bbox": _path_bbox(path),
        "path": [[_r(x), _r(y)] for x, y in path],
        "style": {
            "stroke": stroke,
            "stroke_width": width,
            "arrow_start": False,
            "arrow_end": arrow_end,
            "dashed": dashed,
        },
        "source": "semantic_chart_edge",
    }


def _saturated_line_color(crop: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> str:
    points = []
    h, w = crop.shape[:2]
    steps = max(2, int(math.hypot(x1 - x0, y1 - y0)))
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 * (1 - t) + x1 * t))
        y = int(round(y0 * (1 - t) + y1 * t))
        for yy in range(y - 1, y + 2):
            for xx in range(x - 1, x + 2):
                if 0 <= xx < w and 0 <= yy < h:
                    pix = crop[yy, xx].astype(int)
                    if int(pix.max()) < 245 and int(pix.max() - pix.min()) > 18:
                        points.append(pix)
    if not points:
        return "#4a82a8"
    arr = np.asarray(points)
    # Pick a saturated median-ish color rather than a pale anti-aliased average.
    chroma = arr.max(axis=1) - arr.min(axis=1)
    chosen = arr[np.argsort(chroma)[max(0, int(len(arr) * 0.65) - 1)]]
    return f"#{int(chosen[0]):02x}{int(chosen[1]):02x}{int(chosen[2]):02x}"


def _median_hex(crop: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    patch = crop[y:y + h, x:x + w].reshape(-1, 3)
    if patch.size == 0:
        return "#d6e3f2"
    med = np.median(patch, axis=0).astype(int)
    return f"#{med[0]:02x}{med[1]:02x}{med[2]:02x}"


def _darken(hex_color: str) -> str:
    vals = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    vals = [max(0, int(v * 0.68)) for v in vals]
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _dedupe_primitives(primitives: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for primitive in primitives:
        bbox = _primitive_bbox(primitive)
        if not bbox:
            continue
        if any(
            other.get("type") == primitive.get("type") and
            _iou(bbox, _primitive_bbox(other) or bbox) > 0.88
            for other in out
        ):
            continue
        out.append(primitive)
    return out


def _primitive_bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = primitive.get("bbox")
    if bbox and len(bbox) == 4:
        return tuple(float(v) for v in bbox)
    path = primitive.get("path") or []
    if not path:
        return None
    return tuple(_path_bbox(path))


def _path_bbox(path: list[list[float]]) -> list[float]:
    xs = [float(p[0]) for p in path]
    ys = [float(p[1]) for p in path]
    return [_r(min(xs)), _r(min(ys)), _r(max(xs)), _r(max(ys))]


def _center_in(bbox: tuple[float, float, float, float], box: tuple[float, float, float, float], *, margin: float) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = [float(v) for v in box]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bx0 - margin <= cx <= bx1 + margin and by0 - margin <= cy <= by1 + margin


def _inside(x: float, y: float, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


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


def _rdp(path: list[list[float]], epsilon: float) -> list[list[float]]:
    if len(path) <= 2:
        return path
    arr = np.asarray(path, dtype=np.float32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(arr, epsilon, False)
    pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
    if pts[0] != path[0]:
        pts.insert(0, path[0])
    if pts[-1] != path[-1]:
        pts.append(path[-1])
    return pts


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


def _csv(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def _safe(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()[:90] or "variant"


def _r(value: float) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    main()
