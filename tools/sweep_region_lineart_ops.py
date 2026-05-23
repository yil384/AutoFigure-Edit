#!/usr/bin/env python3
"""Generate/evaluate native line-art rebuild variants for residual regions."""
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
        description="Sweep region-local source line-art primitives")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("regions_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="region_lineart_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--region-ids", required=True)
    ap.add_argument("--min-lengths", default="10,14,18")
    ap.add_argument("--max-lines-values", default="16,28,44")
    ap.add_argument("--delete-policies", default="add,region")
    ap.add_argument("--skip-existing-similar", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--line-similarity-distance", type=float, default=2.8)
    ap.add_argument("--line-similarity-angle", type=float, default=6.0)
    ap.add_argument("--line-similarity-overlap", type=float, default=0.58)
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    source = cv2.cvtColor(cv2.imread(args.source_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    program = load_program(args.program_json)
    regions = [
        region for region in load_panel_regions(args.regions_json)
        if str(region.get("id")) in {x.strip() for x in args.region_ids.split(",") if x.strip()}
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    variants = []
    payloads = []
    for region in regions:
        for min_length in _floats(args.min_lengths):
            for max_lines in _ints(args.max_lines_values):
                primitives, report = detect_lineart_primitives(
                    source,
                    region,
                    min_length=min_length,
                    max_lines=max_lines,
                )
                payloads.append({
                    "region": region,
                    "min_length": min_length,
                    "max_lines": max_lines,
                    "new_primitives": primitives,
                    "detect_report": report,
                })
                if not primitives:
                    continue
                for delete_policy in _csv(args.delete_policies):
                    updated, deleted, added, skipped_existing = _apply_primitives(
                        program,
                        region,
                        primitives,
                        delete_policy=delete_policy,
                        skip_existing_similar=args.skip_existing_similar,
                        similarity_distance=args.line_similarity_distance,
                        similarity_angle=args.line_similarity_angle,
                        similarity_overlap=args.line_similarity_overlap,
                    )
                    if added <= 0 and not deleted:
                        continue
                    variants.append(_write_variant(
                        updated,
                        base,
                        f"{region['id']}_l{min_length:g}_m{max_lines}_{delete_policy}",
                        args.font_family,
                        {
                            "operation": "region_lineart_rebuild",
                            "region": region,
                            "min_length": min_length,
                            "max_lines": max_lines,
                            "delete_policy": delete_policy,
                            "deleted_ids": deleted,
                            "detected": len(primitives),
                            "added": added,
                            "skipped_existing": skipped_existing,
                            "detect_report": report,
                        },
                    ))

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "regions_json": args.regions_json,
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


def detect_lineart_primitives(
    source: np.ndarray,
    region: dict[str, Any],
    *,
    min_length: float,
    max_lines: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in region["bbox"]]
    crop = source[y0:y1, x0:x1]
    if crop.size == 0:
        return [], {"error": "empty_crop"}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 22, 22)
    edges = cv2.Canny(gray, 48, 145)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(10, int(min_length * 0.9)),
        minLineLength=min_length,
        maxLineGap=4,
    )
    segments = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            xa, ya, xb, yb = [float(v) for v in raw]
            length = math.hypot(xb - xa, yb - ya)
            if length < min_length:
                continue
            angle = abs(math.degrees(math.atan2(yb - ya, xb - xa)))
            if _looks_like_text_stroke(xa, ya, xb, yb, length, angle):
                continue
            color = _line_color(crop, xa, ya, xb, yb)
            segments.append({
                "path": [[x0 + xa, y0 + ya], [x0 + xb, y0 + yb]],
                "length": length,
                "angle": angle,
                "color": color,
            })
    segments = _merge_similar_segments(segments)
    segments.sort(key=lambda item: (-item["length"], item["path"][0][1], item["path"][0][0]))
    primitives = []
    for i, seg in enumerate(segments[:max_lines], start=1):
        primitives.append({
            "id": f"lineart_{region['id']}_edge_{i:03d}",
            "type": "edge",
            "role": "semantic_lineart_edge",
            "bbox": _path_bbox(seg["path"]),
            "path": [[_r(x), _r(y)] for x, y in seg["path"]],
            "style": {
                "stroke": seg["color"],
                "stroke_width": 1.05,
                "arrow_start": False,
                "arrow_end": False,
            },
            "source": "semantic_region_lineart",
        })
    return primitives, {
        "raw_segments": 0 if lines is None else int(len(lines)),
        "kept_segments": len(segments),
        "emitted": len(primitives),
        "min_length": min_length,
        "max_lines": max_lines,
    }


def _looks_like_text_stroke(x0: float, y0: float, x1: float, y1: float,
                            length: float, angle: float) -> bool:
    # Short near-vertical/horizontal strokes are often text glyph fragments.
    if length < 12:
        return True
    if length < 18 and (angle < 8 or 82 < angle < 98):
        return True
    return False


def _merge_similar_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg in segments:
        box = _path_bbox(seg["path"])
        if any(_iou(box, _path_bbox(other["path"])) > 0.78 for other in out):
            continue
        out.append(seg)
    return out


def _apply_primitives(
    program: dict[str, Any],
    region: dict[str, Any],
    primitives: list[dict[str, Any]],
    *,
    delete_policy: str,
    skip_existing_similar: bool,
    similarity_distance: float,
    similarity_angle: float,
    similarity_overlap: float,
) -> tuple[dict[str, Any], list[str], int, int]:
    updated = copy.deepcopy(program)
    deleted = []
    if delete_policy == "add":
        keep = list(updated.get("primitives", []))
    else:
        box = tuple(float(v) for v in region["bbox"])
        keep = []
        for primitive in updated.get("primitives", []):
            if primitive.get("type") in {"edge", "shape"}:
                bbox = _primitive_bbox(primitive)
                if bbox and _center_in(bbox, box):
                    deleted.append(str(primitive.get("id")))
                    continue
            keep.append(primitive)
    used = {p.get("id") for p in keep}
    existing_segments = [
        segment for segment in (_segment_from_primitive(p) for p in keep)
        if segment is not None
    ]
    accepted_segments: list[tuple[float, float, float, float]] = []
    skipped_existing = 0
    added = 0
    for primitive in primitives:
        segment = _segment_from_primitive(primitive)
        if (
            skip_existing_similar
            and segment is not None
            and _has_similar_segment(
                segment,
                existing_segments + accepted_segments,
                distance_px=similarity_distance,
                angle_deg=similarity_angle,
                overlap_ratio=similarity_overlap,
            )
        ):
            skipped_existing += 1
            continue
        item = copy.deepcopy(primitive)
        base = str(item.get("id") or "lineart")
        if base in used:
            i = 1
            while f"{base}_{i}" in used:
                i += 1
            item["id"] = f"{base}_{i}"
        used.add(item["id"])
        keep.append(item)
        added += 1
        if segment is not None:
            accepted_segments.append(segment)
    updated["primitives"] = keep
    _refresh_counts(updated)
    return updated, deleted, added, skipped_existing


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


def _line_color(crop: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> str:
    h, w = crop.shape[:2]
    points = []
    steps = max(2, int(math.hypot(x1 - x0, y1 - y0)))
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 * (1 - t) + x1 * t))
        y = int(round(y0 * (1 - t) + y1 * t))
        if 0 <= x < w and 0 <= y < h:
            pix = crop[y, x]
            if int(pix.max()) < 245:
                points.append(pix)
    if not points:
        return "#050505"
    med = np.median(np.asarray(points), axis=0).astype(int)
    if int(med.max()) > 205:
        return "#6f7882"
    return f"#{med[0]:02x}{med[1]:02x}{med[2]:02x}"


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


def _center_in(bbox: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = box
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


def _segment_from_primitive(
    primitive: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    if primitive.get("type") != "edge":
        return None
    path = primitive.get("path") or []
    if len(path) < 2:
        return None
    x0, y0 = path[0]
    x1, y1 = path[-1]
    if math.hypot(float(x1) - float(x0), float(y1) - float(y0)) < 2.0:
        return None
    return (float(x0), float(y0), float(x1), float(y1))


def _has_similar_segment(
    segment: tuple[float, float, float, float],
    existing: list[tuple[float, float, float, float]],
    *,
    distance_px: float,
    angle_deg: float,
    overlap_ratio: float,
) -> bool:
    return any(
        _segments_similar(
            segment,
            other,
            distance_px=distance_px,
            angle_deg=angle_deg,
            overlap_ratio=overlap_ratio,
        )
        for other in existing
    )


def _segments_similar(
    candidate: tuple[float, float, float, float],
    existing: tuple[float, float, float, float],
    *,
    distance_px: float,
    angle_deg: float,
    overlap_ratio: float,
) -> bool:
    cx0, cy0, cx1, cy1 = candidate
    ex0, ey0, ex1, ey1 = existing
    clen = math.hypot(cx1 - cx0, cy1 - cy0)
    elen = math.hypot(ex1 - ex0, ey1 - ey0)
    if clen < 2.0 or elen < 2.0:
        return False
    if _angle_delta(_segment_angle(candidate), _segment_angle(existing)) > angle_deg:
        return False

    ux = (cx1 - cx0) / clen
    uy = (cy1 - cy0) / clen
    ex_proj = sorted([
        (ex0 - cx0) * ux + (ey0 - cy0) * uy,
        (ex1 - cx0) * ux + (ey1 - cy0) * uy,
    ])
    overlap = max(0.0, min(clen, ex_proj[1]) - max(0.0, ex_proj[0]))
    if overlap / max(1.0, min(clen, elen)) < overlap_ratio:
        return False

    distances = [
        abs((px - cx0) * uy - (py - cy0) * ux)
        for px, py in ((ex0, ey0), (ex1, ey1), ((ex0 + ex1) / 2.0, (ey0 + ey1) / 2.0))
    ]
    return min(distances) <= distance_px and np.median(distances) <= distance_px * 1.5


def _segment_angle(segment: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = segment
    return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0


def _angle_delta(a: float, b: float) -> float:
    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }


def _floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


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
