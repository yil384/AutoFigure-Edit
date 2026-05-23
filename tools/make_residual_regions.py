#!/usr/bin/env python3
"""Generate local-composition regions from source/render residual hotspots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create panel regions around source/render residual hotspots")
    ap.add_argument("source_image")
    ap.add_argument("rendered_image")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--debug-image", default=None)
    ap.add_argument("--prefix", default="residual")
    ap.add_argument("--max-regions", type=int, default=24)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--diff-threshold", type=int, default=38)
    ap.add_argument("--min-width", type=int, default=22)
    ap.add_argument("--min-height", type=int, default=14)
    ap.add_argument("--max-width", type=int, default=360)
    ap.add_argument("--max-height", type=int, default=240)
    ap.add_argument("--min-area", type=int, default=450)
    ap.add_argument("--max-area-fraction", type=float, default=0.07)
    ap.add_argument("--nms-iou", type=float, default=0.35)
    args = ap.parse_args()

    source = _read_rgb(args.source_image)
    rendered = _read_rgb(args.rendered_image)
    h = min(source.shape[0], rendered.shape[0])
    w = min(source.shape[1], rendered.shape[1])
    source = source[:h, :w]
    rendered = rendered[:h, :w]

    mask = _residual_mask(source, rendered, diff_threshold=args.diff_threshold)
    regions = _extract_regions(
        mask,
        width=w,
        height=h,
        prefix=args.prefix,
        max_regions=args.max_regions,
        pad=args.pad,
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        min_area=args.min_area,
        max_area_fraction=args.max_area_fraction,
        nms_iou=args.nms_iou,
    )

    payload = {
        "version": "residual-regions-0.1",
        "source_image": args.source_image,
        "rendered_image": args.rendered_image,
        "canvas": {"width": w, "height": h},
        "config": {
            "max_regions": args.max_regions,
            "pad": args.pad,
            "diff_threshold": args.diff_threshold,
            "nms_iou": args.nms_iou,
        },
        "panel_regions": regions,
        "counts": {"total": len(regions)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    if args.debug_image:
        _write_debug_image(args.source_image, args.debug_image, regions)
    print(json.dumps({
        "output": str(output),
        "debug_image": args.debug_image,
        "regions": len(regions),
    }, indent=2))


def _read_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def _residual_mask(
    source: np.ndarray,
    rendered: np.ndarray,
    *,
    diff_threshold: int,
) -> np.ndarray:
    src_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    ren_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(src_gray, ren_gray)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    diff_mask = (diff >= diff_threshold).astype(np.uint8) * 255

    src_edge = cv2.Canny(src_gray, 70, 160)
    ren_edge = cv2.Canny(ren_gray, 70, 160)
    edge_mask = cv2.bitwise_xor(
        cv2.dilate(src_edge, np.ones((2, 2), np.uint8), iterations=1),
        cv2.dilate(ren_edge, np.ones((2, 2), np.uint8), iterations=1),
    )
    mask = cv2.bitwise_or(diff_mask, edge_mask)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return mask


def _extract_regions(
    mask: np.ndarray,
    *,
    width: int,
    height: int,
    prefix: str,
    max_regions: int,
    pad: int,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    min_area: int,
    max_area_fraction: float,
    nms_iou: float,
) -> list[dict]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    canvas_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        bw = x1 - x0
        bh = y1 - y0
        area = bw * bh
        if bw < min_width or bh < min_height:
            continue
        if bw > max_width or bh > max_height:
            continue
        if area < min_area or area > canvas_area * max_area_fraction:
            continue
        crop = mask[y0:y1, x0:x1]
        residual_pixels = int((crop > 0).sum())
        density = residual_pixels / max(1, area)
        if density < 0.025:
            continue
        candidates.append({
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "area": float(area),
            "residual_pixels": residual_pixels,
            "density": density,
            "score": residual_pixels * (0.5 + density),
        })

    candidates.sort(key=lambda item: (-item["score"], item["bbox"][1], item["bbox"][0]))
    kept = []
    for item in candidates:
        if any(_iou(item["bbox"], other["bbox"]) >= nms_iou for other in kept):
            continue
        kept.append(item)
        if len(kept) >= max_regions:
            break

    regions = []
    for index, item in enumerate(kept, start=1):
        regions.append({
            "id": f"{prefix}_{index:02d}",
            "bbox": [round(value, 3) for value in item["bbox"]],
            "kind": "residual_hotspot",
            "area": round(item["area"], 3),
            "residual_pixels": item["residual_pixels"],
            "density": round(item["density"], 6),
            "score": round(item["score"], 3),
        })
    return regions


def _write_debug_image(source_path: str | Path, output_path: str | Path, regions: list[dict]) -> None:
    image = Image.open(source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=2)
        draw.text((x0 + 2, y0 + 2), region["id"], fill=(220, 30, 30))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)


def _iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(1.0, area)


if __name__ == "__main__":
    main()
