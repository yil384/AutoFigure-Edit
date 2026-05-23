#!/usr/bin/env python3
"""Sweep pure-native rotated text additions from 90-degree OCR evidence."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from png_to_drawio import _detect_ocr_text_single, _merge_ocr_blocks  # noqa: E402
from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add candidate rotated native draw.io text from OCR on rotated source views")
    ap.add_argument("program_json")
    ap.add_argument("source_image")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="rotated_ocr_text_augment")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--modes", default="sharp,gray_autocontrast,unsharp")
    ap.add_argument("--angles", default="-90,90")
    ap.add_argument("--psm-values", default="6,11,12")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--conf-threshold", type=float, default=18.0)
    ap.add_argument("--min-confidence", type=float, default=45.0)
    ap.add_argument("--min-support-count", type=int, default=1)
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--max-variants", type=int, default=160)
    ap.add_argument("--font-sizes", default="7,8,9,10,11,12")
    ap.add_argument("--width-scales", default="0.88,1.0,1.12")
    ap.add_argument("--height-scales", default="0.9,1.08")
    ap.add_argument("--text-layouts", default="single,stacked")
    ap.add_argument("--text-corrections", default="",
                    help="comma-separated OCR correction pairs like raw=clean")
    ap.add_argument("--compound-overlap-labels", action="store_true",
                    help="also try adjacent OCR words as one multiline axis label")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name

    program = load_program(args.program_json)
    candidates = collect_rotated_text_candidates(
        args.source_image,
        modes=_parse_csv(args.modes),
        angles=_parse_ints(args.angles),
        psm_values=_parse_ints(args.psm_values),
        scale=args.scale,
        conf_threshold=args.conf_threshold,
        min_confidence=args.min_confidence,
        min_support_count=args.min_support_count,
        corrections=_parse_corrections(args.text_corrections),
        compound_overlap_labels=args.compound_overlap_labels,
    )
    candidates = _dedupe_against_existing(candidates, program)
    candidates = candidates[: max(0, args.max_candidates)]

    variants: list[dict[str, Any]] = []
    variant_index = 0
    for candidate in candidates:
        for layout in _parse_csv(args.text_layouts):
            text = _layout_text(candidate["text"], layout)
            if not text:
                continue
            for font_size in _parse_ints(args.font_sizes):
                for width_scale in _parse_floats(args.width_scales):
                    for height_scale in _parse_floats(args.height_scales):
                        if variant_index >= args.max_variants:
                            break
                        updated = copy.deepcopy(program)
                        primitive = _primitive_from_candidate(
                            candidate,
                            text=text,
                            font_size=font_size,
                            width_scale=width_scale,
                            height_scale=height_scale,
                            index=variant_index,
                        )
                        updated.setdefault("primitives", []).append(primitive)
                        _refresh_counts(updated)
                        name = (
                            f"rot{variant_index:03d}_"
                            f"{_safe_name(candidate['text'])[:32]}"
                            f"_r{int(candidate['rotation'])}"
                            f"_fs{font_size}"
                            f"_w{int(round(width_scale * 100)):03d}"
                            f"_h{int(round(height_scale * 100)):03d}"
                            f"_{layout}"
                        )
                        drawio = Path(f"{base}.{name}.drawio")
                        program_path = Path(f"{base}.{name}.vp_program.json")
                        report_path = Path(f"{base}.{name}.report.json")
                        save_program(updated, program_path)
                        compile_program_to_drawio(
                            updated,
                            drawio,
                            font_family=args.font_family,
                        )
                        report_path.write_text(json.dumps({
                            "operation": "add_rotated_ocr_text",
                            "candidate": candidate,
                            "primitive": primitive,
                            "layout": layout,
                            "font_size": font_size,
                            "width_scale": width_scale,
                            "height_scale": height_scale,
                        }, indent=2, ensure_ascii=True))
                        variants.append({
                            "name": name,
                            "drawio": str(drawio),
                            "program": str(program_path),
                            "report": str(report_path),
                            "candidate": candidate,
                        })
                        variant_index += 1
                    if variant_index >= args.max_variants:
                        break
                if variant_index >= args.max_variants:
                    break
            if variant_index >= args.max_variants:
                break
        if variant_index >= args.max_variants:
            break

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "baseline_drawio": args.baseline_drawio,
        "config": {
            "modes": _parse_csv(args.modes),
            "angles": _parse_ints(args.angles),
            "psm_values": _parse_ints(args.psm_values),
            "scale": args.scale,
            "conf_threshold": args.conf_threshold,
            "min_confidence": args.min_confidence,
            "min_support_count": args.min_support_count,
            "font_sizes": _parse_ints(args.font_sizes),
            "width_scales": _parse_floats(args.width_scales),
            "height_scales": _parse_floats(args.height_scales),
            "text_layouts": _parse_csv(args.text_layouts),
            "text_corrections": _parse_corrections(args.text_corrections),
            "compound_overlap_labels": args.compound_overlap_labels,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "variants": variants,
    }, indent=2, ensure_ascii=True))

    if args.generate_only or not variants:
        print(json.dumps({
            "manifest": str(manifest_path),
            "candidate_count": len(candidates),
            "variants": [item["drawio"] for item in variants],
        }, indent=2))
        return

    ranking_path = Path(f"{base}.ranking.json")
    eval_variants = [Path(item["drawio"]) for item in variants]
    if args.baseline_drawio:
        eval_variants.insert(0, Path(args.baseline_drawio))
    _exec_evaluate(
        source_image=args.source_image,
        variants=eval_variants,
        ranking_path=ranking_path,
        best_stem=Path(f"{base}.best"),
        manifest_path=manifest_path,
        drawio_cli=args.drawio_cli,
    )


def collect_rotated_text_candidates(
    image_path: str | Path,
    *,
    modes: list[str],
    angles: list[int],
    psm_values: list[int],
    scale: float,
    conf_threshold: float,
    min_confidence: float,
    min_support_count: int,
    corrections: dict[str, str],
    compound_overlap_labels: bool,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    grouped: dict[int, list[dict[str, Any]]] = {}
    for angle in angles:
        if angle not in {-90, 90}:
            continue
        rotated = image.rotate(angle, expand=True)
        rotation = 270 if angle == -90 else 90
        raw_items: list[dict[str, Any]] = []
        for mode in modes:
            view = _preprocess(rotated, mode)
            for psm in psm_values:
                for item in _detect_ocr_text_single(
                    view,
                    conf_threshold=conf_threshold,
                    scale=scale,
                    psm=psm,
                ):
                    raw_text = _normalize_text(item.get("text", ""))
                    text = _apply_correction(raw_text, corrections)
                    if not _usable_text(text):
                        continue
                    conf = float(item.get("conf", 0.0))
                    if conf < min_confidence:
                        continue
                    mapped = _map_bbox_from_rotated(
                        item.get("bbox", [0, 0, 1, 1]),
                        angle=angle,
                        width=width,
                        height=height,
                    )
                    if mapped is None or not _looks_like_vertical_source_text(mapped):
                        continue
                    raw_items.append({
                        "text": text,
                        "bbox": [round(v, 3) for v in mapped],
                        "conf": conf,
                        "rotation": rotation,
                        "angle": angle,
                        "ocr_mode": mode,
                        "ocr_psm": psm,
                        "raw_text": raw_text,
                    })
        if raw_items:
            merged = []
            for item in _merge_ocr_blocks(raw_items):
                normalized = dict(item)
                normalized["bbox"] = [float(v) for v in normalized["bbox"]]
                normalized["raw_text"] = normalized.get("raw_text") or normalized.get("text")
                normalized["text"] = _apply_correction(
                    normalized.get("text", ""), corrections)
                normalized["rotation"] = rotation
                normalized["angle"] = angle
                normalized["support_count"] = _support_count(normalized, raw_items)
                merged.append(normalized)
            grouped[rotation] = merged

    candidates = [item for items in grouped.values() for item in items]
    if compound_overlap_labels:
        candidates.extend(_compound_overlapping_axis_words(candidates))
    candidates = [
        item for item in candidates
        if float(item.get("conf", 0.0)) >= min_confidence
        and int(item.get("support_count") or 1) >= min_support_count
        and _looks_like_vertical_source_text(item["bbox"])
    ]
    candidates.sort(key=lambda item: (
        -int(item.get("support_count") or 1),
        -float(item.get("conf", 0.0)),
        item["bbox"][0],
        item["bbox"][1],
    ))
    return _dedupe_candidates(candidates)


def _primitive_from_candidate(
    candidate: dict[str, Any],
    *,
    text: str,
    font_size: int,
    width_scale: float,
    height_scale: float,
    index: int,
) -> dict[str, Any]:
    x0, y0, x1, y1 = [float(v) for v in candidate["bbox"]]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    visual_w = max(4.0, x1 - x0)
    visual_h = max(4.0, y1 - y0)
    geom_w = max(12.0, visual_h * width_scale)
    geom_h = max(8.0, visual_w * height_scale)
    return {
        "id": f"rotated_ocr_text_{index:04d}",
        "type": "text",
        "role": "rotated_axis_label",
        "text": text,
        "bbox": [
            round(cx - geom_w / 2.0, 3),
            round(cy - geom_h / 2.0, 3),
            round(cx + geom_w / 2.0, 3),
            round(cy + geom_h / 2.0, 3),
        ],
        "style": {
            "font_size": int(font_size),
            "bold": True,
            "align": "center",
            "rotation": float(candidate["rotation"]),
        },
        "source": "rotated_ocr_text_augment",
        "confidence": round(float(candidate.get("conf", 0.0)) / 100.0, 3),
        "alignment": {
            "rotated_ocr_bbox": candidate["bbox"],
            "ocr_angle": candidate.get("angle"),
            "ocr_psm": candidate.get("ocr_psm"),
            "ocr_mode": candidate.get("ocr_mode"),
            "support_count": candidate.get("support_count", 1),
        },
    }


def _map_bbox_from_rotated(
    bbox: Any,
    *,
    angle: int,
    width: int,
    height: int,
) -> list[float] | None:
    if not bbox or len(bbox) != 4:
        return None
    rx0, ry0, rx1, ry1 = [float(v) for v in bbox]
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    if angle == 90:
        mapped = [width - ry1, rx0, width - ry0, rx1]
    elif angle == -90:
        mapped = [ry0, height - rx1, ry1, height - rx0]
    else:
        return None
    x0, y0, x1, y1 = mapped
    x0 = max(0.0, min(float(width), x0))
    x1 = max(0.0, min(float(width), x1))
    y0 = max(0.0, min(float(height), y0))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _looks_like_vertical_source_text(bbox: list[float]) -> bool:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w = x1 - x0
    h = y1 - y0
    if w < 6 or h < 16:
        return False
    if w > 75 or h > 220:
        return False
    return h >= 1.25 * w


def _dedupe_against_existing(
    candidates: list[dict[str, Any]],
    program: dict[str, Any],
) -> list[dict[str, Any]]:
    existing = [
        (str(p.get("text", "")), p.get("bbox"))
        for p in program.get("primitives", [])
        if p.get("type") == "text" and p.get("bbox")
    ]
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        key = _norm(candidate.get("text", ""))
        bbox = candidate["bbox"]
        duplicate = False
        for text, other_bbox in existing:
            other_key = _norm(text)
            if not other_key:
                continue
            if key == other_key or (len(key) >= 5 and key in other_key) or (len(other_key) >= 5 and other_key in key):
                if _center_distance(bbox, other_bbox) < 75:
                    duplicate = True
                    break
        if not duplicate:
            out.append(candidate)
    return out


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in candidates:
        key = _norm(item.get("text", ""))
        bbox = item["bbox"]
        if any(
            _norm(other.get("text", "")) == key
            and _center_distance(bbox, other["bbox"]) < 36
            for other in out
        ):
            continue
        out.append(item)
    return out


def _compound_overlapping_axis_words(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compounds: list[dict[str, Any]] = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1:]:
            if int(left.get("rotation", 0)) != int(right.get("rotation", 0)):
                continue
            if not _axis_word_pair_geometry_ok(left["bbox"], right["bbox"]):
                continue
            left_key = _norm(left.get("text", ""))
            right_key = _norm(right.get("text", ""))
            if not left_key or not right_key or left_key == right_key:
                continue
            ordered = sorted((left, right), key=lambda item: item["bbox"][0])
            text = "\n".join(_normalize_text(item.get("text", "")) for item in ordered)
            bbox = _union_bbox(left["bbox"], right["bbox"])
            compounds.append({
                "text": text,
                "raw_text": " + ".join(str(item.get("raw_text") or item.get("text")) for item in ordered),
                "bbox": [round(v, 3) for v in bbox],
                "conf": (float(left.get("conf", 0.0)) + float(right.get("conf", 0.0))) / 2.0,
                "rotation": int(left["rotation"]),
                "angle": left.get("angle"),
                "ocr_mode": "compound",
                "ocr_psm": None,
                "support_count": int(left.get("support_count") or 1) + int(right.get("support_count") or 1),
                "compound_ids": [left.get("text"), right.get("text")],
            })
    return compounds


def _axis_word_pair_geometry_ok(a: list[float], b: list[float]) -> bool:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ah = ay1 - ay0
    bh = by1 - by0
    vertical_overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
    if vertical_overlap / max(1.0, min(ah, bh)) < 0.45:
        return False
    horizontal_gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    if horizontal_gap > 14.0:
        return False
    union = _union_bbox(a, b)
    return _looks_like_vertical_source_text(union)


def _union_bbox(a: Any, b: Any) -> list[float]:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    return [min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)]


def _support_count(item: dict[str, Any], raw_items: list[dict[str, Any]]) -> int:
    key = _norm(item.get("text", ""))
    bbox = item.get("bbox")
    if not key or not bbox:
        return 1
    support = 0
    for raw in raw_items:
        if _norm(raw.get("text", "")) != key:
            continue
        if _iou(bbox, raw.get("bbox", [0, 0, 0, 0])) > 0.24:
            support += 1
    return max(1, support)


def _preprocess(image: Image.Image, mode: str) -> Image.Image:
    if mode == "raw":
        return image
    if mode == "gray_autocontrast":
        return ImageOps.autocontrast(image.convert("L")).convert("RGB")
    if mode == "sharp":
        return image.filter(ImageFilter.SHARPEN)
    if mode == "unsharp":
        return image.filter(ImageFilter.UnsharpMask(
            radius=1.0, percent=180, threshold=3))
    raise ValueError(f"unknown mode: {mode}")


def _layout_text(text: str, layout: str) -> str:
    clean = _normalize_text(text)
    if layout == "single":
        return clean
    if layout == "stacked":
        parts = [part for part in re.split(r"\s+", clean) if part]
        if 1 < len(parts) <= 4 and max(len(part) for part in parts) <= 16:
            return "\n".join(parts)
        return ""
    return ""


def _usable_text(text: str) -> bool:
    clean = re.sub(r"[^A-Za-z0-9+/\- ]+", "", text).strip()
    compact = re.sub(r"\s+", "", clean)
    if len(compact) < 4:
        return False
    if sum(ch.isalnum() for ch in compact) / max(1, len(compact)) < 0.55:
        return False
    if re.fullmatch(r"\d+(?:[-/]\d+)?", compact):
        return False
    return True


def _normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip("|_.,;:")


def _apply_correction(value: Any, corrections: dict[str, str]) -> str:
    text = _normalize_text(value)
    key = _norm(text)
    if key in corrections:
        return corrections[key]
    return text


def _safe_name(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
    return name.lower() or "text"


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _center_distance(a: Any, b: Any) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    acx = (ax0 + ax1) / 2.0
    acy = (ay0 + ay1) / 2.0
    bcx = (bx0 + bx1) / 2.0
    bcy = (by0 + by1) / 2.0
    return math.hypot(acx - bcx, acy - bcy)


def _iou(a: Any, b: Any) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(1.0, area)


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in str(raw).split(",") if part.strip()]


def _parse_corrections(raw: str) -> dict[str, str]:
    corrections: dict[str, str] = {}
    for part in str(raw or "").split(","):
        if not part.strip() or "=" not in part:
            continue
        left, right = part.split("=", 1)
        key = _norm(left)
        value = _normalize_text(right)
        if key and value:
            corrections[key] = value
    return corrections


def _exec_evaluate(
    *,
    source_image: str,
    variants: list[Path],
    ranking_path: Path,
    best_stem: Path,
    manifest_path: Path,
    drawio_cli: str,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "evaluate_drawio_variants.py"),
        source_image,
        *[str(path) for path in variants],
        "-o",
        str(ranking_path),
        "--drawio-cli",
        drawio_cli,
        "--best-stem",
        str(best_stem),
        "--manifest",
        str(manifest_path),
        "--retry-all-null",
        "--retry-all-null-attempts",
        "3",
        "--retry-all-null-delay",
        "8",
    ]
    time.sleep(0.5)
    result = subprocess.run(cmd, check=False, text=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
