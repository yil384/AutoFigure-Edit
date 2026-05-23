#!/usr/bin/env python3
"""Generate/evaluate region-local primitive pruning variants."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.panel_regions import load_panel_regions  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import compact_score_row, evaluate_drawio_variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep deleting primitives by residual region/source/type")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("regions_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="region_prune_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--region-ids", default=None,
                    help="comma-separated region ids; defaults to first 8")
    ap.add_argument("--max-regions", type=int, default=8)
    ap.add_argument("--types", default="edge,shape")
    ap.add_argument("--source-contains", default=(
        "residual_path_augment,cv_line_augment,short_stroke,tiny_mark,"
        "cv_native_shape,cv_line"
    ))
    ap.add_argument("--mode", choices=["center", "intersect", "contained"],
                    default="intersect")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--min-delete", type=int, default=1)
    ap.add_argument("--include-region-all", action="store_true",
                    help="also generate one variant per region deleting all selected sources")
    ap.add_argument("--include-global-all", action="store_true",
                    help="also generate one variant deleting all selected sources in all chosen regions")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    program = load_program(args.program_json)
    regions = _select_regions(
        load_panel_regions(args.regions_json),
        args.region_ids,
        args.max_regions,
    )
    allowed_types = _csv_set(args.types)
    source_needles = _csv_list(args.source_contains)

    variants = []
    for region in regions:
        for needle in source_needles:
            doomed = _matching_ids(
                program,
                region,
                allowed_types,
                [needle],
                mode=args.mode,
                margin=args.margin,
            )
            if len(doomed) < args.min_delete:
                continue
            variants.append(_write_variant(
                program,
                base,
                f"{_safe(region['id'])}_{_safe(needle)}",
                doomed,
                args.font_family,
                {
                    "operation": "region_source_prune",
                    "region": region,
                    "source_contains": [needle],
                    "deleted_ids": doomed,
                },
            ))
        if args.include_region_all:
            doomed = _matching_ids(
                program,
                region,
                allowed_types,
                source_needles,
                mode=args.mode,
                margin=args.margin,
            )
            if len(doomed) >= args.min_delete:
                variants.append(_write_variant(
                    program,
                    base,
                    f"{_safe(region['id'])}_all_sources",
                    doomed,
                    args.font_family,
                    {
                        "operation": "region_all_source_prune",
                        "region": region,
                        "source_contains": source_needles,
                        "deleted_ids": doomed,
                    },
                ))

    if args.include_global_all:
        doomed_set: set[str] = set()
        for region in regions:
            doomed_set.update(_matching_ids(
                program,
                region,
                allowed_types,
                source_needles,
                mode=args.mode,
                margin=args.margin,
            ))
        doomed = sorted(doomed_set)
        if len(doomed) >= args.min_delete:
            variants.append(_write_variant(
                program,
                base,
                "global_all_sources",
                doomed,
                args.font_family,
                {
                    "operation": "global_all_source_prune",
                    "regions": regions,
                    "source_contains": source_needles,
                    "deleted_ids": doomed,
                },
            ))

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "regions_json": args.regions_json,
        "regions": regions,
        "types": sorted(allowed_types),
        "source_contains": source_needles,
        "mode": args.mode,
        "margin": args.margin,
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


def _select_regions(regions: list[dict[str, Any]], raw_ids: str | None,
                    max_regions: int) -> list[dict[str, Any]]:
    if raw_ids:
        wanted = {item.strip() for item in raw_ids.split(",") if item.strip()}
        return [region for region in regions if str(region.get("id")) in wanted]
    return regions[:max_regions]


def _matching_ids(
    program: dict[str, Any],
    region: dict[str, Any],
    allowed_types: set[str],
    source_needles: list[str],
    *,
    mode: str,
    margin: float,
) -> list[str]:
    box = tuple(float(value) for value in region["bbox"])
    out = []
    for primitive in program.get("primitives", []):
        primitive_id = primitive.get("id")
        if not primitive_id:
            continue
        if primitive.get("type") not in allowed_types:
            continue
        source = str(primitive.get("source") or "")
        if not any(needle in source for needle in source_needles):
            continue
        bbox = _primitive_bbox(primitive)
        if not bbox:
            continue
        if _bbox_matches(bbox, box, mode=mode, margin=margin):
            out.append(str(primitive_id))
    return sorted(out)


def _write_variant(
    program: dict[str, Any],
    base: Path,
    tag: str,
    doomed: list[str],
    font_family: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(program)
    doomed_set = set(doomed)
    updated["primitives"] = [
        primitive for primitive in updated.get("primitives", [])
        if primitive.get("id") not in doomed_set
    ]
    _refresh_counts(updated)
    tag = _safe(tag)
    drawio = Path(f"{base}.{tag}.drawio")
    program_path = Path(f"{base}.{tag}.vp_program.json")
    report_path = Path(f"{base}.{tag}.report.json")
    save_program(updated, program_path)
    compile_program_to_drawio(updated, drawio, font_family=font_family)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    return {
        "name": tag,
        "drawio": str(drawio),
        "program": str(program_path),
        "report": str(report_path),
        "deleted": len(doomed),
    }


def _primitive_bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = primitive.get("bbox")
    if bbox and len(bbox) == 4:
        return tuple(float(value) for value in bbox)
    path = primitive.get("path") or []
    if not path:
        return None
    xs = [float(point[0]) for point in path]
    ys = [float(point[1]) for point in path]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_matches(
    bbox: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    *,
    mode: str,
    margin: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = box
    bx0 -= margin
    by0 -= margin
    bx1 += margin
    by1 += margin
    if mode == "intersect":
        return not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1)
    if mode == "contained":
        return x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }


def _csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _csv_list(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def _safe(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()[:80] or "variant"


if __name__ == "__main__":
    main()
