#!/usr/bin/env python3
"""Sweep residual-vector augmentation variants and rank them."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.residual_augment import (  # noqa: E402
    augment_program_from_render_residual,
)
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep source/render residual augmentation variants")
    ap.add_argument("program_json")
    ap.add_argument("source_image")
    ap.add_argument("rendered_image")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="residual_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--render-dilates", default="4,5,6")
    ap.add_argument("--max-shapes-values", default="0,8,16,32")
    ap.add_argument("--max-paths-values", default="20,50")
    ap.add_argument("--min-area", type=int, default=12)
    ap.add_argument("--max-bbox-area", type=int, default=2600)
    ap.add_argument("--text-pad", type=float, default=4.0)
    ap.add_argument("--include-skeleton-paths", action="store_true",
                    help="emit centerline polylines for missing thin residual strokes")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    program = load_program(args.program_json)

    variants = []
    for render_dilate in _parse_ints(args.render_dilates):
        for max_shapes in _parse_ints(args.max_shapes_values):
            for max_paths in _parse_ints(args.max_paths_values):
                name = f"d{render_dilate}_s{max_shapes}_p{max_paths}"
                drawio = Path(f"{base}.{name}.drawio")
                program_path = Path(f"{base}.{name}.vp_program.json")
                report_path = Path(f"{base}.{name}.residual_report.json")
                updated, report = augment_program_from_render_residual(
                    program,
                    args.source_image,
                    args.rendered_image,
                    max_paths=max_paths,
                    max_shapes=max_shapes,
                    render_dilate=render_dilate,
                    min_area=args.min_area,
                    max_bbox_area=args.max_bbox_area,
                    text_pad=args.text_pad,
                    include_skeleton_paths=args.include_skeleton_paths,
                )
                save_program(updated, program_path)
                report_path.write_text(json.dumps(report, indent=2))
                compile_program_to_drawio(
                    updated,
                    drawio,
                    font_family=args.font_family,
                )
                variants.append({
                    "name": name,
                    "drawio": drawio,
                    "program": program_path,
                    "report": report_path,
                    "counts": report.get("counts", {}),
                })

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "rendered_image": args.rendered_image,
        "sweep": {
            "render_dilates": _parse_ints(args.render_dilates),
            "max_shapes_values": _parse_ints(args.max_shapes_values),
            "max_paths_values": _parse_ints(args.max_paths_values),
            "min_area": args.min_area,
            "max_bbox_area": args.max_bbox_area,
            "text_pad": args.text_pad,
            "include_skeleton_paths": args.include_skeleton_paths,
            "font_family": args.font_family,
        },
        "variants": [
            {
                "name": item["name"],
                "drawio": str(item["drawio"]),
                "program": str(item["program"]),
                "report": str(item["report"]),
                "counts": item["counts"],
            }
            for item in variants
        ],
    }, indent=2))

    if args.generate_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "variants": [str(item["drawio"]) for item in variants],
        }, indent=2))
        return

    rows = evaluate_drawio_variants(
        args.source_image,
        [item["drawio"] for item in variants],
        drawio_cli=args.drawio_cli,
        export=True,
    )
    ranking_path = Path(f"{base}.ranking.json")
    ranking_path.write_text(json.dumps({
        "source_image": args.source_image,
        "winner": rows[0]["drawio"] if rows else None,
        "variants": rows,
    }, indent=2))
    best_drawio = Path(f"{base}.best.drawio")
    best_png = Path(str(best_drawio) + ".png")
    best_compare = best_drawio.with_suffix(best_drawio.suffix + ".compare.png")
    if rows:
        shutil.copyfile(rows[0]["drawio"], best_drawio)
        if rows[0].get("rendered_png"):
            shutil.copyfile(rows[0]["rendered_png"], best_png)
        if rows[0].get("compare_png"):
            shutil.copyfile(rows[0]["compare_png"], best_compare)
    print(json.dumps({
        "best_drawio": str(best_drawio) if rows else None,
        "best_png": str(best_png) if best_png.exists() else None,
        "best_compare": str(best_compare) if best_compare.exists() else None,
        "manifest": str(manifest_path),
        "ranking": str(ranking_path),
        "scores": [compact_score_row(row) for row in rows],
    }, indent=2))


def _parse_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in out:
            out.append(value)
    return out


if __name__ == "__main__":
    main()
