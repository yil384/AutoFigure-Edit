#!/usr/bin/env python3
"""Sweep unsupported-edge pruning variants and rank them."""
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
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.unsupported_prune import prune_unsupported_edges  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prune source-unsupported native edges and rank variants")
    ap.add_argument("program_json")
    ap.add_argument("source_image")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="unsupported_prune_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.70,0.80")
    ap.add_argument("--modes", default="augment,all_cv")
    ap.add_argument("--mask-dilate", type=int, default=5)
    ap.add_argument("--stroke-pad", type=float, default=2.0)
    ap.add_argument("--min-length", type=float, default=12.0)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    program = load_program(args.program_json)

    variants = []
    for threshold in _parse_floats(args.thresholds):
        for mode in _parse_strings(args.modes):
            name = f"t{int(round(threshold * 100)):02d}_{mode}"
            drawio = Path(f"{base}.{name}.drawio")
            program_path = Path(f"{base}.{name}.vp_program.json")
            report_path = Path(f"{base}.{name}.prune_report.json")
            updated, report = prune_unsupported_edges(
                program,
                args.source_image,
                support_threshold=threshold,
                mode=mode,
                mask_dilate=args.mask_dilate,
                stroke_pad=args.stroke_pad,
                min_length=args.min_length,
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
        "sweep": {
            "thresholds": _parse_floats(args.thresholds),
            "modes": _parse_strings(args.modes),
            "mask_dilate": args.mask_dilate,
            "stroke_pad": args.stroke_pad,
            "min_length": args.min_length,
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

    drawios = [item["drawio"] for item in variants]
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


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in out:
            out.append(value)
    return out


def _parse_strings(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if value and value not in out:
            out.append(value)
    return out


if __name__ == "__main__":
    main()
