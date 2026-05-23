#!/usr/bin/env python3
"""Sweep low-confidence augmented line rendering parameters."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.line_render_tune import tune_augmented_line_rendering  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and rank low-confidence line rendering variants.")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("--name", default=None)
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--max-confidences", default="0.54,0.56,0.58,0.60")
    ap.add_argument("--width-scales", default="0.55,0.7,0.85")
    ap.add_argument("--strokes", default="#333333,#4f5b66,#6b7280")
    ap.add_argument("--min-width", type=float, default=0.6)
    ap.add_argument("--target-source", default="cv_line_augment")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    program_path = Path(args.program_json)
    program = load_program(program_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or program_path.stem.replace(".vp_program", "")
    base = out_dir / stem

    variants: list[Path] = []
    manifest_items: list[dict[str, Any]] = []
    for max_confidence in _parse_floats(args.max_confidences):
        for width_scale in _parse_floats(args.width_scales):
            for stroke in _parse_strs(args.strokes):
                safe_stroke = stroke.replace("#", "hex")
                name = (
                    f"{stem}.line_soft_c{max_confidence:.2f}"
                    f"_w{width_scale:.2f}_{safe_stroke}"
                )
                tuned, report = tune_augmented_line_rendering(
                    program,
                    max_confidence=max_confidence,
                    stroke=stroke,
                    width_scale=width_scale,
                    min_width=args.min_width,
                    target_source=args.target_source,
                )
                drawio = out_dir / f"{name}.drawio"
                program_out = out_dir / f"{name}.vp_program.json"
                report_out = out_dir / f"{name}.report.json"
                compile_program_to_drawio(
                    tuned,
                    drawio,
                    font_family=args.font_family,
                )
                save_program(tuned, program_out)
                report_out.write_text(json.dumps(report, indent=2, ensure_ascii=True))
                variants.append(drawio)
                manifest_items.append({
                    "name": name,
                    "drawio": str(drawio),
                    "program": str(program_out),
                    "report": str(report_out),
                })

    rows = evaluate_drawio_variants(
        args.source_image,
        variants,
        drawio_cli=args.drawio_cli,
        export=not args.no_export,
    )
    ranking_path = Path(f"{base}.line_render_ranking.json")
    ranking = {
        "source_image": args.source_image,
        "source_program": str(program_path),
        "variants": rows,
        "manifest": manifest_items,
    }
    ranking_path.write_text(json.dumps(ranking, indent=2, ensure_ascii=True))
    print(json.dumps({
        "ranking": str(ranking_path),
        "winner": rows[0]["drawio"] if rows else None,
        "scores": [compact_score_row(row) for row in rows[:18]],
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


def _parse_strs(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if value and value not in out:
            out.append(value)
    return out


if __name__ == "__main__":
    main()
