#!/usr/bin/env python3
"""Apply residual-vector augmentation to a visual primitive program."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import validate_pure_native_drawio  # noqa: E402
from visual_primitives.residual_augment import (  # noqa: E402
    augment_program_from_render_residual,
)
from visual_primitives.schema import load_program, save_program  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add missing source/render residuals as native primitives")
    ap.add_argument("program_json")
    ap.add_argument("source_image")
    ap.add_argument("rendered_image")
    ap.add_argument("-o", "--output-drawio", required=True)
    ap.add_argument("--output-program", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--max-paths", type=int, default=80)
    ap.add_argument("--max-shapes", type=int, default=60)
    ap.add_argument("--render-dilate", type=int, default=4)
    ap.add_argument("--min-area", type=int, default=10)
    ap.add_argument("--max-bbox-area", type=int, default=2600)
    ap.add_argument("--text-pad", type=float, default=2.5)
    args = ap.parse_args()

    program = load_program(args.program_json)
    updated, report = augment_program_from_render_residual(
        program,
        args.source_image,
        args.rendered_image,
        max_paths=args.max_paths,
        max_shapes=args.max_shapes,
        render_dilate=args.render_dilate,
        min_area=args.min_area,
        max_bbox_area=args.max_bbox_area,
        text_pad=args.text_pad,
    )
    output_drawio = Path(args.output_drawio)
    output_program = (
        Path(args.output_program)
        if args.output_program
        else output_drawio.with_suffix(".vp_program.json")
    )
    report_path = (
        Path(args.report)
        if args.report
        else output_drawio.with_suffix(".residual_report.json")
    )
    save_program(updated, output_program)
    compile_stats = compile_program_to_drawio(
        updated,
        output_drawio,
        font_family=args.font_family,
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(json.dumps({
        "drawio": str(output_drawio),
        "program": str(output_program),
        "report": str(report_path),
        "compile": compile_stats,
        "native_purity": validate_pure_native_drawio(output_drawio),
        "counts": updated.get("counts", {}),
        "residual_counts": report.get("counts", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
