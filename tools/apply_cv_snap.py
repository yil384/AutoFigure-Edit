#!/usr/bin/env python3
"""Apply CV geometry evidence to a visual primitive program and emit draw.io."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.cv_snap import snap_program_to_cv_evidence  # noqa: E402
from visual_primitives.cv_augment import augment_program_from_cv_evidence  # noqa: E402
from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import validate_pure_native_drawio  # noqa: E402
from visual_primitives.schema import save_program  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Snap a visual primitive program to CV evidence")
    ap.add_argument("program", help="input .vp_program.json")
    ap.add_argument("cv_evidence", help="input .cv_primitives.json")
    ap.add_argument("-o", "--output", default=None,
                    help="output .drawio path")
    ap.add_argument("--program-output", default=None,
                    help="snapped program JSON path")
    ap.add_argument("--report", default=None,
                    help="snap report JSON path")
    ap.add_argument("--font-family", default="Arial")
    ap.add_argument("--no-regions", action="store_true")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--region-min-score", type=float, default=0.64)
    ap.add_argument("--text-min-score", type=float, default=0.62)
    ap.add_argument("--edge-min-score", type=float, default=0.68)
    ap.add_argument("--guide-tolerance", type=float, default=5.0)
    ap.add_argument("--augment", action="store_true",
                    help="add missing high-confidence CV lines as native primitives")
    ap.add_argument("--augment-text", action="store_true",
                    help="also add missing high-confidence CV text")
    ap.add_argument("--augment-regions", action="store_true",
                    help="also add missing high-confidence CV regions")
    ap.add_argument("--max-augment-text", type=int, default=45)
    ap.add_argument("--max-augment-lines", type=int, default=70)
    ap.add_argument("--min-augment-text-confidence", type=float, default=0.58)
    args = ap.parse_args()

    program_path = Path(args.program)
    evidence_path = Path(args.cv_evidence)
    program = json.loads(program_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    snapped, report = snap_program_to_cv_evidence(
        program,
        evidence,
        snap_regions=not args.no_regions,
        snap_text=not args.no_text,
        snap_edges=not args.no_edges,
        region_min_score=args.region_min_score,
        text_min_score=args.text_min_score,
        edge_min_score=args.edge_min_score,
        guide_tolerance=args.guide_tolerance,
    )
    augment_report = None
    if args.augment:
        snapped, augment_report = augment_program_from_cv_evidence(
            snapped,
            evidence,
            add_text=args.augment_text,
            add_regions=args.augment_regions,
            max_text=args.max_augment_text,
            max_lines=args.max_augment_lines,
            min_text_confidence=args.min_augment_text_confidence,
        )

    drawio_path = (
        Path(args.output)
        if args.output
        else program_path.with_suffix(".cv_snapped.drawio")
    )
    snapped_program_path = (
        Path(args.program_output)
        if args.program_output
        else drawio_path.with_suffix(".vp_program.json")
    )
    report_path = (
        Path(args.report)
        if args.report
        else drawio_path.with_suffix(".cv_snap_report.json")
    )

    save_program(snapped, snapped_program_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    combined_report = {"snap": report}
    if augment_report:
        combined_report["augment"] = augment_report
    report_path.write_text(json.dumps(combined_report, indent=2, ensure_ascii=True))
    compile_stats = compile_program_to_drawio(
        snapped, drawio_path, font_family=args.font_family)
    pure_check = validate_pure_native_drawio(drawio_path)
    print(json.dumps({
        "snapped_program": str(snapped_program_path),
        "drawio": str(drawio_path),
        "report": str(report_path),
        "compile": compile_stats,
        "native_purity": pure_check,
        "snap_counts": report.get("counts", {}),
        "augment_counts": (augment_report or {}).get("counts"),
        "program_counts": snapped.get("counts", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
