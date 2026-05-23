#!/usr/bin/env python3
"""Extract CV geometry evidence for a VLM/draw.io reconstruction agent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.cv_tools import (  # noqa: E402
    draw_cv_overlay,
    extract_cv_primitives,
    save_cv_evidence,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract line/region/junction/arrow CV evidence from an image")
    ap.add_argument("image", help="input PNG/JPG/WebP/TIFF")
    ap.add_argument("-o", "--output", default=None,
                    help="output JSON path; defaults to <image>.cv_primitives.json")
    ap.add_argument("--overlay", default=None,
                    help="debug overlay PNG path; defaults next to JSON")
    ap.add_argument("--max-lines", type=int, default=1200)
    ap.add_argument("--max-regions", type=int, default=240)
    ap.add_argument("--max-components", type=int, default=420)
    ap.add_argument("--min-line-length", type=float, default=12.0)
    ap.add_argument("--axis-angle-tolerance", type=float, default=8.0)
    ap.add_argument("--no-merge-axis-lines", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    image = Path(args.image)
    output = Path(args.output) if args.output else image.with_suffix(".cv_primitives.json")
    overlay = (
        Path(args.overlay)
        if args.overlay
        else output.with_suffix(".cv_overlay.png")
    )

    evidence = extract_cv_primitives(
        image,
        max_lines=args.max_lines,
        max_regions=args.max_regions,
        max_components=args.max_components,
        min_line_length=args.min_line_length,
        axis_angle_tolerance=args.axis_angle_tolerance,
        merge_axis_lines=not args.no_merge_axis_lines,
    )
    save_cv_evidence(evidence, output)
    if not args.no_overlay:
        draw_cv_overlay(image, evidence, overlay)

    print(json.dumps({
        "cv_primitives": str(output),
        "cv_overlay": str(overlay) if not args.no_overlay else None,
        "counts": evidence.get("counts", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
