#!/usr/bin/env python3
"""Create rectangular panel-region JSON for local native-primitive composition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate grid panel regions for local variant evaluation")
    ap.add_argument("source_image")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--prefix", default="grid")
    args = ap.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")

    image = Image.open(args.source_image)
    width, height = image.size
    regions = []
    for row in range(args.rows):
        for col in range(args.cols):
            x0 = width * col / args.cols
            x1 = width * (col + 1) / args.cols
            y0 = height * row / args.rows
            y1 = height * (row + 1) / args.rows
            regions.append({
                "id": f"{args.prefix}_r{row + 1:02d}_c{col + 1:02d}",
                "bbox": [
                    round(x0, 3),
                    round(y0, 3),
                    round(x1, 3),
                    round(y1, 3),
                ],
                "kind": "grid_region",
                "row": row + 1,
                "col": col + 1,
            })

    payload = {
        "version": "grid-regions-0.1",
        "canvas": {"width": width, "height": height},
        "rows": args.rows,
        "cols": args.cols,
        "panel_regions": regions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    print(json.dumps({
        "output": str(output),
        "rows": args.rows,
        "cols": args.cols,
        "regions": len(regions),
    }, indent=2))


if __name__ == "__main__":
    main()
