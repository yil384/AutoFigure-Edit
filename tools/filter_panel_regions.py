#!/usr/bin/env python3
"""Filter and sort panel-region JSON files for local composition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter panel regions by kind/area")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--kinds", default="leaf_region",
                    help="comma-separated allowed region kinds")
    ap.add_argument("--min-area", type=float, default=0.0)
    ap.add_argument("--max-area", type=float, default=None)
    ap.add_argument("--sort", choices=["area-asc", "area-desc", "input"],
                    default="area-asc")
    args = ap.parse_args()

    payload = json.loads(Path(args.input).read_text())
    allowed = {value.strip() for value in args.kinds.split(",") if value.strip()}
    regions = []
    for region in payload.get("panel_regions") or payload.get("regions") or []:
        if allowed and region.get("kind") not in allowed:
            continue
        area = float(region.get("area") or _bbox_area(region.get("bbox")) or 0.0)
        if area < args.min_area:
            continue
        if args.max_area is not None and area > args.max_area:
            continue
        item = dict(region)
        item["area"] = round(area, 3)
        regions.append(item)

    if args.sort == "area-asc":
        regions.sort(key=lambda item: (float(item.get("area") or 0.0), item.get("id", "")))
    elif args.sort == "area-desc":
        regions.sort(key=lambda item: (-float(item.get("area") or 0.0), item.get("id", "")))

    out = {
        "version": "panel-regions-filtered-0.1",
        "source": str(args.input),
        "canvas": payload.get("canvas"),
        "panel_regions": regions,
        "counts": {
            "total": len(regions),
            "kinds": {
                kind: sum(1 for region in regions if region.get("kind") == kind)
                for kind in sorted({region.get("kind") for region in regions})
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=True))
    print(json.dumps({
        "output": str(output),
        "regions": len(regions),
        "kinds": out["counts"]["kinds"],
    }, indent=2))


def _bbox_area(bbox: Any) -> float | None:
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


if __name__ == "__main__":
    main()
