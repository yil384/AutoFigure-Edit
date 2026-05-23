#!/usr/bin/env python3
"""Derive local composition panels from CV evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.panel_regions import (  # noqa: E402
    derive_panel_regions_from_evidence,
    save_panel_regions,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Derive panel regions from a CV evidence JSON or manifest")
    ap.add_argument("input", help="CV evidence JSON or variant manifest JSON")
    ap.add_argument("-o", "--output", required=True, help="output panel JSON")
    ap.add_argument("--max-leaf-panels", type=int, default=12)
    ap.add_argument("--min-leaf-area-fraction", type=float, default=0.008)
    args = ap.parse_args()

    input_path = Path(args.input)
    payload = json.loads(input_path.read_text())
    evidence_path = payload.get("cv_evidence")
    if evidence_path:
        evidence = json.loads(Path(evidence_path).read_text())
    else:
        evidence = payload
        evidence_path = str(input_path)

    report = derive_panel_regions_from_evidence(
        evidence,
        max_leaf_panels=args.max_leaf_panels,
        min_leaf_area_fraction=args.min_leaf_area_fraction,
    )
    report["source_cv_evidence"] = evidence_path
    save_panel_regions(report, args.output)
    print(json.dumps({
        "panel_regions": str(args.output),
        "source_cv_evidence": evidence_path,
        "counts": report.get("counts", {}),
        "regions": report.get("panel_regions", []),
    }, indent=2))


if __name__ == "__main__":
    main()
