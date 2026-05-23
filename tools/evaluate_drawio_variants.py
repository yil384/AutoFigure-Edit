#!/usr/bin/env python3
"""Export and rank pure-native draw.io reconstruction variants."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.qa import (  # noqa: E402
    DEFAULT_DRAWIO_CLI,
)
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
    panel_winners,
    tile_winners,
)
from visual_primitives.panel_regions import load_panel_regions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate draw.io variants by render metrics and purity")
    ap.add_argument("source_image")
    ap.add_argument("variants", nargs="+", help="candidate .drawio files")
    ap.add_argument("-o", "--output", default=None,
                    help="output ranking JSON")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--no-export", action="store_true",
                    help="reuse existing <drawio>.png files")
    ap.add_argument("--tiles", action="store_true",
                    help="also compute quadrant-level scores and winners")
    ap.add_argument("--panel-regions", default=None,
                    help="panel region JSON for local panel-level scores")
    ap.add_argument("--best-stem", default=None,
                    help="copy winner artifacts to <best-stem>.drawio/.png/.compare.png")
    ap.add_argument("--manifest", default=None,
                    help="optional manifest path to echo in stdout JSON")
    ap.add_argument("--retry-all-null", action="store_true",
                    help="if every export fails, wait and re-run the full evaluation")
    ap.add_argument("--retry-all-null-attempts", type=int, default=2)
    ap.add_argument("--retry-all-null-delay", type=float, default=8.0)
    args = ap.parse_args()

    source = Path(args.source_image)
    panel_regions = (
        load_panel_regions(args.panel_regions)
        if args.panel_regions else None
    )
    rows = _evaluate_with_optional_retry(
        source=source,
        variants=[Path(v) for v in args.variants],
        drawio_cli=args.drawio_cli,
        export=not args.no_export,
        include_tiles=args.tiles,
        panel_regions=panel_regions,
        retry_all_null=args.retry_all_null,
        retry_attempts=args.retry_all_null_attempts,
        retry_delay=args.retry_all_null_delay,
    )
    report = {
        "source_image": str(source),
        "winner": rows[0]["drawio"] if rows else None,
        "tile_winners": tile_winners(rows) if args.tiles else {},
        "panel_regions": panel_regions or [],
        "panel_winners": panel_winners(rows) if panel_regions else {},
        "variants": rows,
    }
    output = Path(args.output) if args.output else Path("outputs/visual_primitives/variant_ranking.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    best_drawio = None
    best_png = None
    best_compare = None
    if args.best_stem and rows:
        best_drawio = Path(f"{args.best_stem}.drawio")
        best_png = Path(f"{args.best_stem}.drawio.png")
        best_compare = Path(f"{args.best_stem}.drawio.compare.png")
        best_drawio.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rows[0]["drawio"], best_drawio)
        if rows[0].get("rendered_png"):
            shutil.copyfile(rows[0]["rendered_png"], best_png)
        if rows[0].get("compare_png"):
            shutil.copyfile(rows[0]["compare_png"], best_compare)
    print(json.dumps({
        "ranking": str(output),
        "manifest": args.manifest,
        "best_drawio": str(best_drawio) if best_drawio else None,
        "best_png": str(best_png) if best_png and best_png.exists() else None,
        "best_compare": str(best_compare) if best_compare and best_compare.exists() else None,
        "winner": report["winner"],
        "tile_winners": report["tile_winners"],
        "panel_winners": report["panel_winners"],
        "scores": [compact_score_row(row) for row in rows],
    }, indent=2))


def _evaluate_with_optional_retry(
    *,
    source: Path,
    variants: list[Path],
    drawio_cli: str,
    export: bool,
    include_tiles: bool,
    panel_regions,
    retry_all_null: bool,
    retry_attempts: int,
    retry_delay: float,
):
    attempts = max(1, retry_attempts if retry_all_null else 1)
    rows = []
    for attempt in range(1, attempts + 1):
        rows = evaluate_drawio_variants(
            source,
            variants,
            drawio_cli=drawio_cli,
            export=export,
            include_tiles=include_tiles,
            panel_regions=panel_regions,
        )
        if not rows or any(row.get("metrics") is not None for row in rows):
            return rows
        if attempt < attempts:
            _cleanup_drawio_process()
            time.sleep(retry_delay)
    return rows


def _cleanup_drawio_process() -> None:
    try:
        import subprocess
        subprocess.run(
            ["pkill", "-f", "draw.io"],
            text=True,
            capture_output=True,
            timeout=2,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
