#!/usr/bin/env python3
"""Slow but stable draw.io variant evaluation wrapper."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.variant_eval import compact_score_row, evaluate_drawio_variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate draw.io variants one at a time with cooldowns")
    ap.add_argument("source_image")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--best-stem", default=None)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for variant in args.variants:
        row = _evaluate_one(
            args.source_image,
            Path(variant),
            attempts=args.attempts,
            cooldown=args.cooldown,
        )
        rows.append(row)
    rows.sort(key=lambda item: item["score"], reverse=True)

    report = {
        "source_image": args.source_image,
        "winner": rows[0]["drawio"] if rows else None,
        "variants": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True))

    best_drawio = best_png = best_compare = None
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
        "winner": report["winner"],
        "best_drawio": str(best_drawio) if best_drawio else None,
        "best_png": str(best_png) if best_png and best_png.exists() else None,
        "best_compare": str(best_compare) if best_compare and best_compare.exists() else None,
        "scores": [compact_score_row(row) for row in rows],
    }, indent=2))


def _evaluate_one(
    source_image: str,
    variant: Path,
    *,
    attempts: int,
    cooldown: float,
) -> dict:
    last_row = None
    for attempt in range(max(1, attempts)):
        _cleanup_drawio()
        time.sleep(cooldown)
        rows = evaluate_drawio_variants(
            source_image,
            [variant],
            export=True,
        )
        last_row = rows[0]
        if last_row.get("metrics"):
            return last_row
        time.sleep(cooldown)
    return last_row


def _cleanup_drawio() -> None:
    try:
        subprocess.run(
            ["pkill", "-f", "draw.io"],
            text=True,
            capture_output=True,
            timeout=3,
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
