#!/usr/bin/env python3
"""Sweep text rendering parameters for a visual primitive program."""
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
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.text_render_tune import tune_text_rendering  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate and rank generic text-rendering variants.")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("--name", default=None)
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--font-scales", default="1.0,1.04,1.08,1.12")
    ap.add_argument("--bbox-pads", default="0,1,2")
    ap.add_argument("--bold-modes", default="none,headers")
    ap.add_argument("--font-families", default="Arial,Helvetica")
    ap.add_argument("--max-font-size", type=int, default=16)
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
    for font_family in _parse_strs(args.font_families):
        safe_family = _safe_name(font_family)
        for bold_mode in _parse_strs(args.bold_modes):
            for font_scale in _parse_floats(args.font_scales):
                for bbox_pad in _parse_floats(args.bbox_pads):
                    name = (
                        f"{stem}.font_{safe_family}"
                        f"_b{bold_mode}_s{font_scale:.2f}_p{bbox_pad:g}"
                    )
                    tuned, report = tune_text_rendering(
                        program,
                        font_scale=font_scale,
                        bbox_pad=bbox_pad,
                        max_font_size=args.max_font_size,
                        bold_mode=bold_mode,
                    )
                    drawio = out_dir / f"{name}.drawio"
                    program_out = out_dir / f"{name}.vp_program.json"
                    report_out = out_dir / f"{name}.report.json"
                    compile_program_to_drawio(
                        tuned,
                        drawio,
                        font_family=font_family,
                    )
                    save_program(tuned, program_out)
                    report_out.write_text(json.dumps(
                        {"font_family": font_family, "tune": report},
                        indent=2,
                        ensure_ascii=True,
                    ))
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
    ranking_path = Path(f"{base}.text_render_ranking.json")
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
        "scores": [compact_score_row(row) for row in rows[:16]],
    }, indent=2))


def _parse_floats(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in values:
            values.append(value)
    return values


def _parse_strs(raw: str) -> list[str]:
    values: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    return values


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "font"


if __name__ == "__main__":
    main()
