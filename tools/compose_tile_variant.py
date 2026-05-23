#!/usr/bin/env python3
"""Compose a draw.io program from tile-level variant winners."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import validate_pure_native_drawio  # noqa: E402
from visual_primitives.tile_compose import (  # noqa: E402
    compose_program_from_tile_winners,
    save_composed_program,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compose pure-native draw.io from tile winners")
    ap.add_argument("manifest", help="variant manifest JSON")
    ap.add_argument("ranking", help="variant ranking JSON with tile_winners")
    ap.add_argument("-o", "--output", required=True, help="output drawio path")
    ap.add_argument("--program-output", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--font-family", default="Arial",
                    help="font family used when compiling the composed drawio")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    ranking_path = Path(args.ranking)
    manifest = json.loads(manifest_path.read_text())
    ranking = json.loads(ranking_path.read_text())
    program, report = compose_program_from_tile_winners(manifest, ranking)

    output = Path(args.output)
    program_output = (
        Path(args.program_output)
        if args.program_output
        else output.with_suffix(".vp_program.json")
    )
    report_output = (
        Path(args.report)
        if args.report
        else output.with_suffix(".tile_compose_report.json")
    )
    save_composed_program(program, program_output)
    compile_stats = compile_program_to_drawio(
        program,
        output,
        font_family=args.font_family,
    )
    report_output.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    pure = validate_pure_native_drawio(output)
    print(json.dumps({
        "drawio": str(output),
        "program": str(program_output),
        "report": str(report_output),
        "compile": compile_stats,
        "native_purity": pure,
        "counts": program.get("counts", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
