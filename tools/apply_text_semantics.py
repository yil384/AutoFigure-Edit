#!/usr/bin/env python3
"""Apply lexicon text corrections to a visual primitive program."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.text_semantics import apply_text_replacements  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply semantic text corrections.")
    ap.add_argument("program_json")
    ap.add_argument("-o", "--output-drawio", required=True)
    ap.add_argument("--output-program", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--font-family", default="Helvetica")
    args = ap.parse_args()

    program = load_program(args.program_json)
    updated, report = apply_text_replacements(program)
    output_drawio = Path(args.output_drawio)
    output_program = (
        Path(args.output_program)
        if args.output_program
        else output_drawio.with_suffix(".vp_program.json")
    )
    report_path = (
        Path(args.report)
        if args.report
        else output_drawio.with_suffix(".semantic_report.json")
    )
    compile_program_to_drawio(updated, output_drawio, font_family=args.font_family)
    save_program(updated, output_program)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(json.dumps({
        "drawio": str(output_drawio),
        "program": str(output_program),
        "report": str(report_path),
        "counts": report["counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
