#!/usr/bin/env python3
"""Apply a VLM JSON patch to a visual primitive program and emit draw.io."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio
from visual_primitives.patches import apply_patch_plan, load_patch
from visual_primitives.qa import validate_pure_native_drawio
from visual_primitives.schema import save_program


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply VLM primitive patch JSON and compile to draw.io")
    ap.add_argument("program", help="input .vp_program.json")
    ap.add_argument("patch", help="VLM patch JSON")
    ap.add_argument("-o", "--output", default=None,
                    help="output drawio path")
    ap.add_argument("--program-output", default=None,
                    help="patched program JSON path")
    ap.add_argument("--font-family", default="Helvetica")
    args = ap.parse_args()

    program_path = Path(args.program)
    patch_path = Path(args.patch)
    program = json.loads(program_path.read_text())
    patch_doc = load_patch(patch_path)
    patched = apply_patch_plan(program, patch_doc)

    drawio_path = Path(args.output) if args.output else program_path.with_suffix(".patched.drawio")
    patched_program_path = (
        Path(args.program_output)
        if args.program_output
        else drawio_path.with_suffix(".vp_program.json")
    )
    save_program(patched, patched_program_path)
    compile_stats = compile_program_to_drawio(
        patched,
        drawio_path,
        font_family=args.font_family,
    )
    pure_check = validate_pure_native_drawio(drawio_path)
    print(json.dumps({
        "patched_program": str(patched_program_path),
        "drawio": str(drawio_path),
        "compile": compile_stats,
        "native_purity": pure_check,
        "counts": patched.get("counts", {}),
    }, indent=2))


if __name__ == "__main__":
    main()
