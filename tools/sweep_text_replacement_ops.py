#!/usr/bin/env python3
"""Generate/evaluate one-at-a-time native text replacement variants."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.text_semantics import DEFAULT_REPLACEMENTS  # noqa: E402
from visual_primitives.variant_eval import compact_score_row, evaluate_drawio_variants  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep exact text replacements one primitive at a time")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="text_replacement_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--replacements-json", default=None,
                    help="optional JSON object of exact old->new replacements")
    ap.add_argument("--include-all", action="store_true",
                    help="also generate one variant with every replacement applied")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    program = load_program(args.program_json)
    replacements = _load_replacements(args.replacements_json)
    operations = _find_replacement_operations(program, replacements)
    variants = []
    base = out_dir / args.name

    for index, operation in enumerate(operations, start=1):
        updated = copy.deepcopy(program)
        primitive = _primitive_by_id(updated, operation["primitive_id"])
        if primitive is None:
            continue
        primitive["text"] = operation["after"]
        primitive["source"] = str(primitive.get("source", "text")) + "+text_repair"
        tag = f"op{index:03d}_{_safe_name(operation['old'])}_to_{_safe_name(operation['new'])}"
        variants.append(_write_variant(
            updated,
            base,
            tag,
            args.font_family,
            {
                "mode": "single",
                "operation": operation,
            },
        ))

    if args.include_all and operations:
        updated = copy.deepcopy(program)
        applied = []
        for operation in operations:
            primitive = _primitive_by_id(updated, operation["primitive_id"])
            if primitive is None:
                continue
            primitive["text"] = operation["after"]
            primitive["source"] = str(primitive.get("source", "text")) + "+text_repair"
            applied.append(operation)
        variants.append(_write_variant(
            updated,
            base,
            "all_replacements",
            args.font_family,
            {
                "mode": "all",
                "operations": applied,
            },
        ))

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "replacements": replacements,
        "operations": operations,
        "variants": variants,
    }, indent=2, ensure_ascii=True))
    if args.generate_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "variants": [item["drawio"] for item in variants],
        }, indent=2))
        return

    drawios = [Path(item["drawio"]) for item in variants]
    if args.baseline_drawio:
        drawios.insert(0, Path(args.baseline_drawio))
    rows = evaluate_drawio_variants(
        args.source_image,
        drawios,
        drawio_cli=args.drawio_cli,
        export=True,
    )
    ranking_path = Path(f"{base}.ranking.json")
    ranking_path.write_text(json.dumps({
        "source_image": args.source_image,
        "winner": rows[0]["drawio"] if rows else None,
        "variants": rows,
        "manifest": str(manifest_path),
    }, indent=2, ensure_ascii=True))
    print(json.dumps({
        "manifest": str(manifest_path),
        "ranking": str(ranking_path),
        "winner": rows[0]["drawio"] if rows else None,
        "scores": [compact_score_row(row) for row in rows[:20]],
    }, indent=2))


def _load_replacements(path: str | None) -> dict[str, str]:
    if not path:
        return dict(DEFAULT_REPLACEMENTS)
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("replacements JSON must be an object")
    return {str(key): str(value) for key, value in data.items()}


def _find_replacement_operations(
    program: dict[str, Any],
    replacements: dict[str, str],
) -> list[dict[str, Any]]:
    operations = []
    for primitive in program.get("primitives", []):
        if primitive.get("type") != "text":
            continue
        before = str(primitive.get("text", ""))
        after = before
        used = []
        for old, new in replacements.items():
            if old in new and new in after:
                continue
            if old not in after:
                continue
            after = after.replace(old, new)
            used.append({"old": old, "new": new})
        if after == before:
            continue
        operations.append({
            "primitive_id": primitive.get("id"),
            "before": before,
            "after": after,
            "bbox": primitive.get("bbox"),
            "old": "+".join(item["old"] for item in used),
            "new": "+".join(item["new"] for item in used),
        })
    return operations


def _write_variant(
    program: dict[str, Any],
    base: Path,
    tag: str,
    font_family: str,
    report: dict[str, Any],
) -> dict[str, str]:
    drawio = Path(f"{base}.{tag}.drawio")
    program_path = Path(f"{base}.{tag}.vp_program.json")
    report_path = Path(f"{base}.{tag}.report.json")
    save_program(program, program_path)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    compile_program_to_drawio(program, drawio, font_family=font_family)
    return {
        "name": tag,
        "drawio": str(drawio),
        "program": str(program_path),
        "report": str(report_path),
    }


def _primitive_by_id(program: dict[str, Any], primitive_id: str) -> dict[str, Any] | None:
    for primitive in program.get("primitives", []):
        if primitive.get("id") == primitive_id:
            return primitive
    return None


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")[:42] or "text"


if __name__ == "__main__":
    main()
