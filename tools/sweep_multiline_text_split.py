#!/usr/bin/env python3
"""Sweep pure-native split variants for multi-line draw.io text primitives."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import compact_score_row, evaluate_drawio_variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split multi-line native text cells into per-line text cells")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="multiline_text_split")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--text-ids", default="all")
    ap.add_argument("--font-deltas", default="0,1,2")
    ap.add_argument("--line-gaps", default="-2,0,2,4")
    ap.add_argument("--include-combo", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    program = load_program(args.program_json)
    selected_ids = _selected_ids(program, args.text_ids)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    variants = []

    for text_id in selected_ids:
        for font_delta in _ints(args.font_deltas):
            for line_gap in _floats(args.line_gaps):
                updated, report = split_multiline_text(
                    program,
                    {text_id},
                    font_delta=font_delta,
                    line_gap=line_gap,
                )
                if report["counts"]["split"] <= 0:
                    continue
                variants.append(_write_variant(
                    updated,
                    base,
                    f"{text_id}_fd{font_delta:+d}_gap{line_gap:g}",
                    args.font_family,
                    report,
                ))

    if args.include_combo and len(selected_ids) > 1:
        for font_delta in _ints(args.font_deltas):
            for line_gap in _floats(args.line_gaps):
                updated, report = split_multiline_text(
                    program,
                    set(selected_ids),
                    font_delta=font_delta,
                    line_gap=line_gap,
                )
                if report["counts"]["split"] <= 0:
                    continue
                variants.append(_write_variant(
                    updated,
                    base,
                    f"combo_fd{font_delta:+d}_gap{line_gap:g}",
                    args.font_family,
                    report,
                ))

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "selected_ids": selected_ids,
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
        "scores": [compact_score_row(row) for row in rows[:24]],
    }, indent=2))


def split_multiline_text(
    program: dict[str, Any],
    text_ids: set[str],
    *,
    font_delta: int,
    line_gap: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = copy.deepcopy(program)
    new_primitives = []
    operations = []
    used_ids = {p.get("id") for p in updated.get("primitives", [])}
    for primitive in updated.get("primitives", []):
        if primitive.get("type") != "text" or primitive.get("id") not in text_ids:
            new_primitives.append(primitive)
            continue
        lines = [line.strip() for line in str(primitive.get("text", "")).splitlines() if line.strip()]
        if len(lines) <= 1 or not primitive.get("bbox"):
            new_primitives.append(primitive)
            continue
        x0, y0, x1, y1 = [float(v) for v in primitive["bbox"]]
        total_height = max(4.0, y1 - y0)
        usable_height = max(3.0 * len(lines), total_height - line_gap * (len(lines) - 1))
        line_height = usable_height / len(lines)
        start_y = y0 + (total_height - usable_height - line_gap * (len(lines) - 1)) / 2.0
        style = copy.deepcopy(primitive.get("style") or {})
        style["font_size"] = max(5, int(style.get("font_size") or 8) + font_delta)
        created = []
        for index, line in enumerate(lines, start=1):
            ly0 = start_y + (index - 1) * (line_height + line_gap)
            ly1 = ly0 + line_height
            item = copy.deepcopy(primitive)
            base_id = f"{primitive.get('id')}_line_{index:02d}"
            item["id"] = _next_unique_id(base_id, used_ids)
            used_ids.add(item["id"])
            item["text"] = line
            item["bbox"] = [_r(x0), _r(ly0), _r(x1), _r(ly1)]
            item["style"] = copy.deepcopy(style)
            item["source"] = str(item.get("source", "text")) + "+split_multiline"
            new_primitives.append(item)
            created.append({
                "primitive_id": item["id"],
                "text": line,
                "bbox": item["bbox"],
            })
        operations.append({
            "action": "split_multiline_text",
            "removed_id": primitive.get("id"),
            "created": created,
            "font_delta": font_delta,
            "line_gap": line_gap,
        })
    updated["primitives"] = new_primitives
    _refresh_counts(updated)
    return updated, {
        "operation": "split_multiline_text",
        "config": {"font_delta": font_delta, "line_gap": line_gap},
        "counts": {
            "split": len(operations),
            "created_texts": sum(len(op["created"]) for op in operations),
        },
        "operations": operations,
    }


def _selected_ids(program: dict[str, Any], raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return [
            str(p.get("id"))
            for p in program.get("primitives", [])
            if p.get("type") == "text" and "\n" in str(p.get("text", ""))
        ]
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    return [
        str(p.get("id"))
        for p in program.get("primitives", [])
        if p.get("type") == "text" and str(p.get("id")) in wanted
    ]


def _write_variant(
    program: dict[str, Any],
    base: Path,
    tag: str,
    font_family: str,
    report: dict[str, Any],
) -> dict[str, str]:
    tag = _safe(tag)
    drawio = Path(f"{base}.{tag}.drawio")
    program_path = Path(f"{base}.{tag}.vp_program.json")
    report_path = Path(f"{base}.{tag}.report.json")
    save_program(program, program_path)
    compile_program_to_drawio(program, drawio, font_family=font_family)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    return {"drawio": str(drawio), "program": str(program_path), "report": str(report_path)}


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }


def _next_unique_id(base: str, used: set[Any]) -> str:
    if base not in used:
        return base
    index = 1
    while f"{base}_{index}" in used:
        index += 1
    return f"{base}_{index}"


def _ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _safe(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_+-]+", "_", str(value)).strip("_").replace("+", "p")[:100]


def _r(value: float) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    main()
