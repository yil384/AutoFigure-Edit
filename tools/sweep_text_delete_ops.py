#!/usr/bin/env python3
"""Generate/evaluate one-at-a-time native text deletion variants."""
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
        description="Sweep deleting suspect native text primitives")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="text_delete_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--metrics-json", default=None,
                    help="ranking/eval JSON whose best-row OCR extra_sample guides candidates")
    ap.add_argument("--extra-tokens", default=None,
                    help="comma-separated normalized tokens to target")
    ap.add_argument("--max-candidates", type=int, default=32)
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    program = load_program(args.program_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    extra_tokens = _target_tokens(args.metrics_json, args.extra_tokens)
    candidates = _candidate_texts(program, extra_tokens, args.max_candidates)

    variants = []
    for index, candidate in enumerate(candidates, start=1):
        updated = copy.deepcopy(program)
        updated["primitives"] = [
            primitive for primitive in updated.get("primitives", [])
            if primitive.get("id") != candidate["primitive_id"]
        ]
        _refresh_counts(updated)
        tag = f"del{index:03d}_{_safe_name(candidate['text'])}"
        drawio = Path(f"{base}.{tag}.drawio")
        program_path = Path(f"{base}.{tag}.vp_program.json")
        report_path = Path(f"{base}.{tag}.report.json")
        save_program(updated, program_path)
        compile_program_to_drawio(updated, drawio, font_family=args.font_family)
        report_path.write_text(json.dumps({
            "operation": "delete_text",
            "candidate": candidate,
        }, indent=2, ensure_ascii=True))
        variants.append({
            "name": tag,
            "drawio": str(drawio),
            "program": str(program_path),
            "report": str(report_path),
            "candidate": candidate,
        })

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "target_tokens": sorted(extra_tokens),
        "candidates": candidates,
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


def _target_tokens(metrics_json: str | None, extra_tokens: str | None) -> set[str]:
    tokens = set()
    if extra_tokens:
        tokens.update(_norm_token(item) for item in extra_tokens.split(","))
    if metrics_json:
        data = json.loads(Path(metrics_json).read_text())
        rows = data.get("variants") or []
        if rows:
            rows = sorted(rows, key=lambda row: row.get("score", -1e9), reverse=True)
            ocr = ((rows[0].get("metrics") or {}).get("ocr") or {})
            tokens.update(_norm_token(item) for item in ocr.get("extra_sample") or [])
    return {token for token in tokens if token}


def _candidate_texts(
    program: dict[str, Any],
    extra_tokens: set[str],
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = []
    for primitive in program.get("primitives", []):
        if primitive.get("type") != "text":
            continue
        text = str(primitive.get("text", "")).strip()
        if not text:
            continue
        tokens = {_norm_token(token) for token in re.findall(r"[A-Za-z0-9+\-/]+", text)}
        tokens = {token for token in tokens if token}
        matched = sorted(tokens & extra_tokens)
        if not matched:
            continue
        bbox = primitive.get("bbox") or [0, 0, 0, 0]
        area = max(1.0, (float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))
        candidates.append({
            "primitive_id": primitive.get("id"),
            "text": text,
            "bbox": bbox,
            "source": primitive.get("source"),
            "matched_extra_tokens": matched,
            "area": round(area, 3),
            "font_size": primitive.get("style", {}).get("font_size"),
        })
    candidates.sort(key=lambda item: (
        len(item["matched_extra_tokens"]),
        item["area"],
        item["bbox"][1],
        item["bbox"][0],
    ))
    return candidates[:max_candidates]


def _norm_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9+\-/]+", "", str(value or "").lower())


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")[:42] or "text"


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }


if __name__ == "__main__":
    main()
