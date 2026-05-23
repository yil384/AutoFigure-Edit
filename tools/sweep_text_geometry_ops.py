#!/usr/bin/env python3
"""Generate/evaluate one-at-a-time native text geometry variants."""
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


DEFAULT_OPS = [
    {"kind": "shift", "dx": -2.0, "dy": 0.0},
    {"kind": "shift", "dx": 2.0, "dy": 0.0},
    {"kind": "shift", "dx": 0.0, "dy": -2.0},
    {"kind": "shift", "dx": 0.0, "dy": 2.0},
    {"kind": "shift", "dx": -3.0, "dy": 0.0},
    {"kind": "shift", "dx": 3.0, "dy": 0.0},
    {"kind": "font", "delta": -1},
    {"kind": "font", "delta": 1},
    {"kind": "box", "dw": -4.0, "dh": 0.0},
    {"kind": "box", "dw": 4.0, "dh": 0.0},
    {"kind": "box", "dw": 0.0, "dh": -2.0},
    {"kind": "box", "dw": 0.0, "dh": 2.0},
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sweep text primitive shifts, font-size changes, and bbox resizes")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="text_geometry_sweep")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--metrics-json", default=None,
                    help="ranking/eval JSON whose best-row OCR extras/missing tokens guide candidates")
    ap.add_argument("--candidate-ids", default=None,
                    help="comma-separated text primitive ids to force into the candidate set")
    ap.add_argument("--max-texts", type=int, default=12)
    ap.add_argument("--max-variants", type=int, default=96)
    ap.add_argument("--ops-json", default=None,
                    help="optional JSON array of operations")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    program = load_program(args.program_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    metrics_tokens = _metrics_tokens(args.metrics_json)
    forced_ids = {
        item.strip() for item in (args.candidate_ids or "").split(",")
        if item.strip()
    }
    candidates = _candidate_texts(
        program,
        metrics_tokens,
        forced_ids,
        max_texts=args.max_texts,
    )
    ops = _load_ops(args.ops_json)
    variants = []
    for candidate in candidates:
        for op in ops:
            if len(variants) >= args.max_variants:
                break
            updated = copy.deepcopy(program)
            primitive = _primitive_by_id(updated, candidate["primitive_id"])
            if primitive is None:
                continue
            before = copy.deepcopy({
                "bbox": primitive.get("bbox"),
                "style": primitive.get("style", {}),
            })
            if not _apply_op(primitive, op):
                continue
            tag = f"geo{len(variants) + 1:03d}_{_safe_name(candidate['text'])}_{_op_name(op)}"
            drawio = Path(f"{base}.{tag}.drawio")
            program_path = Path(f"{base}.{tag}.vp_program.json")
            report_path = Path(f"{base}.{tag}.report.json")
            primitive["source"] = str(primitive.get("source", "text")) + "+text_geo_sweep"
            save_program(updated, program_path)
            compile_program_to_drawio(updated, drawio, font_family=args.font_family)
            report_path.write_text(json.dumps({
                "operation": "text_geometry",
                "candidate": candidate,
                "op": op,
                "before": before,
                "after": {
                    "bbox": primitive.get("bbox"),
                    "style": primitive.get("style", {}),
                },
            }, indent=2, ensure_ascii=True))
            variants.append({
                "name": tag,
                "drawio": str(drawio),
                "program": str(program_path),
                "report": str(report_path),
                "candidate": candidate,
                "op": op,
            })
        if len(variants) >= args.max_variants:
            break

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "metrics_tokens": metrics_tokens,
        "forced_ids": sorted(forced_ids),
        "candidates": candidates,
        "ops": ops,
        "variants": variants,
    }, indent=2, ensure_ascii=True))
    if args.generate_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "variant_count": len(variants),
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


def _metrics_tokens(metrics_json: str | None) -> dict[str, list[str]]:
    if not metrics_json:
        return {"extra": [], "missing": []}
    data = json.loads(Path(metrics_json).read_text())
    rows = data.get("variants") or []
    if not rows:
        return {"extra": [], "missing": []}
    rows = sorted(rows, key=lambda row: row.get("score", -1e9), reverse=True)
    ocr = ((rows[0].get("metrics") or {}).get("ocr") or {})
    return {
        "extra": sorted({_norm_token(item) for item in ocr.get("extra_sample") or [] if _norm_token(item)}),
        "missing": sorted({_norm_token(item) for item in ocr.get("missing_sample") or [] if _norm_token(item)}),
    }


def _candidate_texts(
    program: dict[str, Any],
    metrics_tokens: dict[str, list[str]],
    forced_ids: set[str],
    *,
    max_texts: int,
) -> list[dict[str, Any]]:
    extra = set(metrics_tokens.get("extra") or [])
    missing = set(metrics_tokens.get("missing") or [])
    candidates = []
    for primitive in program.get("primitives", []):
        if primitive.get("type") != "text":
            continue
        text = str(primitive.get("text", "")).strip()
        if not text:
            continue
        primitive_id = str(primitive.get("id"))
        tokens = {
            _norm_token(token)
            for token in re.findall(r"[A-Za-z0-9+\-/]+", text)
        }
        tokens = {token for token in tokens if token}
        matched_extra = sorted(tokens & extra)
        matched_missing = sorted(tokens & missing)
        forced = primitive_id in forced_ids
        if not forced and not matched_extra and not matched_missing:
            continue
        bbox = primitive.get("bbox") or [0, 0, 0, 0]
        width = max(1.0, float(bbox[2]) - float(bbox[0]))
        height = max(1.0, float(bbox[3]) - float(bbox[1]))
        style = primitive.get("style") or {}
        candidates.append({
            "primitive_id": primitive_id,
            "text": text,
            "bbox": bbox,
            "source": primitive.get("source"),
            "font_size": style.get("font_size"),
            "matched_extra_tokens": matched_extra,
            "matched_missing_tokens": matched_missing,
            "forced": forced,
            "area": round(width * height, 3),
        })
    candidates.sort(key=lambda item: (
        0 if item["forced"] else 1,
        -len(item["matched_extra_tokens"]),
        -len(item["matched_missing_tokens"]),
        item["area"],
        item["bbox"][1],
        item["bbox"][0],
    ))
    return candidates[:max_texts]


def _load_ops(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return list(DEFAULT_OPS)
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("ops JSON must be an array")
    return [dict(item) for item in data if isinstance(item, dict)]


def _primitive_by_id(program: dict[str, Any], primitive_id: str) -> dict[str, Any] | None:
    for primitive in program.get("primitives", []):
        if primitive.get("id") == primitive_id:
            return primitive
    return None


def _apply_op(primitive: dict[str, Any], op: dict[str, Any]) -> bool:
    bbox = primitive.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    kind = str(op.get("kind"))
    if kind == "shift":
        dx = float(op.get("dx") or 0.0)
        dy = float(op.get("dy") or 0.0)
        if dx == 0 and dy == 0:
            return False
        primitive["bbox"] = [_round(x0 + dx), _round(y0 + dy), _round(x1 + dx), _round(y1 + dy)]
        return True
    if kind == "font":
        delta = int(op.get("delta") or 0)
        if delta == 0:
            return False
        style = primitive.setdefault("style", {})
        current = int(style.get("font_size") or 9)
        updated = max(5, min(24, current + delta))
        if updated == current:
            return False
        style["font_size"] = updated
        return True
    if kind == "box":
        dw = float(op.get("dw") or 0.0)
        dh = float(op.get("dh") or 0.0)
        if dw == 0 and dh == 0:
            return False
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = max(4.0, (x1 - x0) + dw)
        height = max(4.0, (y1 - y0) + dh)
        primitive["bbox"] = [
            _round(cx - width / 2.0),
            _round(cy - height / 2.0),
            _round(cx + width / 2.0),
            _round(cy + height / 2.0),
        ]
        return True
    return False


def _norm_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9+\-/]+", "", str(value or "").lower())


def _op_name(op: dict[str, Any]) -> str:
    kind = str(op.get("kind") or "op")
    if kind == "shift":
        return f"shift_{float(op.get('dx') or 0):g}_{float(op.get('dy') or 0):g}".replace("-", "m")
    if kind == "font":
        return f"font_{int(op.get('delta') or 0):+d}".replace("+", "p").replace("-", "m")
    if kind == "box":
        return f"box_{float(op.get('dw') or 0):g}_{float(op.get('dh') or 0):g}".replace("-", "m")
    return _safe_name(kind)


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")[:36] or "text"


def _round(value: float) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    main()
