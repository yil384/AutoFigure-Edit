#!/usr/bin/env python3
"""Compose a conservative local-panel variant from pure-native programs."""
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
from visual_primitives.panel_regions import load_panel_regions  # noqa: E402
from visual_primitives.qa import validate_pure_native_drawio  # noqa: E402
from visual_primitives.schema import save_program  # noqa: E402
from visual_primitives.tile_compose import (  # noqa: E402
    _copy_with_source_metadata,
    _load_program,
    _primitive_bbox,
)
from visual_primitives.variant_eval import score_variant  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Conservatively replace only locally winning native primitives. "
            "Text is protected unless explicitly listed in --replace-types."
        )
    )
    ap.add_argument("manifest", help="variant manifest JSON")
    ap.add_argument("ranking", help="variant ranking JSON with panel_winners")
    ap.add_argument("-o", "--output", required=True, help="output drawio path")
    ap.add_argument("--panel-regions", required=True, help="panel regions JSON")
    ap.add_argument("--program-output", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--base-drawio", default=None,
                    help="fallback/base drawio; defaults to ranking winner")
    ap.add_argument("--base-program", default=None,
                    help="program JSON for --base-drawio when it is not in manifest")
    ap.add_argument("--replace-types", default="region,shape,edge",
                    help="comma-separated primitive types eligible for replacement")
    ap.add_argument("--min-score-delta", type=float, default=0.004,
                    help="minimum local score gain over the base region")
    ap.add_argument("--max-ocr-f1-drop", type=float, default=0.03,
                    help="reject local replacement if local OCR F1 drops by more than this")
    ap.add_argument("--max-ocr-precision-drop", type=float, default=0.06,
                    help="reject local replacement if local OCR precision drops by more than this")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="containment margin for primitives near region boundaries")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    ranking = json.loads(Path(args.ranking).read_text())
    panel_regions = load_panel_regions(args.panel_regions)
    replace_types = {
        value.strip()
        for value in args.replace_types.split(",")
        if value.strip()
    }
    program, report = compose_selective_panel_program(
        manifest=manifest,
        ranking=ranking,
        panel_regions=panel_regions,
        base_drawio=args.base_drawio,
        base_program_path=args.base_program,
        replace_types=replace_types,
        min_score_delta=args.min_score_delta,
        max_ocr_f1_drop=args.max_ocr_f1_drop,
        max_ocr_precision_drop=args.max_ocr_precision_drop,
        margin=args.margin,
    )

    output = Path(args.output)
    program_output = (
        Path(args.program_output)
        if args.program_output
        else output.with_suffix(".vp_program.json")
    )
    report_output = (
        Path(args.report)
        if args.report
        else output.with_suffix(".selective_compose_report.json")
    )
    save_program(program, program_output)
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
        "selected_regions": len(report.get("selected_regions", [])),
        "removed_primitives": report.get("removed_primitives", 0),
        "added_primitives": report.get("added_primitives", 0),
    }, indent=2))


def compose_selective_panel_program(
    *,
    manifest: dict[str, Any],
    ranking: dict[str, Any],
    panel_regions: list[dict[str, Any]],
    base_drawio: str | None,
    base_program_path: str | None,
    replace_types: set[str],
    min_score_delta: float,
    max_ocr_f1_drop: float,
    max_ocr_precision_drop: float,
    margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    drawio_to_program = {
        item["drawio"]: Path(item["program"])
        for item in manifest.get("variants", [])
        if item.get("drawio") and item.get("program")
    }
    if not drawio_to_program:
        raise ValueError("manifest has no drawio/program entries")

    base_drawio = base_drawio or ranking.get("winner")
    if base_drawio and base_program_path:
        drawio_to_program.setdefault(base_drawio, Path(base_program_path))
    if base_drawio not in drawio_to_program:
        base_drawio = manifest["variants"][0]["drawio"]
    base_program = _load_program(drawio_to_program[base_drawio])
    boxes = {
        str(region["id"]): tuple(float(v) for v in region["bbox"])
        for region in panel_regions
        if region.get("id") and region.get("bbox")
    }
    if not boxes:
        raise ValueError("no panel regions available")

    base_row = _find_variant_row(ranking, base_drawio)
    local_winners = ranking.get("panel_winners") or {}
    selected_regions = _select_regions(
        boxes=boxes,
        base_row=base_row,
        local_winners=local_winners,
        min_score_delta=min_score_delta,
        max_ocr_f1_drop=max_ocr_f1_drop,
        max_ocr_precision_drop=max_ocr_precision_drop,
        base_drawio=base_drawio,
    )

    selected_programs = {}
    for region_id, selected in selected_regions.items():
        winner_drawio = selected["winner_drawio"]
        program_path = drawio_to_program.get(winner_drawio)
        if not program_path:
            continue
        selected_programs[region_id] = {
            "drawio": winner_drawio,
            "program": _load_program(program_path),
            "program_path": str(program_path),
        }

    composed = copy.deepcopy(base_program)
    composed_primitives: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    removed = 0
    added = 0
    preserved_text = 0
    selected_boxes = {
        region_id: boxes[region_id]
        for region_id in selected_programs
        if region_id in boxes
    }

    for primitive in base_program.get("primitives", []):
        region_id = _eligible_region(
            primitive,
            selected_boxes,
            replace_types,
            margin=margin,
        )
        if region_id:
            removed += 1
            continue
        if primitive.get("type") == "text":
            preserved_text += 1
        composed_primitives.append(_copy_with_source_metadata(
            primitive,
            "selective_base",
            "selective",
            used_ids,
            source_drawio=base_drawio,
        ))

    operations = []
    for region_id, selected in selected_programs.items():
        box = boxes[region_id]
        region_added = 0
        for primitive in selected["program"].get("primitives", []):
            if _eligible_region(
                primitive,
                {region_id: box},
                replace_types,
                margin=margin,
            ) != region_id:
                continue
            composed_primitives.append(_copy_with_source_metadata(
                primitive,
                region_id,
                "selective",
                used_ids,
                source_drawio=selected["drawio"],
            ))
            added += 1
            region_added += 1
        operations.append({
            "panel": region_id,
            "bbox": list(box),
            "source_drawio": selected["drawio"],
            "source_program": selected["program_path"],
            "score_delta": selected_regions[region_id]["score_delta"],
            "base_score": selected_regions[region_id]["base_score"],
            "winner_score": selected_regions[region_id]["winner_score"],
            "added_primitives": region_added,
        })

    composed["primitives"] = composed_primitives
    _refresh_counts(composed)
    composed.setdefault("metadata", {})["selective_panel_composition"] = {
        "strategy": "protect_text_replace_contained_native_primitives",
        "base_drawio": base_drawio,
        "replace_types": sorted(replace_types),
        "min_score_delta": min_score_delta,
        "max_ocr_f1_drop": max_ocr_f1_drop,
        "max_ocr_precision_drop": max_ocr_precision_drop,
        "margin": margin,
        "operations": operations,
    }
    return composed, {
        "strategy": "protect_text_replace_contained_native_primitives",
        "base_drawio": base_drawio,
        "replace_types": sorted(replace_types),
        "min_score_delta": min_score_delta,
        "max_ocr_f1_drop": max_ocr_f1_drop,
        "max_ocr_precision_drop": max_ocr_precision_drop,
        "margin": margin,
        "selected_regions": list(operations),
        "removed_primitives": removed,
        "added_primitives": added,
        "preserved_text_primitives": preserved_text,
        "counts": composed.get("counts", {}),
    }


def _select_regions(
    *,
    boxes: dict[str, tuple[float, float, float, float]],
    base_row: dict[str, Any] | None,
    local_winners: dict[str, Any],
    min_score_delta: float,
    max_ocr_f1_drop: float,
    max_ocr_precision_drop: float,
    base_drawio: str,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    base_panel_metrics = (base_row or {}).get("panel_metrics") or {}
    for region_id in boxes:
        winner = local_winners.get(region_id) or {}
        winner_drawio = winner.get("drawio")
        if not winner_drawio or winner_drawio == base_drawio:
            continue
        base_metrics = base_panel_metrics.get(region_id)
        if not base_metrics:
            continue
        base_score = score_variant({
            "native_purity": {"ok": True},
            "metrics": base_metrics,
        })
        winner_score = float(winner.get("score") or -1e9)
        if winner_score - base_score < min_score_delta:
            continue
        base_ocr = base_metrics.get("ocr") or {}
        winner_ocr_f1 = float(winner.get("ocr_f1") or 0.0)
        winner_ocr_precision = float(winner.get("ocr_precision") or 0.0)
        if float(base_ocr.get("f1") or 0.0) - winner_ocr_f1 > max_ocr_f1_drop:
            continue
        if (
            float(base_ocr.get("precision") or 0.0) - winner_ocr_precision
            > max_ocr_precision_drop
        ):
            continue
        selected[region_id] = {
            "winner_drawio": winner_drawio,
            "base_score": base_score,
            "winner_score": winner_score,
            "score_delta": round(winner_score - base_score, 6),
        }
    return selected


def _find_variant_row(ranking: dict[str, Any], drawio: str) -> dict[str, Any] | None:
    for row in ranking.get("variants") or []:
        if row.get("drawio") == drawio:
            return row
    return None


def _eligible_region(
    primitive: dict[str, Any],
    boxes: dict[str, tuple[float, float, float, float]],
    replace_types: set[str],
    *,
    margin: float,
) -> str | None:
    if primitive.get("type") not in replace_types:
        return None
    bbox = _primitive_bbox(primitive)
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    for region_id, box in boxes.items():
        bx0, by0, bx1, by1 = box
        if not (bx0 <= cx < bx1 and by0 <= cy < by1):
            continue
        if primitive.get("path"):
            path = primitive.get("path") or []
            if all(_point_inside(point, box, margin) for point in path):
                return region_id
            continue
        if (
            x0 >= bx0 - margin and
            y0 >= by0 - margin and
            x1 <= bx1 + margin and
            y1 <= by1 + margin
        ):
            return region_id
    return None


def _point_inside(
    point: Any,
    box: tuple[float, float, float, float],
    margin: float,
) -> bool:
    if not point or len(point) < 2:
        return False
    x = float(point[0])
    y = float(point[1])
    return (
        box[0] - margin <= x <= box[2] + margin and
        box[1] - margin <= y <= box[3] + margin
    )


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
