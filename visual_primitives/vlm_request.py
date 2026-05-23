"""Build and parse VLM refinement requests for visual primitive programs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ALLOWED_PATCH_OPS = {"add", "delete", "update", "split", "merge", "restyle", "reroute"}
ALLOWED_PRIMITIVE_TYPES = {"region", "text", "edge", "shape"}


PATCH_SCHEMA = {
    "visual_diagnosis": [
        {
            "issue": "short description",
            "region": [0, 0, 100, 100],
            "primitive_ids": ["text_0001"],
            "severity": "high|medium|low",
        }
    ],
    "patch_plan": [
        {
            "op": "add|delete|update|split|merge|restyle|reroute",
            "target_ids": ["primitive_id"],
            "updates": {
                "bbox": [0, 0, 100, 100],
                "path": [[0, 0], [100, 100]],
                "text": "optional",
                "style": {},
            },
            "new_primitive": {
                "id": "optional_new_id",
                "type": "region|text|edge|shape",
                "bbox": [0, 0, 100, 100],
                "path": [[0, 0], [100, 100]],
                "text": "optional",
                "style": {},
            },
            "reason": "why this patch improves visual fidelity",
        }
    ],
    "stop_or_continue": "continue|stop",
}


def build_refinement_request(*,
                             source_image: str | Path,
                             compare_image: str | Path,
                             program_path: str | Path,
                             qa_report_path: str | Path,
                             program: dict[str, Any],
                             metrics: dict[str, Any] | None = None,
                             compare_crops: dict[str, str] | None = None,
                             cv_evidence_path: str | Path | None = None,
                             cv_overlay_path: str | Path | None = None,
                             cv_evidence: dict[str, Any] | None = None,
                             max_per_type: int = 36) -> dict[str, Any]:
    artifacts = {
        "source_image": str(source_image),
        "side_by_side_image": str(compare_image),
        "side_by_side_quadrants": compare_crops or {},
        "program_path": str(program_path),
        "qa_report_path": str(qa_report_path),
    }
    if cv_evidence_path:
        artifacts["cv_evidence_json"] = str(cv_evidence_path)
    if cv_overlay_path:
        artifacts["cv_overlay_image"] = str(cv_overlay_path)
    return {
        "task": "vlm_visual_primitive_refinement",
        "paper_positioning": {
            "core_idea": "Think and draw with visual primitives",
            "goal": "convert a noisy raster scientific figure into pure-native editable draw.io primitives",
            "reference_gap_fix": "every critique and edit must point to primitive ids or pixel coordinates",
            "cv_tool_role": "CV proposes measurable geometry evidence; VLM decides semantic grouping and edit patches.",
        },
        "hard_constraints": {
            "drawio_native_only": True,
            "forbidden": [
                "source image overlay",
                "raster crop",
                "data:image",
                "base64",
                "shape=image",
                "shape=stencil",
                "stencil(",
            ],
        },
        "artifacts": artifacts,
        "metrics": metrics or {},
        "program_counts": program.get("counts", {}),
        "program_excerpt": _program_excerpt(program, max_per_type=max_per_type),
        "cv_evidence_excerpt": _cv_evidence_excerpt(cv_evidence or {}),
        "spatial_index": _spatial_index(program),
        "patch_schema": PATCH_SCHEMA,
        "instructions": _instructions(),
    }


def build_tile_refinement_request(*,
                                  base_request: dict[str, Any],
                                  program: dict[str, Any],
                                  tile_name: str,
                                  max_per_type: int = 36) -> dict[str, Any]:
    tile = base_request.get("spatial_index", {}).get(tile_name)
    if not tile:
        raise ValueError(f"unknown tile: {tile_name}")
    crop_path = base_request["artifacts"].get(
        "side_by_side_quadrants", {}).get(tile_name)
    tile_program = _filter_program_to_bbox(program, tile["bbox"])
    request = dict(base_request)
    request["task"] = "vlm_tile_visual_primitive_refinement"
    request["focus_tile"] = {
        "name": tile_name,
        "bbox": tile["bbox"],
        "side_by_side_crop": crop_path,
    }
    request["artifacts"] = dict(base_request["artifacts"])
    if crop_path:
        request["artifacts"]["focused_side_by_side_image"] = crop_path
    request["program_counts"] = tile_program.get("counts", {})
    request["program_excerpt"] = _program_excerpt(
        tile_program, max_per_type=max_per_type)
    request["cv_evidence_excerpt"] = _filter_cv_evidence_to_bbox(
        base_request.get("cv_evidence_excerpt", {}),
        tile["bbox"],
    )
    request["spatial_index"] = {tile_name: tile}
    request["instructions"] = [
        f"Focus only on the {tile_name} crop and its primitives.",
        "Patch only primitives in this tile unless a connector crosses the tile boundary.",
        *_instructions(),
    ]
    return request


def write_refinement_bundle(request: dict[str, Any],
                            json_path: str | Path,
                            prompt_path: str | Path) -> None:
    json_out = Path(json_path)
    prompt_out = Path(prompt_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(request, indent=2, ensure_ascii=True))
    prompt_out.write_text(request_to_prompt(request))


def request_to_prompt(request: dict[str, Any]) -> str:
    artifacts = request["artifacts"]
    return f"""You are the VLM critic/refiner in a PNG-to-pure-drawio system.

Reference image: {artifacts['source_image']}
Side-by-side comparison image: {artifacts['side_by_side_image']}
Side-by-side quadrant crops:
{json.dumps(artifacts.get('side_by_side_quadrants', {}), indent=2)}
Full visual primitive program JSON: {artifacts['program_path']}
QA report: {artifacts['qa_report_path']}
CV geometry evidence JSON: {artifacts.get('cv_evidence_json', 'not provided')}
CV debug overlay image: {artifacts.get('cv_overlay_image', 'not provided')}

Core idea:
- Think with visual primitives.
- Every diagnosis must cite primitive ids and/or exact pixel-coordinate regions.
- Use CV evidence for coordinates: snap text/boxes/lines to cv_region/cv_line/cv_node ids where they are visually correct.
- Return edits as a machine-applicable primitive patch plan.

Hard constraints:
{json.dumps(request['hard_constraints'], indent=2)}

Current metrics:
{json.dumps(request.get('metrics', {}), indent=2)}

Primitive counts:
{json.dumps(request.get('program_counts', {}), indent=2)}

Program excerpt:
{json.dumps(request.get('program_excerpt', {}), indent=2)}

CV evidence excerpt:
{json.dumps(request.get('cv_evidence_excerpt', {}), indent=2)}

Spatial index:
{json.dumps(request.get('spatial_index', {}), indent=2)}

Return JSON only, with this schema:
{json.dumps(PATCH_SCHEMA, indent=2)}
"""


def parse_patch_response(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        match = re.search(r"(\{.*\})", text, re.S)
        if match:
            text = match.group(1)
    patch = json.loads(text)
    validate_patch_doc(patch)
    return patch


def validate_patch_doc(patch: dict[str, Any],
                       program: dict[str, Any] | None = None) -> None:
    plan = patch.get("patch_plan", [])
    if not isinstance(plan, list):
        raise ValueError("patch_plan must be a list")
    ids = {p.get("id") for p in (program or {}).get("primitives", [])}
    for i, op in enumerate(plan):
        if not isinstance(op, dict):
            raise ValueError(f"patch_plan[{i}] must be an object")
        kind = op.get("op")
        if kind not in ALLOWED_PATCH_OPS:
            raise ValueError(f"patch_plan[{i}] has unsupported op: {kind}")
        targets = op.get("target_ids") or []
        if isinstance(targets, str):
            targets = [targets]
        if kind != "add" and not targets:
            raise ValueError(f"patch_plan[{i}] must name target_ids")
        if program and kind != "add":
            missing = [t for t in targets if t not in ids]
            if missing:
                raise ValueError(f"patch_plan[{i}] target ids not found: {missing}")
        if kind == "add":
            primitive = op.get("new_primitive")
            if not isinstance(primitive, dict):
                raise ValueError(f"patch_plan[{i}] add requires new_primitive")
            ptype = primitive.get("type")
            if ptype not in ALLOWED_PRIMITIVE_TYPES:
                raise ValueError(f"patch_plan[{i}] unsupported primitive type: {ptype}")


def _instructions() -> list[str]:
    return [
        "Inspect the side-by-side image visually; do not rely only on metrics.",
        "Use CV geometry evidence as coordinate anchors when line, panel, junction, or arrow placement is uncertain.",
        "Prefer small targeted edits over replacing the entire program.",
        "Prioritize OCR/text placement, connector topology, panel geometry, and dirty speckles.",
        "Use add/delete/update/restyle/reroute patches over vague natural-language advice.",
        "Do not propose raster overlays, image cells, SVG stencils, or base64 payloads.",
    ]


def _program_excerpt(program: dict[str, Any], max_per_type: int) -> dict[str, Any]:
    primitives = program.get("primitives", [])
    by_type = {ptype: [] for ptype in sorted(ALLOWED_PRIMITIVE_TYPES)}
    for primitive in primitives:
        ptype = primitive.get("type")
        if ptype in by_type:
            by_type[ptype].append(_compact_primitive(primitive))
    return {
        "regions": by_type["region"][:max(80, max_per_type)],
        "texts": _select_texts(by_type["text"], max(160, max_per_type)),
        "edges": _select_edges(by_type["edge"], max_per_type),
        "shapes": _select_shapes(by_type["shape"], max_per_type),
    }


def _spatial_index(program: dict[str, Any]) -> dict[str, Any]:
    canvas = program.get("canvas", {})
    width = float(canvas.get("width", 0) or 0)
    height = float(canvas.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return {}
    tiles = {
        "top_left": [0, 0, width / 2, height / 2],
        "top_right": [width / 2, 0, width, height / 2],
        "bottom_left": [0, height / 2, width / 2, height],
        "bottom_right": [width / 2, height / 2, width, height],
    }
    out = {}
    for name, bbox in tiles.items():
        items = [
            p for p in program.get("primitives", [])
            if p.get("bbox") and _bbox_overlap_fraction(p["bbox"], bbox) > 0.15
        ]
        out[name] = {
            "bbox": [round(v, 3) for v in bbox],
            "counts": {
                "region": sum(1 for p in items if p.get("type") == "region"),
                "text": sum(1 for p in items if p.get("type") == "text"),
                "edge": sum(1 for p in items if p.get("type") == "edge"),
                "shape": sum(1 for p in items if p.get("type") == "shape"),
            },
            "primitive_ids": [p.get("id") for p in items[:120]],
            "texts": [
                {
                    "id": p.get("id"),
                    "bbox": p.get("bbox"),
                    "text": p.get("text"),
                    "confidence": p.get("confidence"),
                }
                for p in items
                if p.get("type") == "text"
            ][:80],
        }
    return out


def _filter_program_to_bbox(program: dict[str, Any],
                            bbox: list[float]) -> dict[str, Any]:
    filtered = dict(program)
    primitives = [
        p for p in program.get("primitives", [])
        if p.get("bbox") and _bbox_overlap_fraction(p["bbox"], bbox) > 0.05
    ]
    filtered["primitives"] = primitives
    filtered["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
    return filtered


def _compact_primitive(primitive: dict[str, Any]) -> dict[str, Any]:
    out = {
        "id": primitive.get("id"),
        "type": primitive.get("type"),
        "role": primitive.get("role"),
        "bbox": primitive.get("bbox"),
        "source": primitive.get("source"),
        "confidence": primitive.get("confidence"),
    }
    if primitive.get("text"):
        out["text"] = primitive.get("text")
    if primitive.get("shape"):
        out["shape"] = primitive.get("shape")
    if primitive.get("path"):
        path = primitive.get("path")
        if len(path) <= 6:
            out["path"] = path
        else:
            out["path_sample"] = path[:3] + path[-3:]
            out["path_points"] = len(path)
    style = primitive.get("style")
    if style:
        out["style"] = style
    length = primitive.get("length")
    if length is not None:
        out["length"] = length
    return out


def _select_texts(texts: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    def key(item):
        conf = item.get("confidence")
        conf = float(conf) if conf is not None else 100.0
        text = item.get("text", "")
        return (conf, -len(text))
    low_conf = sorted(texts, key=key)[:max_items]
    seen = {t.get("id") for t in low_conf}
    ordered = low_conf + [t for t in texts if t.get("id") not in seen]
    return ordered[:max_items]


def _bbox_overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / area


def _select_edges(edges: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    return sorted(edges, key=lambda e: -float(e.get("length") or 0.0))[:max_items]


def _select_shapes(shapes: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    def area(item):
        bbox = item.get("bbox") or [0, 0, 0, 0]
        return -max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return sorted(shapes, key=area)[:max_items]


def _cv_evidence_excerpt(evidence: dict[str, Any],
                         max_regions: int = 80,
                         max_lines: int = 120,
                         max_junctions: int = 80,
                         max_arrowheads: int = 60,
                         max_text_lines: int = 90) -> dict[str, Any]:
    if not evidence:
        return {}
    return {
        "counts": evidence.get("counts", {}),
        "regions": evidence.get("regions", [])[:max_regions],
        "line_segments": _select_cv_lines(
            evidence.get("line_segments", []), max_lines),
        "junctions": evidence.get("junctions", [])[:max_junctions],
        "arrowheads": evidence.get("arrowheads", [])[:max_arrowheads],
        "text_lines": evidence.get("text_lines", [])[:max_text_lines],
        "snap_guides": {
            "x": evidence.get("snap_guides", {}).get("x", [])[:80],
            "y": evidence.get("snap_guides", {}).get("y", [])[:80],
        },
    }


def _select_cv_lines(lines: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    return sorted(lines, key=lambda line: -float(line.get("length") or 0.0))[:max_items]


def _filter_cv_evidence_to_bbox(evidence: dict[str, Any],
                                bbox: list[float]) -> dict[str, Any]:
    if not evidence:
        return {}

    def keep(item: dict[str, Any]) -> bool:
        if item.get("bbox"):
            return _bbox_overlap_fraction(item["bbox"], bbox) > 0.04
        if item.get("point"):
            x, y = item["point"]
            x0, y0, x1, y1 = bbox
            return x0 <= x <= x1 and y0 <= y <= y1
        if item.get("centroid"):
            x, y = item["centroid"]
            x0, y0, x1, y1 = bbox
            return x0 <= x <= x1 and y0 <= y <= y1
        return False

    out = {
        "regions": [i for i in evidence.get("regions", []) if keep(i)],
        "line_segments": [i for i in evidence.get("line_segments", []) if keep(i)],
        "junctions": [i for i in evidence.get("junctions", []) if keep(i)],
        "arrowheads": [i for i in evidence.get("arrowheads", []) if keep(i)],
        "text_lines": [i for i in evidence.get("text_lines", []) if keep(i)],
        "snap_guides": evidence.get("snap_guides", {}),
    }
    out["counts"] = {
        "regions": len(out["regions"]),
        "line_segments": len(out["line_segments"]),
        "junctions": len(out["junctions"]),
        "arrowheads": len(out["arrowheads"]),
        "text_lines": len(out["text_lines"]),
    }
    return out
