"""Apply VLM-produced patch plans to visual primitive programs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_patch(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def apply_patch_plan(program: dict[str, Any],
                     patch_doc: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(program)
    primitives = updated.setdefault("primitives", [])
    patch_plan = patch_doc.get("patch_plan", patch_doc)
    if isinstance(patch_plan, dict):
        patch_plan = [patch_plan]
    if not isinstance(patch_plan, list):
        raise ValueError("patch plan must be a list or object")

    for op in patch_plan:
        _apply_one(primitives, op)
    _refresh_counts(updated)
    updated.setdefault("metadata", {})["last_patch_diagnosis"] = patch_doc.get(
        "visual_diagnosis", [])
    return updated


def _apply_one(primitives: list[dict[str, Any]], op: dict[str, Any]) -> None:
    kind = op.get("op")
    targets = op.get("target_ids") or op.get("targets") or []
    if isinstance(targets, str):
        targets = [targets]

    if kind == "add":
        new_primitive = copy.deepcopy(op.get("new_primitive") or {})
        if not new_primitive:
            return
        if "id" not in new_primitive:
            new_primitive["id"] = _next_id(primitives, "vlm")
        new_primitive.setdefault("source", "vlm_patch")
        primitives.append(new_primitive)
        return

    if kind == "add_many":
        for raw in op.get("new_primitives") or []:
            new_primitive = copy.deepcopy(raw)
            if not new_primitive:
                continue
            if "id" not in new_primitive:
                new_primitive["id"] = _next_id(primitives, "vlm")
            new_primitive.setdefault("source", "vlm_patch")
            primitives.append(new_primitive)
        return

    if kind == "delete":
        doomed = set(targets)
        primitives[:] = [p for p in primitives if p.get("id") not in doomed]
        return

    if kind == "delete_in_bbox":
        doomed = _ids_in_bbox(primitives, op)
        primitives[:] = [p for p in primitives if p.get("id") not in doomed]
        return

    for primitive in primitives:
        if primitive.get("id") not in targets:
            continue
        if kind in {"update", "split", "merge"}:
            patch = op.get("updates") or op.get("new_primitive") or {}
            _deep_merge(primitive, patch)
            primitive["source"] = primitive.get("source", "unknown") + "+vlm_update"
        elif kind == "restyle":
            style = op.get("style") or (op.get("new_primitive") or {}).get("style") or {}
            primitive.setdefault("style", {}).update(style)
            primitive["source"] = primitive.get("source", "unknown") + "+vlm_style"
        elif kind == "reroute":
            path = op.get("path") or (op.get("new_primitive") or {}).get("path")
            if path:
                primitive["path"] = path
                primitive["source"] = primitive.get("source", "unknown") + "+vlm_route"


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _next_id(primitives: list[dict[str, Any]], prefix: str) -> str:
    existing = {p.get("id") for p in primitives}
    i = 1
    while f"{prefix}_{i:04d}" in existing:
        i += 1
    return f"{prefix}_{i:04d}"


def _ids_in_bbox(primitives: list[dict[str, Any]], op: dict[str, Any]) -> set[str]:
    bbox = op.get("bbox")
    if not bbox or len(bbox) != 4:
        return set()
    box = tuple(float(v) for v in bbox)
    allowed_types = set(op.get("types") or [])
    protect_text = bool(op.get("protect_text", True))
    mode = str(op.get("mode") or "center")
    margin = float(op.get("margin") or 0.0)
    out: set[str] = set()
    for primitive in primitives:
        primitive_id = primitive.get("id")
        if not primitive_id:
            continue
        ptype = primitive.get("type")
        if protect_text and ptype == "text":
            continue
        if allowed_types and ptype not in allowed_types:
            continue
        primitive_bbox = _primitive_bbox(primitive)
        if not primitive_bbox:
            continue
        if _bbox_matches(primitive_bbox, box, mode=mode, margin=margin):
            out.add(str(primitive_id))
    return out


def _primitive_bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = primitive.get("bbox")
    if bbox:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return (x0, y0, x1, y1)
    path = primitive.get("path") or []
    if path:
        xs = [float(point[0]) for point in path]
        ys = [float(point[1]) for point in path]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _bbox_matches(
    bbox: tuple[float, float, float, float],
    box: tuple[float, float, float, float],
    *,
    mode: str,
    margin: float,
) -> bool:
    x0, y0, x1, y1 = bbox
    bx0, by0, bx1, by1 = box
    bx0 -= margin
    by0 -= margin
    bx1 += margin
    by1 += margin
    if mode == "intersect":
        return not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1)
    if mode == "contained":
        return x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
