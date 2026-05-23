"""Compose a visual primitive program from per-tile variant winners."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .schema import save_program


def compose_program_from_tile_winners(
    manifest: dict[str, Any],
    ranking: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a native program by selecting primitives from tile winners."""
    if not manifest.get("variants"):
        raise ValueError("manifest has no variants")
    base_program = _load_program(Path(manifest["variants"][0]["program"]))
    canvas = base_program.get("canvas", {})
    width = float(canvas.get("width", 0))
    height = float(canvas.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("program canvas is invalid")
    return _compose_program_from_region_winners(
        manifest,
        ranking,
        _tile_boxes(width, height),
        winners_key="tile_winners",
        strategy="center_in_quadrant",
        metadata_prefix="tile",
    )


def compose_program_from_panel_winners(
    manifest: dict[str, Any],
    ranking: dict[str, Any],
    panel_regions: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a native program by selecting primitives from panel winners."""
    regions = panel_regions or ranking.get("panel_regions") or []
    boxes = {
        str(region["id"]): tuple(float(v) for v in region["bbox"])
        for region in regions
        if region.get("id") and region.get("bbox")
    }
    if not boxes:
        raise ValueError("no panel regions available")
    return _compose_program_from_region_winners(
        manifest,
        ranking,
        boxes,
        winners_key="panel_winners",
        strategy="center_in_cv_panel",
        metadata_prefix="panel",
    )


def _compose_program_from_region_winners(
    manifest: dict[str, Any],
    ranking: dict[str, Any],
    boxes: dict[str, tuple[float, float, float, float]],
    *,
    winners_key: str,
    strategy: str,
    metadata_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    drawio_to_program = {
        item["drawio"]: Path(item["program"])
        for item in manifest.get("variants", [])
        if item.get("drawio") and item.get("program")
    }
    if not manifest.get("variants"):
        raise ValueError("manifest has no variants")

    base_program = _load_program(Path(manifest["variants"][0]["program"]))
    canvas = base_program.get("canvas", {})
    width = float(canvas.get("width", 0))
    height = float(canvas.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("program canvas is invalid")

    global_winner = ranking.get("winner")
    local_winners = ranking.get(winners_key) or {}

    selected_programs: dict[str, dict[str, Any]] = {}
    for region_name in boxes:
        winner_drawio = (local_winners.get(region_name) or {}).get("drawio") or global_winner
        program_path = drawio_to_program.get(winner_drawio)
        if not program_path:
            program_path = Path(manifest["variants"][0]["program"])
            winner_drawio = manifest["variants"][0]["drawio"]
        selected_programs[region_name] = {
            "drawio": winner_drawio,
            "program": _load_program(program_path),
            "program_path": str(program_path),
        }
    fallback_path = drawio_to_program.get(global_winner)
    if not fallback_path:
        fallback_path = Path(manifest["variants"][0]["program"])
        global_winner = manifest["variants"][0]["drawio"]
    fallback_program = _load_program(fallback_path)

    composed = copy.deepcopy(base_program)
    composed_primitives = []
    used_ids: set[str] = set()
    operations = []
    fallback_count = 0
    for primitive in fallback_program.get("primitives", []):
        if _best_region_for_primitive(primitive, boxes) is not None:
            continue
        new_primitive = _copy_with_source_metadata(
            primitive,
            f"{metadata_prefix}_fallback",
            metadata_prefix,
            used_ids,
            source_drawio=global_winner,
        )
        composed_primitives.append(new_primitive)
        fallback_count += 1

    for region_name, box in boxes.items():
        selected = selected_programs[region_name]
        program = selected["program"]
        selected_count = 0
        for primitive in program.get("primitives", []):
            if _best_region_for_primitive(primitive, boxes) != region_name:
                continue
            new_primitive = _copy_with_source_metadata(
                primitive,
                region_name,
                metadata_prefix,
                used_ids,
                source_drawio=selected["drawio"],
            )
            composed_primitives.append(new_primitive)
            selected_count += 1
        operations.append({
            metadata_prefix: region_name,
            "bbox": list(box),
            "source_drawio": selected["drawio"],
            "source_program": selected["program_path"],
            "selected_primitives": selected_count,
        })

    composed["primitives"] = composed_primitives
    _refresh_counts(composed)
    composed.setdefault("metadata", {})[f"{metadata_prefix}_composition"] = {
        "strategy": strategy,
        "fallback_source_drawio": global_winner,
        "fallback_primitives": fallback_count,
        "operations": operations,
    }
    report = {
        "strategy": strategy,
        "global_winner": global_winner,
        "fallback_primitives": fallback_count,
        "operations": operations,
        "counts": composed.get("counts", {}),
    }
    return composed, report


def save_composed_program(program: dict[str, Any], path: str | Path) -> None:
    save_program(program, path)


def _load_program(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text())


def _tile_boxes(width: float, height: float) -> dict[str, tuple[float, float, float, float]]:
    return {
        "top_left": (0.0, 0.0, width / 2.0, height / 2.0),
        "top_right": (width / 2.0, 0.0, width, height / 2.0),
        "bottom_left": (0.0, height / 2.0, width / 2.0, height),
        "bottom_right": (width / 2.0, height / 2.0, width, height),
    }


def _primitive_center_in_tile(
    primitive: dict[str, Any],
    box: tuple[float, float, float, float],
) -> bool:
    bbox = primitive.get("bbox")
    if not bbox and primitive.get("path"):
        path = primitive["path"]
        xs = [float(p[0]) for p in path]
        ys = [float(p[1]) for p in path]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    if not bbox:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bx0, by0, bx1, by1 = box
    return bx0 <= cx < bx1 and by0 <= cy < by1


def _best_region_for_primitive(
    primitive: dict[str, Any],
    boxes: dict[str, tuple[float, float, float, float]],
) -> str | None:
    bbox = _primitive_bbox(primitive)
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    candidates = [
        (name, box)
        for name, box in boxes.items()
        if box[0] <= cx < box[2] and box[1] <= cy < box[3]
    ]
    if not candidates:
        return None
    if primitive.get("type") == "region":
        p_area = _box_area(bbox)
        candidates.sort(
            key=lambda item: (
                _overlap_fraction(bbox, item[1]),
                -abs(_box_area(item[1]) - p_area),
            ),
            reverse=True,
        )
        return candidates[0][0]
    candidates.sort(key=lambda item: _box_area(item[1]))
    return candidates[0][0]


def _primitive_bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = primitive.get("bbox")
    if bbox:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return (x0, y0, x1, y1)
    if primitive.get("path"):
        path = primitive["path"]
        xs = [float(p[0]) for p in path]
        ys = [float(p[1]) for p in path]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _copy_with_source_metadata(
    primitive: dict[str, Any],
    region_name: str,
    metadata_prefix: str,
    used_ids: set[str],
    *,
    source_drawio: str | None,
) -> dict[str, Any]:
    new_primitive = copy.deepcopy(primitive)
    old_id = str(new_primitive.get("id", "primitive"))
    new_id = f"{region_name}_{old_id}"
    suffix = 1
    while new_id in used_ids:
        suffix += 1
        new_id = f"{region_name}_{old_id}_{suffix}"
    new_primitive["id"] = new_id
    metadata = new_primitive.setdefault("metadata", {})
    metadata[f"{metadata_prefix}_source_id"] = old_id
    metadata[f"{metadata_prefix}_name"] = region_name
    metadata[f"{metadata_prefix}_source_drawio"] = source_drawio
    used_ids.add(new_id)
    return new_primitive


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap_fraction(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max(1.0, _box_area(a))


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
