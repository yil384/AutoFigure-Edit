"""Render and rank pure-native draw.io reconstruction variants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .qa import (
    DEFAULT_DRAWIO_CLI,
    compute_render_metrics,
    export_drawio_png,
    make_side_by_side,
    validate_pure_native_drawio,
)


def evaluate_drawio_variants(
    source_image: str | Path,
    variants: list[str | Path],
    *,
    drawio_cli: str = DEFAULT_DRAWIO_CLI,
    export: bool = True,
    include_tiles: bool = False,
    panel_regions: list[dict[str, Any]] | None = None,
    expected_tokens: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Export variants, compute render metrics, and return ranked rows."""
    source = Path(source_image)
    rows = []
    for variant in variants:
        drawio = Path(variant)
        png = Path(str(drawio) + ".png")
        compare = drawio.with_suffix(drawio.suffix + ".compare.png")
        export_result = None
        if export or not png.exists():
            if export and png.exists():
                png.unlink()
            export_result = export_drawio_png(drawio, png, drawio_cli=drawio_cli)
        pure = validate_pure_native_drawio(drawio)
        metrics = None
        compare_result = None
        tile_metrics = None
        panel_metrics = None
        if png.exists() and (not export or not export_result or export_result.get("ok")):
            metrics = compute_render_metrics(
                source,
                png,
                expected_tokens=expected_tokens,
            )
            _attach_program_cleanliness(metrics, drawio)
            compare_result = make_side_by_side(source, png, compare)
            tile_metrics = (
                compute_tile_metrics(
                    source,
                    png,
                    drawio,
                    expected_tokens=expected_tokens,
                )
                if include_tiles else None
            )
            panel_metrics = (
                compute_region_metrics(
                    source,
                    png,
                    drawio,
                    panel_regions,
                    "panel",
                    expected_tokens=expected_tokens,
                )
                if panel_regions else None
            )
        row = {
            "drawio": str(drawio),
            "rendered_png": str(png) if png.exists() else None,
            "compare_png": str(compare) if compare.exists() else None,
            "native_purity": pure,
            "export": export_result,
            "metrics": metrics,
            "tile_metrics": tile_metrics,
            "panel_metrics": panel_metrics,
            "compare": compare_result,
        }
        row["score"] = score_variant(row)
        rows.append(row)
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def tile_winners(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the best variant per tile by the same scoring function."""
    return metric_winners(rows, "tile_metrics")


def panel_winners(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the best variant per derived panel by the same scoring function."""
    return metric_winners(rows, "panel_metrics")


def metric_winners(
    rows: list[dict[str, Any]],
    metrics_key: str,
) -> dict[str, dict[str, Any]]:
    """Return the best variant per local metric region."""
    local_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for local_name, metrics in (row.get(metrics_key) or {}).items():
            local_row = {
                "drawio": row.get("drawio"),
                "native_purity": row.get("native_purity"),
                "metrics": metrics,
            }
            local_row["score"] = score_variant(local_row)
            local_rows.setdefault(local_name, []).append(local_row)
    winners = {}
    for local_name, candidates in local_rows.items():
        candidates.sort(key=lambda row: row["score"], reverse=True)
        winners[local_name] = compact_score_row(candidates[0])
    return winners


def compute_tile_metrics(
    source_image: str | Path,
    rendered_image: str | Path,
    artifact_stem: str | Path,
    expected_tokens: list[str] | None = None,
) -> dict[str, Any]:
    source = Image.open(source_image).convert("RGB")
    rendered = Image.open(rendered_image).convert("RGB")
    w = min(source.width, rendered.width)
    h = min(source.height, rendered.height)
    source = source.crop((0, 0, w, h))
    rendered = rendered.crop((0, 0, w, h))
    boxes = {
        "top_left": (0, 0, w // 2, h // 2),
        "top_right": (w // 2, 0, w, h // 2),
        "bottom_left": (0, h // 2, w // 2, h),
        "bottom_right": (w // 2, h // 2, w, h),
    }
    stem = Path(str(artifact_stem) + ".tile")
    out: dict[str, Any] = {}
    for name, box in boxes.items():
        src_path = Path(f"{stem}.{name}.source.png")
        ren_path = Path(f"{stem}.{name}.rendered.png")
        source.crop(box).save(src_path)
        rendered.crop(box).save(ren_path)
        out[name] = compute_render_metrics(
            src_path,
            ren_path,
            expected_tokens=expected_tokens,
        )
    return out


def compute_region_metrics(
    source_image: str | Path,
    rendered_image: str | Path,
    artifact_stem: str | Path,
    regions: list[dict[str, Any]],
    label: str = "region",
    expected_tokens: list[str] | None = None,
) -> dict[str, Any]:
    source = Image.open(source_image).convert("RGB")
    rendered = Image.open(rendered_image).convert("RGB")
    w = min(source.width, rendered.width)
    h = min(source.height, rendered.height)
    source = source.crop((0, 0, w, h))
    rendered = rendered.crop((0, 0, w, h))
    stem = Path(str(artifact_stem) + f".{label}")
    out: dict[str, Any] = {}
    for index, region in enumerate(regions, start=1):
        region_id = _safe_region_id(region.get("id") or f"{label}_{index:02d}")
        box = _clamp_box(region.get("bbox"), w, h)
        if not box:
            continue
        src_path = Path(f"{stem}.{region_id}.source.png")
        ren_path = Path(f"{stem}.{region_id}.rendered.png")
        source.crop(box).save(src_path)
        rendered.crop(box).save(ren_path)
        metrics = compute_render_metrics(
            src_path,
            ren_path,
            expected_tokens=expected_tokens,
        )
        metrics["region"] = {
            "id": region_id,
            "bbox": list(box),
            "kind": region.get("kind"),
        }
        out[region_id] = metrics
    return out


def _safe_region_id(value: str) -> str:
    chars = []
    for char in str(value):
        if char.isalnum() or char in {"_", "-"}:
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "region"


def _clamp_box(
    raw: Any,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if not raw or len(raw) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in raw]
    x0 = max(0, min(width, int(round(x0))))
    y0 = max(0, min(height, int(round(y0))))
    x1 = max(0, min(width, int(round(x1))))
    y1 = max(0, min(height, int(round(y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return (x0, y0, x1, y1)


def score_variant(row: dict[str, Any]) -> float:
    if not row.get("native_purity", {}).get("ok"):
        return -1e9
    metrics = row.get("metrics")
    if not metrics:
        return -1e9
    edge = metrics.get("edge") or {}
    ocr = metrics.get("ocr") or {}
    semantic_ocr = ocr.get("semantic") or {}
    target_ocr = ocr.get("target_semantic") or {}
    edge_f1 = float(edge.get("f1") or 0.0)
    edge_precision = float(edge.get("precision") or 0.0)
    ocr_f1 = max(
        float(ocr.get("f1") or 0.0),
        float(semantic_ocr.get("f1") or 0.0),
        float(target_ocr.get("f1") or 0.0),
    )
    ocr_precision = max(
        float(ocr.get("precision") or 0.0),
        float(semantic_ocr.get("precision") or 0.0),
        float(target_ocr.get("precision") or 0.0),
    )
    changed = float(metrics.get("changed_pixel_ratio_t30") or 0.0)
    rendered_edges = float(edge.get("rendered_edges") or 0.0)
    reference_edges = float(edge.get("reference_edges") or 1.0)

    dirt_penalty = max(0.0, changed - 0.17) * 0.45
    overdraw_ratio = rendered_edges / max(1.0, reference_edges)
    overdraw_penalty = max(0.0, overdraw_ratio - 0.86) * 0.10
    cleanliness = metrics.get("program_cleanliness") or {}
    residual_text_overlap_count = float(
        cleanliness.get("residual_text_overlap_count") or 0.0
    )
    residual_text_overlap_penalty = min(
        0.004,
        residual_text_overlap_count * 0.00004,
    )
    return round(
        0.34 * edge_f1 +
        0.34 * ocr_f1 +
        0.16 * edge_precision +
        0.16 * ocr_precision -
        dirt_penalty -
        overdraw_penalty -
        residual_text_overlap_penalty,
        6,
    )


def compact_score_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    edge = metrics.get("edge") or {}
    ocr = metrics.get("ocr") or {}
    semantic_ocr = ocr.get("semantic") or {}
    target_ocr = ocr.get("target_semantic") or {}
    return {
        "drawio": row.get("drawio"),
        "score": row.get("score"),
        "edge_f1": edge.get("f1"),
        "edge_precision": edge.get("precision"),
        "ocr_f1": ocr.get("f1"),
        "ocr_precision": ocr.get("precision"),
        "ocr_semantic_f1": semantic_ocr.get("f1"),
        "ocr_semantic_precision": semantic_ocr.get("precision"),
        "ocr_target_semantic_f1": target_ocr.get("f1"),
        "ocr_target_semantic_precision": target_ocr.get("precision"),
        "residual_text_overlap_count": (
            (metrics.get("program_cleanliness") or {})
            .get("residual_text_overlap_count")
        ),
        "changed_pixel_ratio_t30": metrics.get("changed_pixel_ratio_t30"),
        "pure": row.get("native_purity", {}).get("ok"),
    }


def _attach_program_cleanliness(metrics: dict[str, Any],
                                drawio_path: str | Path) -> None:
    program_path = Path(drawio_path).with_suffix(".vp_program.json")
    if not program_path.exists():
        return
    try:
        program = json.loads(program_path.read_text())
    except Exception:
        return
    metrics["program_cleanliness"] = _program_cleanliness_metrics(program)


def _program_cleanliness_metrics(program: dict[str, Any]) -> dict[str, Any]:
    primitives = program.get("primitives") or []
    texts = [
        primitive
        for primitive in primitives
        if primitive.get("type") == "text" and _bbox(primitive)
    ]
    residuals = [
        primitive
        for primitive in primitives
        if primitive.get("type") in {"edge", "shape"}
        and _is_residual_micro_primitive(primitive)
        and _bbox(primitive)
    ]
    overlap_pairs = []
    for residual in residuals:
        residual_box = _bbox(residual)
        if not residual_box:
            continue
        for text in texts:
            text_box = _bbox(text)
            if text_box and _intersects(residual_box, _pad_bbox(text_box, 1.5)):
                overlap_pairs.append({
                    "residual_id": residual.get("id"),
                    "text_id": text.get("id"),
                    "text": text.get("text"),
                })
                break
    return {
        "residual_text_overlap_count": len(overlap_pairs),
        "residual_text_overlap_sample": overlap_pairs[:12],
    }


def _is_residual_micro_primitive(primitive: dict[str, Any]) -> bool:
    source = str(primitive.get("source") or "")
    role = str(primitive.get("role") or "")
    if "residual" not in source and "residual" not in role:
        return False
    box = _bbox(primitive)
    if not box:
        return False
    x0, y0, x1, y1 = box
    return (x1 - x0) <= 18.0 and (y1 - y0) <= 18.0


def _bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = primitive.get("bbox")
    if raw and len(raw) == 4:
        return tuple(float(value) for value in raw)
    path = primitive.get("path") or []
    if not path:
        return None
    xs = [float(point[0]) for point in path if len(point) >= 2]
    ys = [float(point[1]) for point in path if len(point) >= 2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _pad_bbox(
    box: tuple[float, float, float, float],
    pad: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def _intersects(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or ax0 > bx1 or ay1 < by0 or ay0 > by1)
