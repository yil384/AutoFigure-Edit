"""Render and rank pure-native draw.io reconstruction variants."""
from __future__ import annotations

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
            metrics = compute_render_metrics(source, png)
            compare_result = make_side_by_side(source, png, compare)
            tile_metrics = (
                compute_tile_metrics(source, png, drawio)
                if include_tiles else None
            )
            panel_metrics = (
                compute_region_metrics(source, png, drawio, panel_regions, "panel")
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
        out[name] = compute_render_metrics(src_path, ren_path)
    return out


def compute_region_metrics(
    source_image: str | Path,
    rendered_image: str | Path,
    artifact_stem: str | Path,
    regions: list[dict[str, Any]],
    label: str = "region",
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
        metrics = compute_render_metrics(src_path, ren_path)
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
    edge_f1 = float(edge.get("f1") or 0.0)
    edge_precision = float(edge.get("precision") or 0.0)
    ocr_f1 = float(ocr.get("f1") or 0.0)
    ocr_precision = float(ocr.get("precision") or 0.0)
    changed = float(metrics.get("changed_pixel_ratio_t30") or 0.0)
    rendered_edges = float(edge.get("rendered_edges") or 0.0)
    reference_edges = float(edge.get("reference_edges") or 1.0)

    dirt_penalty = max(0.0, changed - 0.17) * 0.45
    overdraw_ratio = rendered_edges / max(1.0, reference_edges)
    overdraw_penalty = max(0.0, overdraw_ratio - 0.86) * 0.10
    return round(
        0.34 * edge_f1 +
        0.34 * ocr_f1 +
        0.16 * edge_precision +
        0.16 * ocr_precision -
        dirt_penalty -
        overdraw_penalty,
        6,
    )


def compact_score_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    edge = metrics.get("edge") or {}
    ocr = metrics.get("ocr") or {}
    return {
        "drawio": row.get("drawio"),
        "score": row.get("score"),
        "edge_f1": edge.get("f1"),
        "edge_precision": edge.get("precision"),
        "ocr_f1": ocr.get("f1"),
        "ocr_precision": ocr.get("precision"),
        "changed_pixel_ratio_t30": metrics.get("changed_pixel_ratio_t30"),
        "pure": row.get("native_purity", {}).get("ok"),
    }
