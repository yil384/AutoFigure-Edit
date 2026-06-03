"""Compile a visual primitive program back to pure-native draw.io XML."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from png_to_drawio import (
    _emit_canvas_anchor_cell,
    _emit_edge_cell,
    _emit_native_shape_cell,
    _emit_rect_cell,
    _xml_escape,
    _wrap_drawio,
    assert_pure_native_drawio,
)


def compile_program_to_drawio(program: dict[str, Any],
                              output_path: str | Path,
                              font_family: str = "Arial") -> dict[str, Any]:
    canvas = program.get("canvas", {})
    width = int(canvas.get("width", 0))
    height = int(canvas.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("program canvas must define positive width and height")

    cells: list[str] = []
    cid = 100
    cells.append(_emit_canvas_anchor_cell(cid, width, height, parent="1"))
    cid += 1

    primitives = program.get("primitives", [])
    for primitive in primitives:
        if primitive.get("type") == "region":
            cells.append(_emit_rect_cell(cid, _as_rect(primitive), parent="2"))
            cid += 1
    for primitive in primitives:
        if primitive.get("type") == "shape":
            cells.append(_emit_native_shape_cell(
                cid, _as_shape(primitive), parent="7"))
            cid += 1
    for primitive in primitives:
        if primitive.get("type") == "edge":
            cells.append(_emit_edge_cell(cid, _as_edge(primitive), parent="3"))
            cid += 1
    for primitive in primitives:
        if primitive.get("type") == "text":
            cells.append(_emit_text_cell(
                cid, _as_text(primitive), parent="4",
                font_family=font_family))
            cid += 1

    drawio = _wrap_drawio(
        cells, width, height,
        overlay_visible=True,
        native_overlay_visible=True,
        include_source_layer=False,
        include_raster_layers=False,
    )
    assert_pure_native_drawio(drawio)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(drawio)
    return {
        "output": str(out),
        "cells": len(cells),
        "width": width,
        "height": height,
    }


def _as_rect(primitive: dict[str, Any]) -> dict[str, Any]:
    style = primitive.get("style", {})
    return {
        "bbox": primitive["bbox"],
        "fill": style.get("fill") or "#eaf2f7",
        "stroke": style.get("stroke") or "#8f9cac",
    }


def _as_shape(primitive: dict[str, Any]) -> dict[str, Any]:
    style = primitive.get("style", {})
    return {
        "bbox": primitive["bbox"],
        "shape": primitive.get("shape", "rectangle"),
        "fill": style.get("fill") or "#d8e6ef",
        "stroke": style.get("stroke") or "#6f8190",
        "direction": style.get("direction"),
    }


def _as_edge(primitive: dict[str, Any]) -> dict[str, Any]:
    style = primitive.get("style", {})
    path = primitive.get("path")
    if not path or len(path) < 2:
        bbox = primitive.get("bbox", [0, 0, 1, 1])
        path = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
    return {
        "orient": "P" if len(path) > 2 else "H",
        "path": path,
        "stroke": style.get("stroke") or "#050505",
        "width": float(style.get("stroke_width") or 1.1),
        "arrow_start": bool(style.get("arrow_start")),
        "arrow_end": bool(style.get("arrow_end")),
        "dashed": bool(style.get("dashed")),
    }


def _as_text(primitive: dict[str, Any]) -> dict[str, Any]:
    style = primitive.get("style", {})
    return {
        "bbox": primitive["bbox"],
        "text": primitive.get("text", ""),
        "font_size": int(style.get("font_size") or 9),
        "bold": bool(style.get("bold")),
        "align": style.get("align", "center"),
        "rotation": style.get("rotation"),
        "font_color": style.get("font_color") or "#050505",
    }


def _emit_text_cell(cid: int,
                    item: dict[str, Any],
                    parent: str = "4",
                    font_family: str = "Arial") -> str:
    x0, y0, x1, y1 = item["bbox"]
    w = max(2.0, x1 - x0)
    h = max(2.0, y1 - y0)
    text = _xml_escape(item.get("text", ""))
    if "\n" in text:
        text = text.replace("\n", "&lt;br&gt;")
    align = item.get("align", "center")
    font_color = item.get("font_color") or "#050505"
    style_parts = [
        "text", "html=1", "strokeColor=none", "fillColor=none",
        f"align={align}", "verticalAlign=middle", "whiteSpace=wrap",
        "rounded=0", f"fontFamily={font_family}",
        f"fontSize={int(item.get('font_size') or 9)}",
        f"fontColor={font_color}",
    ]
    if item.get("rotation") is not None:
        style_parts.append(f"rotation={float(item['rotation']):.1f}")
    if item.get("bold"):
        style_parts.append("fontStyle=1")
    style = ";".join(style_parts) + ";"
    return (
        f'<mxCell id="{cid}" value="{text}" style="{style}" '
        f'vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x0 - 1:.1f}" y="{y0 - 1:.1f}" '
        f'width="{w + 2:.1f}" height="{h + 2:.1f}" as="geometry"/>'
        f'</mxCell>'
    )
