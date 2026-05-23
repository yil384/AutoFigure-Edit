#!/usr/bin/env python3
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


W = 1376
H = 768


def fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def style(parts: dict[str, str | None]) -> str:
    out = []
    for k, v in parts.items():
        out.append(k if v is None else f"{k}={v}")
    return ";".join(out) + ";"


class Diagram:
    def __init__(self) -> None:
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "app.diagrams.net",
                "modified": "2026-05-20T00:00:00.000Z",
                "agent": "codex_native_from_scratch",
                "version": "24.7.0",
                "type": "device",
            },
        )
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": "native_from_scratch", "name": "native"})
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": str(W),
                "dy": str(H),
                "grid": "0",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(W),
                "pageHeight": str(H),
                "math": "0",
                "shadow": "0",
                "background": "#ffffff",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.n = 2

    def cell(self, value: str, st: dict[str, str | None], vertex: bool = True, edge: bool = False) -> ET.Element:
        attrs = {"id": str(self.n), "value": value, "style": style(st), "parent": "1"}
        if vertex:
            attrs["vertex"] = "1"
        if edge:
            attrs.pop("vertex", None)
            attrs["edge"] = "1"
        self.n += 1
        c = ET.SubElement(self.root, "mxCell", attrs)
        return c

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "#ffffff",
        stroke: str = "#8f9cac",
        sw: float = 1.2,
        rounded: bool = False,
        arc: int = 8,
    ) -> ET.Element:
        c = self.cell(
            "",
            {
                "rounded": "1" if rounded else "0",
                "whiteSpace": "wrap",
                "html": "1",
                "fillColor": fill,
                "strokeColor": stroke,
                "strokeWidth": fmt(sw),
                "arcSize": str(arc),
            },
        )
        ET.SubElement(c, "mxGeometry", {"x": fmt(x), "y": fmt(y), "width": fmt(w), "height": fmt(h), "as": "geometry"})
        return c

    def text(
        self,
        txt: str,
        x: float,
        y: float,
        w: float,
        h: float,
        fs: float = 8,
        bold: bool = False,
        align: str = "center",
        color: str = "#111111",
        rotation: int | None = None,
    ) -> ET.Element:
        st = {
            "text": None,
            "html": "1",
            "strokeColor": "none",
            "fillColor": "none",
            "align": align,
            "verticalAlign": "middle",
            "whiteSpace": "wrap",
            "rounded": "0",
            "fontFamily": "Arial",
            "fontSize": fmt(fs),
            "fontColor": color,
            "spacing": "0",
            "spacingTop": "0",
            "spacingBottom": "0",
            "spacingLeft": "0",
            "spacingRight": "0",
        }
        if bold:
            st["fontStyle"] = "1"
        if rotation is not None:
            st["rotation"] = str(rotation)
        c = self.cell(txt, st)
        ET.SubElement(c, "mxGeometry", {"x": fmt(x), "y": fmt(y), "width": fmt(w), "height": fmt(h), "as": "geometry"})
        return c

    def ellipse(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "#ffffff",
        stroke: str = "#58636f",
        sw: float = 1.2,
    ) -> ET.Element:
        c = self.cell(
            "",
            {
                "ellipse": None,
                "whiteSpace": "wrap",
                "html": "1",
                "fillColor": fill,
                "strokeColor": stroke,
                "strokeWidth": fmt(sw),
            },
        )
        ET.SubElement(c, "mxGeometry", {"x": fmt(x), "y": fmt(y), "width": fmt(w), "height": fmt(h), "as": "geometry"})
        return c

    def shape(
        self,
        shp: str,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "#ffffff",
        stroke: str = "#58636f",
        sw: float = 1.2,
        direction: str | None = None,
    ) -> ET.Element:
        st = {
            "shape": shp,
            "whiteSpace": "wrap",
            "html": "1",
            "fillColor": fill,
            "strokeColor": stroke,
            "strokeWidth": fmt(sw),
        }
        if direction:
            st["direction"] = direction
        c = self.cell("", st)
        ET.SubElement(c, "mxGeometry", {"x": fmt(x), "y": fmt(y), "width": fmt(w), "height": fmt(h), "as": "geometry"})
        return c

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#111111",
        sw: float = 1.4,
        arrow: str = "none",
        dashed: bool = False,
        start_arrow: str = "none",
    ) -> ET.Element:
        st = {
            "html": "1",
            "rounded": "0",
            "strokeColor": color,
            "strokeWidth": fmt(sw),
            "endArrow": arrow,
            "endFill": "1" if arrow != "none" else "0",
            "startArrow": start_arrow,
            "startFill": "1" if start_arrow != "none" else "0",
        }
        if dashed:
            st["dashed"] = "1"
            st["dashPattern"] = "4 4"
        c = self.cell("", st, vertex=False, edge=True)
        g = ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(g, "mxPoint", {"x": fmt(x1), "y": fmt(y1), "as": "sourcePoint"})
        ET.SubElement(g, "mxPoint", {"x": fmt(x2), "y": fmt(y2), "as": "targetPoint"})
        return c

    def polyline(
        self,
        points: list[tuple[float, float]],
        color: str = "#111111",
        sw: float = 1.4,
        arrow: str = "none",
        dashed: bool = False,
        start_arrow: str = "none",
        rounded: bool = False,
    ) -> ET.Element:
        if len(points) < 2:
            raise ValueError("polyline needs at least two points")
        st = {
            "html": "1",
            "rounded": "1" if rounded else "0",
            "strokeColor": color,
            "strokeWidth": fmt(sw),
            "endArrow": arrow,
            "endFill": "1" if arrow != "none" else "0",
            "startArrow": start_arrow,
            "startFill": "1" if start_arrow != "none" else "0",
            "edgeStyle": "orthogonalEdgeStyle",
        }
        if dashed:
            st["dashed"] = "1"
            st["dashPattern"] = "4 4"
        c = self.cell("", st, vertex=False, edge=True)
        g = ET.SubElement(c, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(g, "mxPoint", {"x": fmt(points[0][0]), "y": fmt(points[0][1]), "as": "sourcePoint"})
        if len(points) > 2:
            arr = ET.SubElement(g, "Array", {"as": "points"})
            for x, y in points[1:-1]:
                ET.SubElement(arr, "mxPoint", {"x": fmt(x), "y": fmt(y)})
        ET.SubElement(g, "mxPoint", {"x": fmt(points[-1][0]), "y": fmt(points[-1][1]), "as": "targetPoint"})
        return c

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.mxfile).write(path, encoding="UTF-8", xml_declaration=True)


PANEL = "#eaf1fa"
HEADER = "#dbe7f4"
STROKE = "#8f9cac"
BLUE = "#4f90bd"
GREEN = "#8fc77d"
RED = "#d37a73"
YELLOW = "#efc56f"
PURPLE = "#a68ac9"
INK = "#161616"


def panel(d: Diagram, x: float, y: float, w: float, h: float, title: str | None = None, header_h: float = 27) -> None:
    d.rect(x, y, w, h, PANEL, STROKE, 1.3, True, 7)
    if title:
        d.rect(x, y, w, header_h, HEADER, STROKE, 1.0, True, 7)
        d.line(x, y + header_h, x + w, y + header_h, STROKE, 1.0)
        d.text(title, x + 4, y + 4, w - 8, header_h - 6, 10, True)


def module(d: Diagram, x: float, y: float, w: float, h: float, title: str, header_h: float = 32) -> None:
    d.rect(x, y, w, h, PANEL, STROKE, 1.3, True, 7)
    d.rect(x, y, w, header_h, HEADER, STROKE, 1.0, False)
    d.text(title, x + 4, y + 4, w - 8, header_h - 8, 9, True)
    d.line(x, y + header_h, x + w, y + header_h, STROKE, 1.0)


def chip(d: Diagram, x: float, y: float, w: float = 42, h: float = 42, label: str = "") -> None:
    for i in range(5):
        d.line(x - 6, y + 6 + i * 7, x, y + 6 + i * 7, "#3d4145", 1.4)
        d.line(x + w, y + 6 + i * 7, x + w + 6, y + 6 + i * 7, "#3d4145", 1.4)
    for i in range(4):
        d.line(x + 7 + i * 8, y - 6, x + 7 + i * 8, y, "#3d4145", 1.4)
        d.line(x + 7 + i * 8, y + h, x + 7 + i * 8, y + h + 6, "#3d4145", 1.4)
    d.rect(x, y, w, h, "#30373c", "#30373c", 1.0, True, 6)
    d.ellipse(x + 11, y + 9, w - 22, h - 18, "#b9d8e8", "#244a62", 1.0)
    if label:
        d.text(label, x + 4, y + h / 2 - 8, w - 8, 16, 13, True, color="#ffffff")


def robot(d: Diagram, x: float, y: float) -> None:
    d.rect(x + 16, y + 20, 43, 45, "#d7e6ec", "#5a6a73", 1.5, True, 8)
    d.rect(x + 21, y + 4, 34, 25, "#d7e6ec", "#5a6a73", 1.5, True, 8)
    d.rect(x + 27, y + 11, 23, 10, "#223843", "#223843", 1.0, True, 4)
    d.ellipse(x + 32, y + 14, 3, 3, "#ffffff", "#ffffff")
    d.ellipse(x + 43, y + 14, 3, 3, "#ffffff", "#ffffff")
    d.line(x + 38, y + 4, x + 38, y - 6, "#5a6a73", 1.2)
    d.ellipse(x + 34, y - 11, 8, 8, "#d7e6ec", "#5a6a73", 1.2)
    d.rect(x + 27, y + 38, 22, 10, "#ffffff", "#5a6a73", 1.0, True, 3)
    d.line(x + 16, y + 36, x + 2, y + 28, "#5a6a73", 1.4)
    d.line(x + 59, y + 36, x + 72, y + 26, "#5a6a73", 1.4)
    d.line(x + 22, y + 65, x + 18, y + 76, "#5a6a73", 1.4)
    d.line(x + 52, y + 65, x + 56, y + 76, "#5a6a73", 1.4)
    d.ellipse(x + 15, y + 74, 7, 7, "#d7e6ec", "#5a6a73", 1.2)
    d.ellipse(x + 53, y + 74, 7, 7, "#d7e6ec", "#5a6a73", 1.2)


def tanner_blocks(d: Diagram, x: float, y: float) -> None:
    colors = [RED, GREEN, YELLOW, BLUE, "#c9d870", PURPLE, "#d59774", "#f0d076"]
    positions = [(8, 0), (31, 0), (54, 12), (0, 26), (23, 26), (46, 26), (69, 26), (31, 48)]
    for i, (px, py) in enumerate(positions):
        d.rect(x + px, y + py, 21, 18, colors[i % len(colors)], "#59656d", 0.8, True, 3)
    for sx, sy, tx, ty in [(18, 18, 18, 26), (41, 18, 41, 26), (64, 30, 64, 44), (78, 44, 78, 61), (41, 44, 41, 67)]:
        d.line(x + sx, y + sy, x + tx, y + ty, "#333333", 1.0)
    d.line(x + 2, y + 24, x + 88, y + 24, "#59656d", 0.7)


def graph_icon(d: Diagram, x: float, y: float, scale: float = 1.0) -> None:
    pts = [(0, 18), (22, 4), (44, 12), (16, 38), (46, 42), (30, 25)]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 5), (5, 2), (5, 4), (3, 4), (1, 5)]:
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        d.line(x + x1 * scale, y + y1 * scale, x + x2 * scale, y + y2 * scale, "#64727c", 1.0)
    for px, py in pts:
        d.ellipse(x + px * scale - 4, y + py * scale - 4, 8, 8, "#d8e8f0", "#64727c", 1.0)


def colored_graph_icon(d: Diagram, x: float, y: float, scale: float = 1.0) -> None:
    pts = [(0, 18), (23, 5), (46, 13), (15, 39), (48, 43), (31, 26)]
    colors = [BLUE, RED, YELLOW, RED, GREEN, PURPLE]
    for a, b in [(0, 1), (1, 2), (0, 3), (3, 5), (5, 2), (5, 4), (3, 4), (1, 5)]:
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        d.line(x + x1 * scale, y + y1 * scale, x + x2 * scale, y + y2 * scale, "#5f6870", 1.0)
    for i, (px, py) in enumerate(pts):
        d.rect(x + px * scale - 4, y + py * scale - 4, 8, 8, colors[i], "#5f6870", 0.9, True, 2)


def funnel(d: Diagram, x: float, y: float) -> None:
    d.ellipse(x + 4, y + 1, 44, 9, "#d7e6f0", "#5e6d76", 1.0)
    d.line(x + 6, y + 7, x + 24, y + 30, "#5e6d76", 1.3)
    d.line(x + 46, y + 7, x + 29, y + 30, "#5e6d76", 1.3)
    d.line(x + 24, y + 30, x + 24, y + 63, "#5e6d76", 1.3)
    d.line(x + 29, y + 30, x + 29, y + 63, "#5e6d76", 1.3)
    d.line(x + 24, y + 63, x + 29, y + 63, "#5e6d76", 1.3)
    d.line(x + 10, y + 8, x + 41, y + 8, "#6f8291", 0.9)


def document_chart(d: Diagram, x: float, y: float) -> None:
    d.rect(x, y, 34, 54, "#f5f7fa", "#909aa3", 1.1, True, 4)
    for i in range(4):
        d.line(x + 7, y + 10 + i * 8, x + 27, y + 10 + i * 8, "#8a9299", 1.0)
    for i, h in enumerate([12, 22, 32]):
        d.rect(x + 48 + i * 17, y + 54 - h, 10, h, [HEADER, "#a9c9df", BLUE][i], "#71818c", 0.8)
    d.line(x + 44, y + 54, x + 96, y + 54, "#71818c", 1.0)
    d.line(x + 44, y + 18, x + 44, y + 54, "#71818c", 1.0)
    d.ellipse(x + 58, y + 4, 9, 9, "#ffffff", "#71818c")
    d.line(x + 62, y + 0, x + 70, y - 10, BLUE, 1.6, "classic")


def seed_codes(d: Diagram, x: float, y: float) -> None:
    panel(d, x, y, 96, 84, None)
    d.text("Seed codes", x + 8, y + 8, 80, 13, 8, True)
    for i in range(3):
        d.rect(x + 10, y + 28 + i * 19, 28, 13, "#ffffff", "#87939c", 1.0, True, 3)
        d.line(x + 14, y + 34 + i * 19, x + 33, y + 34 + i * 19, "#87939c", 1.0)
    d.line(x + 38, y + 54, x + 50, y + 54, "#87939c", 1.0, "classic")
    colored_graph_icon(d, x + 55, y + 30, 0.78)


def neutral_trap(d: Diagram, x: float, y: float) -> None:
    for ang in [(0, 0, 56, 38), (56, 0, 0, 38), (6, 18, 50, 18), (28, -8, 28, 46)]:
        d.line(x + ang[0], y + ang[1], x + ang[2], y + ang[3], "#7e9ab3", 5.0)
        d.line(x + ang[0], y + ang[1], x + ang[2], y + ang[3], "#5c7285", 1.0)
    d.ellipse(x + 22, y + 12, 14, 14, RED, "#6f6f6f", 1.0)


def pie_noise(d: Diagram, x: float, y: float) -> None:
    d.ellipse(x, y, 58, 58, BLUE, "#58636f", 1.1)
    d.shape("triangle", x + 2, y + 10, 34, 38, RED, "#58636f", 1.0, "east")
    d.shape("triangle", x + 8, y + 2, 26, 26, "#d7dbe0", "#58636f", 1.0, "south")
    d.text("X/Z", x + 25, y + 23, 24, 12, 8, True, color="#ffffff")


def mini_sliders(d: Diagram, x: float, y: float) -> None:
    for i, off in enumerate([0, 17, 34]):
        d.line(x, y + off, x + 82, y + off, "#86a8c2", 3.0)
        d.ellipse(x + [32, 55, 24][i], y + off - 5, 10, 10, BLUE, BLUE)


def switch_circuit(d: Diagram, x: float, y: float) -> None:
    d.line(x, y + 42, x + 18, y + 42, INK, 1.7)
    d.ellipse(x + 18, y + 36, 11, 11, BLUE, "#5c6268", 1.0)
    taps = [(50, 22, GREEN), (50, 43, YELLOW), (50, 65, RED)]
    for tx, ty, col in taps:
        d.line(x + 29, y + 42, x + tx, y + ty + 5, INK, 1.4)
        d.ellipse(x + tx, y + ty, 11, 11, col, "#5c6268", 1.0)
    d.line(x + 70, y + 43, x + 93, y + 43, INK, 1.5)
    d.line(x + 93, y + 22, x + 93, y + 65, INK, 1.5)
    d.line(x + 76, y + 22, x + 93, y + 22, INK, 1.5)
    d.line(x + 76, y + 65, x + 93, y + 65, INK, 1.5)
    d.line(x + 93, y + 43, x + 118, y + 43, INK, 1.7, "classic")


def qft_target(d: Diagram, x: float, y: float) -> None:
    d.line(x + 16, y + 60, x + 72, y + 60, "#324855", 1.4, "classic")
    for i in range(2):
        d.line(x + 14, y + 58 - i * 6, x + 68, y + 38 - i * 6, "#6e86a5", 1.2)
        d.line(x + 14, y + 58 - i * 6, x + 68, y + 84 - i * 6, "#6e86a5", 1.2)
    for i in range(5):
        d.line(x + 18 + i * 12, y + 100, x + 52 + i * 12, y + 134, "#6e8a8c", 1.1)
        d.line(x + 18 + i * 12, y + 134, x + 52 + i * 12, y + 100, "#6e8a8c", 1.1)


def gaussian_primitives(d: Diagram, x: float, y: float) -> None:
    d.shape("triangle", x + 7, y + 30, 32, 42, "#9fc2df", "#5d7588", 1.0, "north")
    d.shape("triangle", x + 58, y + 28, 36, 46, "#d5a0ad", "#8a5963", 1.0, "north")
    d.text("Gaussian", x + 0, y + 75, 48, 13, 8)
    d.text("Non-<br>Gauge<br>coupling", x + 53, y + 74, 48, 32, 8, True)


def cv_primitives(d: Diagram, x: float, y: float) -> None:
    d.ellipse(x + 8, y + 34, 28, 28, "#d6a3d3", "#7d6b85", 1.0)
    d.line(x + 22, y + 48, x + 31, y + 36, "#7d6b85", 1.2, "classic")
    d.shape("rhombus", x + 58, y + 37, 24, 24, "#eef6fb", "#6f8b9a", 1.2)
    d.text("Bloch-<br>Messiah", x + 4, y + 70, 42, 25, 6)
    d.text("Interfero-<br>meters", x + 53, y + 70, 48, 25, 6)
    graph_icon(d, x + 10, y + 100, 0.65)
    d.line(x + 60, y + 112, x + 92, y + 92, "#66808a", 1.2)
    d.line(x + 60, y + 92, x + 92, y + 112, "#66808a", 1.2)
    d.text("Non-<br>Gaussian", x + 4, y + 130, 44, 26, 6)
    d.text("Gadgets", x + 58, y + 130, 42, 12, 6)


def chip_phic(d: Diagram, x: float, y: float) -> None:
    chip(d, x, y, 45, 45, "Phic")


def shield(d: Diagram, x: float, y: float) -> None:
    d.shape("hexagon", x, y, 48, 52, BLUE, "#486a7e", 1.3)
    d.text("✓", x + 10, y + 11, 28, 28, 22, True, color="#ffffff")


def circuit(d: Diagram, x: float, y: float) -> None:
    for i in range(4):
        d.line(x, y + i * 18, x + 95, y + i * 18, "#1f262b", 1.1)
    d.rect(x + 22, y - 5, 28, 64, "#ffffff", "#7b8790", 1.0, False)
    for i in range(4):
        d.text("C", x + 30, y + i * 18 - 5, 12, 10, 7)
    d.shape("triangle", x - 8, y + 43, 18, 14, "#ffffff", "#7b8790", 1.0, "east")
    for i in range(3):
        d.ellipse(x + 60, y + i * 18 - 3, 6, 6, "#ffffff", "#1f262b", 1.0)


def brain_head(d: Diagram, x: float, y: float) -> None:
    d.shape("actor", x + 12, y + 4, 30, 45, "#5f9cc1", "#4f7c98", 1.1)
    for i, (cx, cy) in enumerate([(25, 9), (30, 11), (22, 14), (34, 16), (27, 19), (20, 21), (32, 23)]):
        d.ellipse(x + cx, y + cy, 5, 5, "#d7edf7", "#d7edf7", 0.5)
        if i:
            d.line(x + 27, y + 18, x + cx + 2, y + cy + 2, "#d7edf7", 0.8)


def small_hypergraph(d: Diagram, x: float, y: float) -> None:
    pts = [(0, 0), (34, 2), (16, 25), (52, 26)]
    for a, b in [(0, 2), (2, 1), (1, 3), (0, 1), (2, 3)]:
        ax, ay = pts[a]
        bx, by = pts[b]
        d.line(x + ax + 5, y + ay + 5, x + bx + 5, y + by + 5, "#6d7a84", 1.0)
    for px, py in pts:
        d.rect(x + px, y + py, 10, 10, "#b6cde0", "#6d7a84", 0.8, True, 2)


def molecule_icon(d: Diagram, x: float, y: float, scale: float = 1.0) -> None:
    pts = [(26, 18, RED), (10, 31, GREEN), (43, 31, YELLOW), (26, 46, "#b5a1b8"), (26, 5, "#85b7b5")]
    for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (2, 3)]:
        ax, ay, _ = pts[a]
        bx, by, _ = pts[b]
        d.line(x + ax * scale, y + ay * scale, x + bx * scale, y + by * scale, "#84909a", 0.9)
    for px, py, fill in pts:
        d.ellipse(x + px * scale - 4, y + py * scale - 4, 8, 8, fill, "#84909a", 0.8)


def lattice_cluster(d: Diagram, x: float, y: float, scale: float = 1.0) -> None:
    for row in range(3):
        for col in range(4):
            px = x + (8 + col * 10 + (row % 2) * 5) * scale
            py = y + (10 + row * 11) * scale
            d.ellipse(px, py, 5 * scale, 5 * scale, ["#d9d98e", "#9fca82", "#d7a36f"][(row + col) % 3], "#839282", 0.6)
            if col:
                d.line(px - 8 * scale, py + 2.5 * scale, px, py + 2.5 * scale, "#839282", 0.7)
            if row:
                d.line(px - 3 * scale, py - 8 * scale, px, py, "#839282", 0.7)


def spiky_particle(d: Diagram, x: float, y: float) -> None:
    d.shape("cloud", x + 1, y + 8, 36, 25, "#9eaebe", "#7a8792", 1.0)
    for sx, sy, tx, ty in [(18, 0, 18, 8), (3, 8, 10, 12), (34, 8, 29, 13), (7, 34, 12, 29), (31, 34, 27, 29)]:
        d.line(x + sx, y + sy, x + tx, y + ty, "#7a8792", 1.0)


def magnifier_icon(d: Diagram, x: float, y: float) -> None:
    d.ellipse(x, y, 24, 24, "#a7c9d7", "#6b818b", 1.0)
    d.line(x + 18, y + 19, x + 32, y + 32, "#6b818b", 1.5)


def chat_bubbles(d: Diagram, x: float, y: float) -> None:
    d.rect(x + 1, y, 38, 24, "#e9f6fb", "#507084", 1.1, True, 6)
    d.shape("triangle", x + 7, y + 18, 12, 12, "#e9f6fb", "#507084", 1.0, "south")
    d.rect(x + 27, y + 13, 42, 26, BLUE, "#507084", 1.1, True, 6)
    d.shape("triangle", x + 50, y + 32, 12, 11, BLUE, "#507084", 1.0, "south")


def python_logo(d: Diagram, x: float, y: float) -> None:
    d.rect(x + 4, y, 24, 20, "#3676aa", "#3676aa", 1.0, True, 5)
    d.rect(x + 18, y + 14, 24, 20, "#f3c53c", "#f3c53c", 1.0, True, 5)
    d.ellipse(x + 11, y + 6, 3, 3, "#ffffff", "#ffffff", 0.5)
    d.ellipse(x + 32, y + 23, 3, 3, "#ffffff", "#ffffff", 0.5)


def hand_sheet(d: Diagram, x: float, y: float) -> None:
    d.rect(x, y, 42, 30, "#f5f7fa", "#8b969e", 1.0, True, 3)
    for i in range(3):
        d.line(x + 8, y + 9 + i * 7, x + 34, y + 9 + i * 7, "#8b969e", 0.8)
    d.text("layout", x + 7, y + 5, 28, 8, 4)


def workload_icons(d: Diagram, x: float, y: float) -> None:
    molecule_icon(d, x + 25, y + 10, 0.8)
    lattice_cluster(d, x + 116, y + 12, 0.75)
    spiky_particle(d, x + 205, y + 14)
    magnifier_icon(d, x + 290, y + 18)
    d.text("...", x + 350, y + 17, 36, 18, 15, True)


def zx_graph(d: Diagram, x: float, y: float) -> None:
    nodes = [
        (20, 30, "2X", GREEN),
        (70, 30, "2X", RED),
        (20, 72, "2X", GREEN),
        (70, 72, "2X", RED),
        (45, 105, "ZX", GREEN),
    ]
    edges = [(0, 3), (1, 2), (0, 4), (1, 4), (2, 4), (3, 4)]
    for a, b in edges:
        ax, ay, _, _ = nodes[a]
        bx, by, _, _ = nodes[b]
        d.line(x + ax + 13, y + ay + 13, x + bx + 13, y + by + 13, "#222222", 1.4)
    for px, py, lab, fill in nodes:
        d.ellipse(x + px, y + py, 24, 24, fill, "#5d6a5f", 1.0)
        d.text(lab, x + px + 2, y + py + 6, 20, 12, 7, True)


def benchmark_box(d: Diagram, x: float, y: float) -> None:
    for i in range(3):
        d.line(x + 18, y + 48 + i * 9, x + 80, y + 48 + i * 9, "#6b86a0", 1.0)
        d.line(x + 30 + i * 18, y + 40, x + 30 + i * 18, y + 75, "#6b86a0", 1.0)
    d.text("+", x + 70, y + 50, 18, 18, 16, True)
    d.rect(x + 88, y + 40, 44, 44, "#f5f7fa", "#7f8a93", 1.0, False)
    for i in range(2):
        for j in range(2):
            d.rect(x + 94 + i * 18, y + 46 + j * 16, 13, 12, "#d7e3f0", "#9aa7b1", 0.7)


def charts(d: Diagram, x: float, y: float) -> None:
    for k in range(3):
        yy = y + k * 95
        d.line(x, yy + 75, x + 115, yy + 75, INK, 1.2, "classic")
        d.line(x, yy + 75, x, yy, INK, 1.2, "classic")
        if k < 2:
            for i, h in enumerate([20, 35, 56]):
                d.rect(x + 26 + i * 28, yy + 75 - h, 16, h, ["#d8e2ec", "#9eb8d0", BLUE][i], "#70808d", 0.8)
            d.line(x + 16, yy + 60, x + 103, yy + 10, "#6d8dad", 1.8)
            if k == 1:
                d.line(x + 15, yy + 65, x + 103, yy + 35, "#6c9a6b", 1.4)
        else:
            d.line(x + 12, yy + 66, x + 104, yy + 15, "#6c9a6b", 1.6)
            d.line(x + 16, yy + 70, x + 104, yy + 42, "#6d8dad", 1.2)
            d.line(x + 16, yy + 72, x + 104, yy + 62, "#b06d6d", 1.2, dashed=True)


def figure1(d: Diagram) -> None:
    robot(d, 12, 86)
    d.text("LLM-driven<br>Exploration", 18, 148, 70, 32, 10, True)
    d.line(82, 121, 116, 121, INK, 1.5, "classic")
    d.text("Propose partially<br>assembled,<br>colored code", 104, 55, 112, 51, 10, True)
    tanner_blocks(d, 112, 106)
    d.text("Features", 105, 166, 52, 12, 8)
    d.text("Features", 170, 166, 52, 12, 8)
    d.text("Tanner-graph", 119, 184, 86, 13, 8)
    d.line(200, 113, 245, 113, INK, 1.2, "classic")
    d.line(245, 127, 200, 127, INK, 1.2, "classic")
    chip(d, 252, 92, 43, 43)
    d.text("RL-based<br>Exploitation", 238, 148, 82, 32, 10, True)
    d.line(296, 121, 338, 121, INK, 1.5, "classic")
    panel(d, 340, 66, 184, 114, None)
    d.text("Filter Funnel", 386, 70, 94, 16, 10, True)
    funnel(d, 365, 93)
    d.line(421, 110, 450, 110, INK, 1.4, "classic")
    graph_icon(d, 460, 91, 1.0)
    d.text("Algebraic<br>Validity<br>Check", 350, 130, 74, 45, 10)
    d.text("GNN<br>Surrogate<br>Filtering", 443, 130, 76, 45, 10)
    panel(d, 386, 194, 99, 36, None)
    d.text("Classical<br>Metrics Ranker", 396, 199, 80, 23, 7)
    d.line(433, 234, 433, 204, INK, 1.2, "classic")
    d.polyline([(362, 177), (362, 194), (386, 194)], STROKE, 1.0, dashed=True)
    d.polyline([(454, 180), (454, 194), (485, 194)], STROKE, 1.0, dashed=True)
    d.line(524, 122, 557, 122, INK, 1.5, "classic")
    panel(d, 557, 102, 112, 126, None)
    d.text("Validated Novel<br>Code Families", 564, 107, 100, 34, 10, True)
    document_chart(d, 579, 148)
    d.text("Improved<br>Surgery Metrics", 568, 194, 96, 31, 10, True)
    seed_codes(d, 20, 215)
    d.line(60, 215, 60, 181, INK, 1.4, "classic")
    d.polyline([(118, 257), (611, 257), (611, 228)], INK, 1.4)
    d.polyline([(162, 247), (162, 203)], INK, 1.3, "classic")
    d.polyline([(162, 247), (275, 247), (275, 181)], INK, 1.3, "classic")
    d.text("Ranked Candidates & Surgery Scores", 195, 16, 220, 16, 11, True)
    d.line(62, 35, 550, 35, INK, 1.2)
    d.line(62, 35, 62, 84, INK, 1.2, "classic")
    d.line(548, 35, 548, 92, INK, 1.2, "classic")
    d.line(273, 35, 273, 89, INK, 1.2, "classic")
    d.text("VT Team<br>(Full ZX-Calculus<br>Surgery Analysis)", 535, 45, 105, 43, 8, True)
    d.ellipse(576, 8, 16, 16, "#4e5155", "#4e5155")
    d.rect(570, 28, 28, 16, "#4e5155", "#4e5155", 1.0, True, 5)
    d.ellipse(593, 34, 8, 8, "#ffffff", "#4e5155", 1.0)
    d.line(597, 38, 604, 43, "#4e5155", 1.0)
    d.text("Ranked Candidates & Surgery Scores", 284, 251, 220, 13, 7)
    d.line(20, 316, 668, 316, INK, 1.4, "classic")
    d.line(247, 306, 247, 326, INK, 1.0)
    d.line(479, 306, 479, 326, INK, 1.0)
    d.ellipse(227, 288, 18, 18, "#ffffff", "#8f9cac", 1.1)
    d.ellipse(459, 288, 18, 18, "#ffffff", "#8f9cac", 1.1)
    d.text("Weekly<br>batching", 249, 280, 54, 31, 9, True)
    d.text("Weekly<br>batching", 480, 280, 56, 31, 9, True)
    d.text("Months", 22, 321, 45, 13, 10)
    d.text("1-3", 126, 321, 45, 13, 10, True)
    d.text("4-6", 344, 321, 45, 13, 10, True)
    d.text("7-9", 576, 321, 45, 13, 10, True)
    d.text("Figure 1: AI-Enabled Co-Design of Quantum LDPC Codes and Fault-Tolerant Interfaces.", 10, 347, 660, 22, 14, True, "left")


def figure2(d: Diagram) -> None:
    panel(d, 713, 23, 200, 284, "Noise analysis/Input", 27)
    d.text("Neutral<br>atom/trap", 735, 96, 62, 35, 9, True)
    neutral_trap(d, 730, 138)
    d.text("Neutral<br>atom/ion trap", 728, 184, 78, 35, 8, True)
    panel(d, 825, 85, 77, 118, None)
    d.text("Noise<br>fingerprint<br>extraction", 835, 90, 58, 45, 10, True)
    pie_noise(d, 835, 139)
    d.text("X/Z bias", 837, 194, 60, 13, 10, True)
    d.line(790, 164, 825, 164, INK, 1.4, "classic")
    d.line(902, 164, 924, 164, INK, 1.4, "classic")
    panel(d, 924, 23, 246, 291, "LLM-Driven Recipe Composition", 27)
    d.rect(937, 75, 226, 232, "none", INK, 1.2, True, 7)
    chip(d, 958, 78, 42, 42)
    d.rect(960, 82, 38, 34, "#d6b4de", "#8c6a94", 1.0, True, 4)
    d.text("LLM", 968, 93, 24, 12, 8, True)
    d.text("Propose Algebraic<br>code recipes", 1010, 62, 108, 31, 8, True)
    d.polyline([(1000, 105), (1130, 105), (1130, 148)], INK, 1.4, "classic")
    d.text("Modulate<br>dX, dZ targets", 1002, 125, 100, 32, 10, True)
    mini_sliders(d, 1014, 165)
    chip(d, 1120, 142, 42, 42)
    d.text("RL Policy", 1104, 193, 58, 14, 9)
    graph_icon(d, 1020, 229, 0.9)
    d.text("Candidate<br>codes", 952, 232, 64, 30, 8)
    d.text("Noise<br>annotation", 1090, 246, 68, 30, 8)
    d.text("GNN Surrogate<br>(Fast Feedback)", 998, 272, 100, 31, 10, True)
    d.polyline([(958, 118), (958, 169), (1005, 169)], INK, 1.3, "classic")
    d.polyline([(946, 92), (937, 92), (937, 239), (965, 239)], INK, 1.2, "classic")
    d.line(1095, 169, 1120, 169, INK, 1.2, "classic")
    d.line(1118, 173, 1095, 173, INK, 1.2, "classic")
    d.line(1051, 204, 1051, 226, INK, 1.3, "classic")
    d.line(1055, 226, 1055, 204, INK, 1.3, "classic")
    d.polyline([(1118, 246), (1095, 246), (1075, 246)], INK, 1.4, "classic")
    d.line(1170, 168, 1183, 168, INK, 1.4, "classic")
    panel(d, 1183, 24, 180, 284, None)
    d.text("Switchable<br>Code<br>Families", 1192, 99, 74, 45, 10, True)
    switch_circuit(d, 1195, 132)
    d.polyline([(1268, 156), (1288, 130), (1298, 130)], INK, 1.2, "classic")
    d.line(1268, 175, 1298, 175, INK, 1.2, "classic")
    d.polyline([(1268, 199), (1288, 220), (1298, 220)], INK, 1.2, "classic")
    d.text("Switching<br>circuits", 1287, 109, 62, 31, 9)
    d.text("Decoders", 1277, 162, 74, 15, 9, True)
    d.text("Trotter<br>Ordering", 1265, 204, 86, 37, 10, True)
    d.text("Phase I: LLM+RL pipeline", 1192, 286, 160, 15, 10)
    d.text("Iterative GNN Retraining", 983, 320, 154, 15, 10, True)
    d.text("Figure 2: ML-Driven Codesign of QEC for Strongly Correlated Spin Dynamics.", 742, 347, 620, 22, 14, True, "left")


def figure3(d: Diagram) -> None:
    module(d, 18, 442, 92, 184, "QFT<br>Simulation<br>Target", 50)
    qft_target(d, 27, 465)
    brain_head(d, 154, 391)
    module(d, 130, 442, 113, 185, "Physics-Informed<br>Agent", 36)
    d.text("Select Truncation", 140, 488, 92, 16, 9, True)
    d.line(185, 505, 185, 524, INK, 1.3, "classic")
    d.rect(139, 519, 98, 104, PANEL, STROKE, 1.1, True, 6)
    d.text("Sub-unitaries", 151, 523, 74, 13, 8, True)
    gaussian_primitives(d, 141, 528)
    small_hypergraph(d, 295, 401)
    module(d, 263, 442, 112, 184, "Decomposition<br>Agent", 36)
    d.rect(270, 482, 98, 141, PANEL, STROKE, 1.1, True, 6)
    d.text("CV Primitives", 279, 486, 82, 16, 9, True)
    cv_primitives(d, 272, 491)
    module(d, 391, 442, 76, 184, "Hardware<br>Agent", 36)
    chip_phic(d, 407, 518)
    d.text("Scheduling", 394, 570, 70, 13, 8, True)
    module(d, 483, 442, 77, 184, "Verification<br>Agent", 36)
    d.text("Iterative<br>feedback", 494, 491, 56, 30, 8, True)
    shield(d, 500, 525)
    d.text("Error", 505, 574, 45, 15, 10, True, color="#b33838")
    circuit(d, 586, 505)
    d.text("Executable<br>CV Quantum<br>Circuit", 586, 446, 76, 50, 9, True)
    panel(d, 276, 632, 82, 69, None)
    d.ellipse(302, 642, 34, 20, HEADER, "#8aa0b0", 1.0)
    d.text("Model<br>Distillation", 288, 665, 60, 31, 10, True)
    d.line(110, 540, 130, 540, INK, 1.4, "classic")
    d.line(243, 540, 263, 540, INK, 1.4, "classic")
    d.line(375, 540, 391, 540, INK, 1.4, "classic")
    d.line(467, 540, 483, 540, INK, 1.4, "classic")
    d.line(483, 547, 467, 547, "#b94d4d", 1.2, "classic")
    d.line(560, 540, 586, 540, INK, 1.4, "classic")
    d.line(662, 540, 682, 540, INK, 1.4, "classic")
    d.polyline([(185, 627), (185, 667), (276, 667)], INK, 1.2, "classic")
    d.polyline([(358, 667), (521, 667), (521, 626)], INK, 1.2, "classic")
    d.polyline([(540, 571), (540, 600), (514, 600), (514, 575)], "#b94d4d", 1.1, "classic", rounded=True)
    d.text("Fast-track", 414, 650, 68, 13, 8, True)
    d.text("For Routine Circuit Families", 367, 672, 154, 12, 8)
    d.text("Figure 3: AI-Enhanced Continuous-Variable Quantum Computing for Quantum Field", 12, 721, 700, 23, 14, True, "left")
    d.text("Theory.", 12, 742, 80, 22, 14, True, "left")


def figure4(d: Diagram) -> None:
    panel(d, 768, 400, 302, 85, "Input Workloads", 26)
    workload_icons(d, 775, 420)
    d.text("Quantum Chemistry<br>(FeMoco)", 770, 454, 102, 28, 7)
    d.text("Hubbard<br>Model", 888, 454, 58, 28, 7)
    d.line(830, 485, 830, 506, INK, 1.3, "classic")
    d.line(900, 485, 900, 506, INK, 1.3, "classic")
    d.line(964, 485, 964, 506, INK, 1.3, "classic")
    chat_bubbles(d, 724, 553)
    d.text("Natural-<br>Language<br>Frontend", 716, 596, 76, 38, 8, True)
    graph_icon(d, 818, 560, 0.9)
    d.text("Graph<br>Diffusion<br>Model", 812, 596, 76, 38, 8, True)
    d.line(793, 575, 816, 575, INK, 1.4, "classic")
    d.line(868, 575, 881, 575, INK, 1.4, "classic")
    panel(d, 881, 508, 102, 141, None)
    d.text("ZX Diagram<br>Synthesis", 896, 512, 70, 32, 10, True)
    zx_graph(d, 892, 520)
    d.line(983, 578, 1015, 578, INK, 1.4, "classic")
    panel(d, 1015, 537, 69, 61, None)
    d.rect(1030, 550, 38, 30, "#eef6fb", "#6b8794", 1.1, True, 5)
    d.text("AI", 1038, 556, 24, 15, 12, True)
    d.text("ZX-FT Protocol<br>Synthesis", 1016, 592, 76, 31, 8, True)
    panel(d, 1110, 414, 115, 255, "Benchmarking", 27)
    benchmark_box(d, 1117, 435)
    d.text("Qiskit+Surface<br>Code", 1124, 493, 94, 30, 8, True)
    chip(d, 1150, 532, 42, 42, "RL")
    d.text("RL Circuit Token", 1118, 583, 92, 13, 9, True)
    python_logo(d, 1128, 612)
    hand_sheet(d, 1181, 612)
    d.text("+", 1165, 621, 16, 16, 14, True)
    d.text("PyZX+Hand Layout", 1118, 650, 100, 14, 9, True)
    d.line(1084, 578, 1110, 578, INK, 1.4, "classic")
    d.polyline([(1084, 552), (1097, 552), (1097, 475), (1110, 475)], INK, 1.2, "classic")
    d.polyline([(1084, 606), (1097, 606), (1097, 633), (1110, 633)], INK, 1.2, "classic")
    charts(d, 1268, 407)
    d.text("Sample<br>Efficiency", 1238, 430, 46, 42, 8, rotation=270)
    d.text("Physical-Qubit<br>Overhead", 1228, 535, 58, 52, 8, rotation=270)
    d.text("Soundness<br>fnc", 1229, 625, 54, 45, 8, rotation=270)
    d.text("Scaling", 1292, 694, 55, 13, 9, True)
    panel(d, 725, 675, 500, 29, None)
    d.text("Key:", 732, 682, 34, 14, 10, True)
    d.ellipse(770, 683, 15, 12, "#95a2ab", "#6f7b84", 1.0)
    d.rect(783, 687, 26, 4, "#95a2ab", "#6f7b84", 1.0, False)
    d.line(801, 691, 801, 695, "#6f7b84", 1.0)
    d.line(806, 691, 806, 695, "#6f7b84", 1.0)
    d.line(815, 689, 838, 689, "#6f7b84", 1.2)
    d.text("Input Metrics", 838, 681, 88, 13, 8)
    graph_icon(d, 932, 678, 0.45)
    d.line(980, 689, 1000, 689, "#87a887", 1.2)
    d.text("Physical-Qubit Brand", 1000, 681, 112, 13, 8)
    d.line(1122, 689, 1144, 689, "#b06d6d", 1.2, dashed=True)
    d.text("Soundness", 1150, 681, 70, 13, 8)
    d.text("Figure 4: ZX Calculus as a Unifying Language for AI-Driven, Application-Aware QEC.", 718, 721, 645, 23, 14, True, "left")


def main() -> None:
    d = Diagram()
    figure1(d)
    figure2(d)
    figure3(d)
    figure4(d)
    d.save(Path("outputs/png_to_svg/codex_drawio_native_from_scratch.drawio"))


if __name__ == "__main__":
    main()
