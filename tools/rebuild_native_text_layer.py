#!/usr/bin/env python3
"""Rebuild a visible native-text layer on top of a pure-vector drawio base.

This is intentionally narrow: it does not embed or reference the source image.
It starts from the high-fidelity vector drawio, removes small stencil cells in
known text regions, then appends editable draw.io text cells copied from the
component/OCR drawio with conservative corrections.
"""

from __future__ import annotations

import argparse
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image, ImageStat
except Exception:  # pragma: no cover - optional for pure XML operation
    Image = None
    ImageStat = None


STYLE_SEP = ";"


def _f(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _bbox(geom: ET.Element) -> tuple[float, float, float, float]:
    x = _f(geom.get("x"))
    y = _f(geom.get("y"))
    w = _f(geom.get("width"))
    h = _f(geom.get("height"))
    return x, y, w, h


def _intersect(a: tuple[float, float, float, float],
               b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def _pad(b: tuple[float, float, float, float], px: float,
         py: float | None = None) -> tuple[float, float, float, float]:
    if py is None:
        py = px
    x, y, w, h = b
    return x - px, y - py, w + 2 * px, h + 2 * py


def _style_dict(style: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for part in style.split(STYLE_SEP):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
        else:
            result[part] = None
    return result


def _style_string(parts: dict[str, str | None]) -> str:
    out: list[str] = []
    for k, v in parts.items():
        if v is None:
            out.append(k)
        else:
            out.append(f"{k}={v}")
    return ";".join(out) + ";"


def _set_style(cell: ET.Element, **updates: str | None) -> None:
    parts = _style_dict(cell.get("style", ""))
    for k, v in updates.items():
        if v is None:
            parts.pop(k, None)
        else:
            parts[k] = v
    cell.set("style", _style_string(parts))


def _set_geom(cell: ET.Element, **updates: float) -> None:
    geom = cell.find("mxGeometry")
    if geom is None:
        geom = ET.SubElement(cell, "mxGeometry", {"as": "geometry"})
    for k, v in updates.items():
        geom.set(k, f"{v:.1f}".rstrip("0").rstrip("."))


def _fit_text_width(cell: ET.Element) -> None:
    value = cell.get("value", "")
    if not value:
        return
    geom = cell.find("mxGeometry")
    if geom is None:
        return
    style = _style_dict(cell.get("style", ""))
    try:
        font_size = float(style.get("fontSize") or 8)
    except (TypeError, ValueError):
        font_size = 8.0
    plain_lines = [re.sub(r"<[^>]+>", "", line) for line in re.split(r"<br\s*/?>", value)]
    max_chars = max((len(line) for line in plain_lines), default=0)
    if max_chars <= 0:
        return
    min_width = max_chars * font_size * 0.62 + 5.0
    width = _f(geom.get("width"))
    if width < min_width:
        geom.set("width", f"{min_width:.1f}".rstrip("0").rstrip("."))


def _geom_bbox(cell: ET.Element) -> tuple[float, float, float, float] | None:
    geom = cell.find("mxGeometry")
    if geom is None:
        return None
    return _bbox(geom)


def _next_id(root: ET.Element) -> int:
    max_id = 0
    for cell in root.findall(".//mxCell"):
        cid = cell.get("id", "")
        if cid.isdigit():
            max_id = max(max_id, int(cid))
    return max_id + 1


TEXT_FIXES = {
    "Al-Enabled": "AI-Enabled",
    "Al-Enhanced": "AI-Enhanced",
    "Al-Driven": "AI-Driven",
    "coloreled code": "colored code",
    "atom/lrap": "atom/trap",
    "XIZ bias": "X/Z bias",
    "dX,dZ targets": "dX, dZ targets",
    "PyZXtHand Layout |": "PyZX+Hand Layout",
    "Input Metrocs": "Input Metrics",
    "Physical-Qubit Drand": "Physical-Qubit Brand",
    "(FeMoco}": "(FeMoco)",
    "Elcloncy": "Efficiency",
    "Soinple": "Sample",
    "Overcad": "Overhead",
    "Phpeical Qublt": "Physical-Qubit",
    "RL Policy _": "RL Policy",
    "RL Circuit Token": "RL Circuit Token",
    "Messtah": "Messiah",
    "Inerters-": "Interfero-",
    "Qiskit+Surface": "Qiskit+Surface",
}

SKIP_VALUES = {
    "",
    "Je",
    ";",
    "Neu",
    "tral",
    "itives",
    "cali",
    "8",
    "Messiah",
}


def _fixed_value(value: str) -> str:
    out = value
    for src, dst in TEXT_FIXES.items():
        out = out.replace(src, dst)
    return out


def _text_cells(path: Path) -> list[ET.Element]:
    tree = ET.parse(path)
    root = tree.getroot()
    cells: list[ET.Element] = []
    for cell in root.findall(".//mxCell"):
        value = cell.get("value")
        if value is None or value in SKIP_VALUES:
            continue
        geom = cell.find("mxGeometry")
        if geom is None:
            continue
        fixed = _fixed_value(value)
        if fixed in SKIP_VALUES:
            continue
        c = copy.deepcopy(cell)
        c.set("value", fixed)
        c.set("parent", "1")
        c.set("vertex", "1")
        c.attrib.pop("edge", None)
        _set_style(
            c,
            strokeColor="none",
            fillColor="none",
            fontColor="#111111",
            fontFamily="DejaVu Sans",
            verticalAlign="middle",
            spacing="0",
            spacingTop="0",
            spacingBottom="0",
            spacingLeft="0",
            spacingRight="0",
        )
        _fit_text_width(c)
        cells.append(c)
    return cells


def _apply_manual_text_geometry(cells: list[ET.Element]) -> None:
    for cell in cells:
        value = cell.get("value", "")
        style = _style_dict(cell.get("style", ""))

        if value == "Ranked Candidates & Surgery Scores":
            _set_geom(cell, x=195.0, y=16.5, width=220.0, height=16.0)
            _set_style(cell, fontSize="11", fontStyle="1", align="center")
        elif value == "LM":
            cell.set("value", "LLM")
            _set_geom(cell, x=976.0, y=86.0, width=23.0, height=13.0)
            _set_style(cell, fontSize="8", align="center", fontStyle="1")
        elif value == "No":
            cell.set("value", "Noise")
            _set_geom(cell, x=839.0, y=89.0, width=45.0, height=14.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Figure 1: AI-Enabled Co-Design of Quantum LDPC Codes and Fault-Tolerant Interfaces.":
            _set_geom(cell, x=11.0, y=347.5, width=660.0, height=21.0)
            _set_style(cell, fontSize="13", fontStyle="1", align="left")
        elif value == "Figure 2: ML-Driven Codesign of QEC for Strongly Correlated Spin Dynamics":
            cell.set("value", value + ".")
            _set_geom(cell, x=742.0, y=347.5, width=620.0, height=21.0)
            _set_style(cell, fontSize="13", fontStyle="1", align="left")
        elif value == "Figure 3: AI-Enhanced Continuous-Variable Quantum Computing for Quantum Field":
            _set_geom(cell, x=12.0, y=721.0, width=700.0, height=23.0)
            _set_style(cell, fontSize="13", fontStyle="1", align="left")
        elif value == "Theory:":
            cell.set("value", "Theory.")
            _set_geom(cell, x=12.0, y=742.5, width=80.0, height=22.0)
            _set_style(cell, fontSize="13", fontStyle="1", align="left")
        elif value == "Figure 4: ZX Calculus as a Unifying Language for AI-Driven, Application-Aware QEC.":
            _set_geom(cell, x=718.0, y=721.0, width=645.0, height=23.0)
            _set_style(cell, fontSize="13", fontStyle="1", align="left")
        elif value == "Noise analysis/Input":
            _set_geom(cell, x=754.0, y=29.0, width=150.0, height=15.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "LLM-Driven Recipe Composition":
            _set_geom(cell, x=954.0, y=29.0, width=185.0, height=15.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Switchable":
            _set_geom(cell, x=1190.0, y=99.5, width=86.0, height=15.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Switching":
            _set_geom(cell, x=1286.0, y=109.0, width=66.0, height=13.0)
            _set_style(cell, fontSize="9", align="center")
        elif value == "LLM+RL pipeline":
            _set_geom(cell, x=1238.0, y=286.0, width=112.0, height=15.0)
            _set_style(cell, fontSize="10", align="center")
        elif value == "Phase":
            _set_geom(cell, x=1192.0, y=286.0, width=45.0, height=13.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Code":
            x, y, _w, _h = _geom_bbox(cell) or (0, 0, 0, 0)
            if x > 1180 and y < 145:
                _set_geom(cell, x=1204.0, y=115.0, width=44.0, height=13.0)
                _set_style(cell, fontSize="9", fontStyle="1", align="center")
            elif x > 1130 and y > 480:
                _set_geom(cell, x=1128.0, y=507.0, width=86.0, height=13.0)
                _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Families":
            x, y, _w, _h = _geom_bbox(cell) or (0, 0, 0, 0)
            if x > 1180 and y < 150:
                _set_geom(cell, x=1197.0, y=130.0, width=70.0, height=13.0)
                _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "QFT":
            _set_geom(cell, x=35.0, y=444.5, width=47.0, height=16.0)
            _set_style(cell, fontSize="12", fontStyle="1", align="center")
        elif value == "Simulation":
            _set_geom(cell, x=25.0, y=459.5, width=75.0, height=13.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Target":
            _set_geom(cell, x=34.0, y=475.0, width=56.0, height=14.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Physics-Informed":
            _set_geom(cell, x=132.0, y=444.0, width=104.0, height=16.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Decomposition":
            _set_geom(cell, x=265.0, y=444.5, width=98.0, height=16.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Hardware":
            _set_geom(cell, x=392.0, y=444.5, width=75.0, height=14.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Verification":
            _set_geom(cell, x=483.0, y=444.5, width=76.0, height=14.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Input Workloads":
            _set_geom(cell, x=866.0, y=402.0, width=108.0, height=16.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Benchmarking":
            _set_geom(cell, x=1118.0, y=418.0, width=98.0, height=16.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Hubbard":
            _set_geom(cell, x=890.0, y=455.5, width=54.0, height=12.0)
            _set_style(cell, fontSize="7", fontStyle="1", align="center")
        elif value == "Scheduling":
            _set_geom(cell, x=378.0, y=567.5, width=80.0, height=13.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Distillation":
            _set_geom(cell, x=276.0, y=682.5, width=72.0, height=13.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Input Metrics":
            _set_geom(cell, x=838.0, y=681.0, width=88.0, height=13.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Physical-Qubit Brand":
            _set_geom(cell, x=984.0, y=681.0, width=120.0, height=14.0)
            _set_style(cell, fontSize="10", align="center")
        elif value == "Soundness":
            _set_geom(cell, x=1149.0, y=681.0, width=74.0, height=13.0)
            _set_style(cell, fontSize="8", align="center")
        elif value == "RL Circuit Token":
            _set_geom(cell, x=1117.0, y=582.0, width=93.0, height=13.0)
            _set_style(cell, fontSize="9", fontStyle="1", align="center")
        elif value == "Qiskit+Surface":
            _set_geom(cell, x=1124.0, y=492.0, width=94.0, height=14.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Natural-":
            _set_geom(cell, x=714.0, y=596.0, width=52.0, height=12.0)
            _set_style(cell, fontSize="8", align="center")
        elif value == "Frontend":
            _set_geom(cell, x=714.0, y=620.0, width=52.0, height=12.0)
            _set_style(cell, fontSize="8", align="center")
        elif value == "Graph":
            x, y, _w, _h = _geom_bbox(cell) or (0, 0, 0, 0)
            if 760 < x < 820 and 585 < y < 620:
                _set_geom(cell, x=790.0, y=596.0, width=76.0, height=12.0)
                _set_style(cell, fontSize="8", align="center")
        elif value == "Model":
            x, y, _w, _h = _geom_bbox(cell) or (0, 0, 0, 0)
            if 810 < x < 860 and 615 < y < 635:
                _set_geom(cell, x=790.0, y=620.0, width=76.0, height=12.0)
                _set_style(cell, fontSize="8", align="center")
        elif value == "LLM-driven":
            _set_geom(cell, x=22.0, y=148.0, width=62.0, height=14.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Exploration":
            _set_geom(cell, x=22.0, y=164.0, width=62.0, height=13.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "RL-based":
            _set_geom(cell, x=242.0, y=148.0, width=76.0, height=14.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "Exploitation":
            _set_geom(cell, x=241.0, y=164.0, width=78.0, height=13.0)
            _set_style(cell, fontSize="8", fontStyle="1", align="center")
        elif value == "Propose partially<br>assembled,":
            _set_geom(cell, x=101.0, y=56.0, width=118.0, height=31.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "colored code":
            _set_geom(cell, x=112.0, y=88.0, width=104.0, height=14.0)
            _set_style(cell, fontSize="10", fontStyle="1", align="center")
        elif value == "VT Team<br>(Full ZX-Calculus":
            _set_geom(cell, x=535.0, y=45.0, width=105.0, height=27.0)
            _set_style(cell, align="center", fontStyle="1")
        elif value == "Surgery Analysis)":
            _set_geom(cell, x=530.0, y=73.5, width=106.0, height=15.0)
            _set_style(cell, align="center", fontStyle="1")
        elif value == "Filter Funnel":
            _set_geom(cell, x=384.0, y=69.5, width=92.0, height=15.0)
            _set_style(cell, align="center", fontSize="9", fontStyle="1")
        elif value in {"Efficiency", "Sample", "Overhead", "Physical-Qubit"}:
            _set_style(cell, align="center")

        if "rotation" in style:
            _set_style(cell, rotation=style["rotation"])


def _manual_extra_text(next_id: int) -> tuple[list[ET.Element], int]:
    extras = [
        ("Propose partially<br>assembled,<br>colored code", 100.0, 56.0, 120.0, 48.0, 10, True, 0),
        ("LLM-driven<br>Exploration", 18.0, 148.0, 68.0, 31.0, 10, True, 0),
        ("Features", 106.0, 166.0, 52.0, 12.0, 8, False, 0),
        ("Features", 170.0, 166.0, 52.0, 12.0, 8, False, 0),
        ("Tanner-graph", 121.0, 184.0, 82.0, 14.0, 8, False, 0),
        ("RL-based<br>Exploitation", 242.0, 148.0, 80.0, 31.0, 10, True, 0),
        ("Algebraic<br>Validity<br>Check", 352.0, 132.0, 70.0, 46.0, 10, True, 0),
        ("GNN<br>Surrogate<br>Filtering", 444.0, 132.0, 72.0, 46.0, 10, True, 0),
        ("Classical<br>Metrics Ranker", 396.0, 198.0, 80.0, 27.0, 8, False, 0),
        ("Validated Novel<br>Code Families", 562.0, 106.0, 102.0, 34.0, 10, True, 0),
        ("Improved<br>Surgery Metrics", 562.0, 191.0, 104.0, 34.0, 10, True, 0),
        ("Ranked Candidates & Surgery Scores", 284.0, 251.0, 220.0, 14.0, 8, False, 0),
        ("Weekly<br>batching", 245.0, 280.0, 58.0, 30.0, 9, True, 0),
        ("Weekly<br>batching", 477.0, 280.0, 58.0, 30.0, 9, True, 0),
        ("1-3", 127.0, 320.0, 42.0, 14.0, 10, True, 0),
        ("4-6", 344.0, 320.0, 42.0, 14.0, 10, True, 0),
        ("7-9", 576.0, 320.0, 42.0, 14.0, 10, True, 0),
        ("Neutral<br>atom/trap", 733.0, 96.0, 58.0, 36.0, 9, True, 0),
        ("Noise<br>fingerprint<br>extraction", 834.0, 88.0, 62.0, 48.0, 10, True, 0),
        ("Neutral<br>atom/ion trap", 726.0, 185.0, 76.0, 36.0, 8, True, 0),
        ("Propose Algebraic<br>code recipes", 1008.0, 62.0, 112.0, 33.0, 8, True, 0),
        ("Modulate<br>dX, dZ targets", 1002.0, 124.0, 100.0, 33.0, 10, True, 0),
        ("Candidate<br>codes", 952.0, 232.0, 66.0, 30.0, 8, False, 0),
        ("Noise<br>annotation", 1090.0, 246.0, 68.0, 28.0, 8, False, 0),
        ("GNN Surrogate<br>(Fast Feedback)", 998.0, 272.0, 98.0, 31.0, 10, True, 0),
        ("Switchable<br>Code<br>Families", 1192.0, 100.0, 72.0, 44.0, 10, True, 0),
        ("Switching<br>circuits", 1288.0, 110.0, 62.0, 30.0, 9, False, 0),
        ("Decoders", 1277.0, 162.0, 74.0, 14.0, 9, True, 0),
        ("Trotter<br>Ordering", 1265.0, 204.0, 86.0, 36.0, 10, True, 0),
        ("Phase I: LLM+RL pipeline", 1192.0, 286.0, 160.0, 15.0, 10, False, 0),
        ("QFT<br>Simulation<br>Target", 32.0, 444.0, 64.0, 48.0, 10, True, 0),
        ("Physics-Informed<br>Agent", 132.0, 444.0, 104.0, 34.0, 9, True, 0),
        ("Decomposition<br>Agent", 266.0, 444.0, 98.0, 34.0, 9, True, 0),
        ("Hardware<br>Agent", 392.0, 444.0, 75.0, 34.0, 9, True, 0),
        ("Verification<br>Agent", 483.0, 444.0, 77.0, 34.0, 9, True, 0),
        ("Executable<br>CV Quantum<br>Circuit", 588.0, 446.0, 74.0, 48.0, 9, True, 0),
        ("Select Truncation", 140.0, 488.0, 92.0, 16.0, 9, True, 0),
        ("Sub-unitaries", 151.0, 523.0, 74.0, 12.0, 8, True, 0),
        ("Gaussian", 141.0, 579.0, 55.0, 13.0, 8, False, 0),
        ("Non-<br>Gauge<br>coupling", 191.0, 578.0, 42.0, 37.0, 8, True, 0),
        ("CV Primitives", 279.0, 486.0, 82.0, 16.0, 9, True, 0),
        ("Bloch-<br>Messiah", 274.0, 526.0, 44.0, 25.0, 6, False, 0),
        ("Interfero-<br>meters", 323.0, 526.0, 50.0, 36.0, 6, False, 0),
        ("Non-<br>Gaussian", 272.0, 592.0, 44.0, 26.0, 6, False, 0),
        ("Gadgets", 319.0, 592.0, 44.0, 12.0, 6, False, 0),
        ("Iterative<br>feedback", 495.0, 491.0, 56.0, 30.0, 8, True, 0),
        ("Scheduling", 378.0, 567.0, 80.0, 14.0, 8, True, 0),
        ("Error", 505.0, 572.0, 45.0, 16.0, 10, True, 0, "#b33838"),
        ("Model<br>Distillation", 287.0, 664.0, 62.0, 31.0, 10, True, 0),
        ("Fast-track", 414.0, 650.0, 68.0, 13.0, 8, True, 0),
        ("For Routine Circuit Families", 367.0, 672.0, 154.0, 12.0, 8, False, 0),
        ("Input Workloads", 866.0, 402.0, 108.0, 16.0, 10, True, 0),
        ("Quantum Chemistry<br>(FeMoco)", 770.0, 455.0, 102.0, 29.0, 7, False, 0),
        ("Hubbard<br>Model", 888.0, 455.0, 58.0, 28.0, 7, False, 0),
        ("Natural-<br>Language<br>Frontend", 708.0, 596.0, 64.0, 38.0, 8, True, 0),
        ("Graph<br>Diffusion<br>Model", 790.0, 596.0, 76.0, 38.0, 8, True, 0),
        ("ZX Diagram<br>Synthesis", 896.0, 512.0, 70.0, 32.0, 10, True, 0),
        ("ZX-FT Protocol<br>Synthesis", 1015.0, 592.0, 76.0, 31.0, 8, True, 0),
        ("Benchmarking", 1118.0, 418.0, 98.0, 16.0, 10, True, 0),
        ("Qiskit+Surface<br>Code", 1124.0, 492.0, 94.0, 31.0, 8, True, 0),
        ("RL Circuit Token", 1117.0, 582.0, 93.0, 13.0, 9, True, 0),
        ("PyZX+Hand Layout", 1117.5, 649.0, 101.5, 15.0, 9, True, 0),
        ("Key:", 732.0, 682.0, 34.0, 14.0, 10, True, 0),
        ("Input Metrics", 838.0, 681.0, 88.0, 13.0, 8, False, 0),
        ("Physical-Qubit Brand", 984.0, 681.0, 120.0, 14.0, 10, False, 0),
        ("Soundness", 1149.0, 681.0, 74.0, 13.0, 8, False, 0),
        ("Scaling", 1292.0, 694.0, 55.0, 13.0, 9, True, 0),
    ]
    cells: list[ET.Element] = []
    for item in extras:
        value, x, y, w, h, fs, bold, rotation = item[:8]
        color = item[8] if len(item) > 8 else "#111111"
        style = {
            "text": None,
            "html": "1",
            "strokeColor": "none",
            "fillColor": "none",
            "align": "center",
            "verticalAlign": "middle",
            "whiteSpace": "nowrap",
            "rounded": "0",
            "fontFamily": "DejaVu Sans",
            "fontSize": str(fs),
            "fontColor": color,
        }
        if bold:
            style["fontStyle"] = "1"
        if rotation:
            style["rotation"] = str(rotation)
        cell = ET.Element(
            "mxCell",
            {
                "id": str(next_id),
                "value": value,
                "style": _style_string(style),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{x:.1f}".rstrip("0").rstrip("."),
                "y": f"{y:.1f}".rstrip("0").rstrip("."),
                "width": f"{w:.1f}".rstrip("0").rstrip("."),
                "height": f"{h:.1f}".rstrip("0").rstrip("."),
                "as": "geometry",
            },
        )
        cells.append(cell)
        next_id += 1
    return cells, next_id


def _filter_overlapped_donor_text(donor_cells: list[ET.Element],
                                  manual_cells: list[ET.Element]) -> tuple[list[ET.Element], int]:
    manual_boxes = []
    for cell in manual_cells:
        b = _geom_bbox(cell)
        if b is not None:
            manual_boxes.append(_pad(b, 2.0, 1.5))
    if not manual_boxes:
        return donor_cells, 0

    kept: list[ET.Element] = []
    removed = 0
    for cell in donor_cells:
        b = _geom_bbox(cell)
        if b is None:
            kept.append(cell)
            continue
        cx = b[0] + b[2] / 2
        cy = b[1] + b[3] / 2
        area = max(1.0, _area(b))
        discard = False
        for mb in manual_boxes:
            center_inside = mb[0] <= cx <= mb[0] + mb[2] and mb[1] <= cy <= mb[1] + mb[3]
            if center_inside or _intersect(b, mb) / area >= 0.35:
                discard = True
                break
        if discard:
            removed += 1
        else:
            kept.append(cell)
    return kept, removed


def _native_rect(id_value: int, x: float, y: float, w: float, h: float,
                 *, rounded: bool = True, fill: str = "none",
                 stroke: str = "#8f9cac",
                 stroke_width: float = 1.0) -> ET.Element:
    style = {
        "rounded": "1" if rounded else "0",
        "whiteSpace": "wrap",
        "html": "1",
        "fillColor": fill,
        "strokeColor": stroke,
        "strokeWidth": f"{stroke_width:.1f}".rstrip("0").rstrip("."),
        "arcSize": "8",
        "connectable": "0",
    }
    cell = ET.Element(
        "mxCell",
        {
            "id": str(id_value),
            "value": "",
            "style": _style_string(style),
            "vertex": "1",
            "parent": "1",
        },
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        {
            "x": f"{x:.1f}".rstrip("0").rstrip("."),
            "y": f"{y:.1f}".rstrip("0").rstrip("."),
            "width": f"{w:.1f}".rstrip("0").rstrip("."),
            "height": f"{h:.1f}".rstrip("0").rstrip("."),
            "as": "geometry",
        },
    )
    return cell


def _native_edge(id_value: int, x1: float, y1: float, x2: float, y2: float,
                 *, arrow: str = "classic", stroke: str = "#111111",
                 stroke_width: float = 1.4) -> ET.Element:
    style = {
        "html": "1",
        "rounded": "0",
        "strokeColor": stroke,
        "strokeWidth": f"{stroke_width:.1f}".rstrip("0").rstrip("."),
        "endArrow": arrow,
        "endFill": "1" if arrow != "none" else "0",
    }
    cell = ET.Element(
        "mxCell",
        {
            "id": str(id_value),
            "value": "",
            "style": _style_string(style),
            "edge": "1",
            "parent": "1",
        },
    )
    geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.SubElement(
        geom,
        "mxPoint",
        {
            "x": f"{x1:.1f}".rstrip("0").rstrip("."),
            "y": f"{y1:.1f}".rstrip("0").rstrip("."),
            "as": "sourcePoint",
        },
    )
    ET.SubElement(
        geom,
        "mxPoint",
        {
            "x": f"{x2:.1f}".rstrip("0").rstrip("."),
            "y": f"{y2:.1f}".rstrip("0").rstrip("."),
            "as": "targetPoint",
        },
    )
    return cell


def _native_rect_specs() -> list[tuple[float, float, float, float, bool]]:
    return [
        # Figure 1
        (20, 215, 98, 88, True),
        (340, 66, 184, 114, True),
        (386, 194, 99, 36, True),
        (557, 102, 112, 126, True),
        # Figure 2
        (713, 23, 200, 284, True),
        (924, 23, 246, 291, True),
        (1183, 24, 180, 284, True),
        (825, 85, 77, 118, True),
        (959, 77, 41, 43, False),
        # Figure 3
        (18, 442, 92, 184, True),
        (130, 442, 113, 185, True),
        (263, 442, 112, 184, True),
        (391, 442, 76, 184, True),
        (483, 442, 77, 184, True),
        (276, 632, 82, 69, True),
        (138, 485, 98, 32, True),
        (139, 519, 98, 104, True),
        (270, 482, 98, 141, True),
        # Figure 4
        (768, 400, 302, 85, True),
        (881, 508, 102, 141, True),
        (1015, 537, 69, 61, True),
        (1110, 414, 115, 255, True),
        (725, 675, 500, 29, True),
    ]


def _native_primitive_cells(next_id: int, *,
                            include_rects: bool = True,
                            include_extra_edges: bool = False) -> tuple[list[ET.Element], int]:
    cells: list[ET.Element] = []
    if include_rects:
        for x, y, w, h, rounded in _native_rect_specs():
            cells.append(_native_rect(next_id, x, y, w, h, rounded=rounded))
            next_id += 1

    if include_extra_edges:
        edges = [
            # Intentionally conservative. The base file already contains
            # many detected native edges; extra arrows are opt-in because
            # duplicate arrows quickly make the figure visually heavier.
            (82, 121, 113, 121), (205, 121, 242, 121), (304, 121, 341, 121),
            (524, 122, 558, 122), (790, 164, 825, 164), (902, 164, 924, 164),
            (110, 540, 130, 540), (243, 540, 263, 540), (375, 540, 391, 540),
            (467, 540, 483, 540), (560, 540, 585, 540), (735, 577, 800, 577),
            (860, 577, 881, 577), (983, 578, 1015, 578), (1084, 578, 1110, 578),
        ]
        for x1, y1, x2, y2 in edges:
            cells.append(_native_edge(next_id, x1, y1, x2, y2))
            next_id += 1
    return cells, next_id


def _style_value(style: str, key: str, default: str = "") -> str:
    prefix = key + "="
    for part in style.split(";"):
        if part.startswith(prefix):
            return part.split("=", 1)[1]
    return default


def _is_light_panel_fill(color: str) -> bool:
    if not color.startswith("#") or len(color) != 7:
        return False
    try:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
    except ValueError:
        return False
    return r + g + b >= 610 and max(r, g, b) - min(r, g, b) >= 4


def _rect_match_score(cell_box: tuple[float, float, float, float],
                      rect: tuple[float, float, float, float]) -> float:
    x, y, w, h = cell_box
    rx, ry, rw, rh = rect
    return (
        abs(x - rx) / max(1.0, rw)
        + abs(y - ry) / max(1.0, rh)
        + abs(w - rw) / max(1.0, rw)
        + abs(h - rh) / max(1.0, rh)
    )


def _replace_rect_stencils(root: ET.Element, next_id: int) -> tuple[int, int, int]:
    """Replace confidently matched stencil panel boxes with native rects.

    Unlike overlaying native rectangles, this removes matching old stencil
    geometry first and inserts the native shape at the same z-order position,
    so borders do not get double-painted.
    """
    graph_root = root.find(".//root")
    if graph_root is None:
        raise RuntimeError("drawio root not found")
    replaced_stencils = 0
    native_rects = 0
    for rx, ry, rw, rh, rounded in _native_rect_specs():
        rect = (rx, ry, rw, rh)
        matches: list[tuple[int, ET.Element, str, str, float]] = []
        for idx, cell in enumerate(list(graph_root)):
            style = cell.get("style", "")
            if "shape=stencil(" not in style:
                continue
            b = _geom_bbox(cell)
            if b is None:
                continue
            score = _rect_match_score(b, rect)
            if score > 0.18:
                continue
            fill = _style_value(style, "fillColor")
            stroke = _style_value(style, "strokeColor", "none")
            area = _area(b)
            target_area = rw * rh
            if area < target_area * 0.62 or area > target_area * 1.55:
                continue
            if not (_is_light_panel_fill(fill) or fill.lower() in {"#959ba1", "#8f9cac", "#d7e3f0"}):
                continue
            matches.append((idx, cell, fill, stroke, score))
        if not matches:
            continue
        matches.sort(key=lambda row: (row[4], row[0]))
        # Keep all near-identical panel/background stencils for this rectangle,
        # but avoid removing interior icons by requiring close geometry.
        to_remove = [m for m in matches if m[4] <= max(0.08, matches[0][4] + 0.05)]
        insert_at = min(m[0] for m in to_remove)
        panel_fills = [m[2] for m in to_remove if _is_light_panel_fill(m[2])]
        fill = panel_fills[0] if panel_fills else "#ebf0f8"
        stroke = "#8f9cac"
        for _idx, cell, _fill, _stroke, _score in to_remove:
            graph_root.remove(cell)
        native = _native_rect(
            next_id,
            rx,
            ry,
            rw,
            rh,
            rounded=rounded,
            fill=fill,
            stroke=stroke,
            stroke_width=1.0,
        )
        graph_root.insert(insert_at, native)
        next_id += 1
        replaced_stencils += len(to_remove)
        native_rects += 1
    return replaced_stencils, native_rects, next_id


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _background_sampler(path: Path | None):
    if path is None or Image is None or not path.exists():
        return None
    img = Image.open(path).convert("RGB")

    def sample(box: tuple[float, float, float, float]) -> str:
        x, y, w, h = box
        left = max(0, int(x))
        top = max(0, int(y))
        right = min(img.width, int(x + w + 0.999))
        bottom = min(img.height, int(y + h + 0.999))
        if right <= left or bottom <= top:
            return "#ffffff"
        crop = img.crop((left, top, right, bottom))
        pixels = list(crop.getdata())
        bright = [p for p in pixels if (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) > 175]
        if len(bright) >= max(4, len(pixels) // 10):
            pixels = bright
        # Median is more stable than mean around anti-aliased glyph edges.
        channels = []
        for idx in range(3):
            vals = sorted(p[idx] for p in pixels)
            channels.append(vals[len(vals) // 2])
        # Pull only actual white paper backgrounds to white; light-blue panels
        # must keep their tint or the eraser boxes become visible.
        if min(channels) > 252:
            return "#ffffff"
        return _hex(tuple(channels))  # type: ignore[arg-type]

    return sample


def _panel_fill_for_box(box: tuple[float, float, float, float]) -> str | None:
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    for rx, ry, rw, rh, _rounded in _native_rect_specs():
        if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
            continue
        if (rx, ry, rw, rh) == (924, 23, 246, 291):
            return "#d7e3f0"
        # Header bands in the original are a slightly darker blue than body
        # panels. Keeping this deterministic avoids noisy source-image samples.
        header_h = min(29.0, rh * 0.24)
        if cy <= ry + header_h:
            return "#dbe7f4"
        return "#ebf0f8"
    return None


def _eraser_cells(cells: Iterable[ET.Element], next_id: int,
                  sampler=None) -> tuple[list[ET.Element], int]:
    erasers: list[ET.Element] = []
    for cell in cells:
        b = _geom_bbox(cell)
        if b is None:
            continue
        value = cell.get("value", "")
        pad_x = 3.6
        pad_y = 1.7
        if value.startswith("Figure ") or value == "Theory.":
            pad_x = 4.8
            pad_y = 2.0
        elif len(re.sub("<br>", "", value)) <= 3:
            pad_x = 2.4
            pad_y = 1.1
        eb = _pad(b, pad_x, pad_y)
        panel_fill = _panel_fill_for_box(eb)
        fill = panel_fill or (sampler(eb) if sampler else "#ffffff")
        style = {
            "rounded": "0",
            "whiteSpace": "wrap",
            "html": "1",
            "fillColor": fill,
            "strokeColor": "none",
            "connectable": "0",
            "selectable": "0",
        }
        rotation = _style_dict(cell.get("style", "")).get("rotation")
        if rotation:
            style["rotation"] = rotation
        e = ET.Element(
            "mxCell",
            {
                "id": str(next_id),
                "value": "",
                "style": _style_string(style),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            e,
            "mxGeometry",
            {
                "x": f"{eb[0]:.1f}".rstrip("0").rstrip("."),
                "y": f"{eb[1]:.1f}".rstrip("0").rstrip("."),
                "width": f"{eb[2]:.1f}".rstrip("0").rstrip("."),
                "height": f"{eb[3]:.1f}".rstrip("0").rstrip("."),
                "as": "geometry",
            },
        )
        erasers.append(e)
        next_id += 1
    return erasers, next_id


def _manual_eraser_cells(next_id: int) -> tuple[list[ET.Element], int]:
    regions = [
        (704.0, 594.0, 68.0, 40.0, "#ffffff"),
        (772.0, 594.0, 100.0, 42.0, "#ffffff"),
        (118.0, 318.0, 54.0, 17.0, "#ffffff"),
        (274.0, 484.0, 92.0, 22.0, "#ebf0f8"),
        (270.0, 523.0, 52.0, 45.0, "#ebf0f8"),
        (322.0, 523.0, 54.0, 43.0, "#ebf0f8"),
        (268.0, 589.0, 54.0, 31.0, "#ebf0f8"),
        (316.0, 589.0, 48.0, 27.0, "#ebf0f8"),
        (502.0, 571.0, 50.0, 25.0, "#ebf0f8"),
        (558.0, 188.0, 108.0, 39.0, "#ebf0f8"),
    ]
    cells: list[ET.Element] = []
    for x, y, w, h, fill in regions:
        style = {
            "rounded": "0",
            "whiteSpace": "wrap",
            "html": "1",
            "fillColor": fill,
            "strokeColor": "none",
            "connectable": "0",
            "selectable": "0",
        }
        cell = ET.Element(
            "mxCell",
            {
                "id": str(next_id),
                "value": "",
                "style": _style_string(style),
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": f"{x:.1f}".rstrip("0").rstrip("."),
                "y": f"{y:.1f}".rstrip("0").rstrip("."),
                "width": f"{w:.1f}".rstrip("0").rstrip("."),
                "height": f"{h:.1f}".rstrip("0").rstrip("."),
                "as": "geometry",
            },
        )
        cells.append(cell)
        next_id += 1
    return cells, next_id


def _text_boxes(cells: Iterable[ET.Element]) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for cell in cells:
        b = _geom_bbox(cell)
        if b is None:
            continue
        value = cell.get("value", "")
        pad_x = 3.0
        pad_y = 2.0
        if value.startswith("Figure ") or value == "Theory.":
            pad_x = 6.0
            pad_y = 3.0
        boxes.append(_pad(b, pad_x, pad_y))
    return boxes


def _looks_like_removable_glyph(cell: ET.Element) -> bool:
    if cell.get("vertex") != "1":
        return False
    if cell.get("value"):
        return False
    style = cell.get("style", "")
    if "shape=stencil(" not in style:
        return False
    b = _geom_bbox(cell)
    if b is None:
        return False
    _, _, w, h = b
    if w <= 0 or h <= 0:
        return False
    if w > 34 or h > 28:
        return False
    # Avoid intentionally drawn straight rules/ticks when possible.
    if h < 2.4 and w > 10:
        return False
    if w < 2.4 and h > 10:
        return False
    return True


def _remove_old_glyphs(root: ET.Element,
                       boxes: list[tuple[float, float, float, float]]) -> int:
    graph_root = root.find(".//root")
    if graph_root is None:
        raise RuntimeError("drawio root not found")
    removed = 0
    for cell in list(graph_root):
        if not _looks_like_removable_glyph(cell):
            continue
        b = _geom_bbox(cell)
        if b is None:
            continue
        a = _area(b)
        if a <= 0:
            continue
        for tb in boxes:
            overlap = _intersect(b, tb)
            cx = b[0] + b[2] / 2
            cy = b[1] + b[3] / 2
            inside_center = tb[0] <= cx <= tb[0] + tb[2] and tb[1] <= cy <= tb[1] + tb[3]
            if inside_center and overlap / a >= 0.45:
                graph_root.remove(cell)
                removed += 1
                break
    return removed


def rebuild(base: Path, donor: Path, output: Path, *, source: Path | None,
            remove_glyphs: bool, add_erasers: bool,
            add_native_primitives: bool, add_extra_edges: bool) -> None:
    tree = ET.parse(base)
    root = tree.getroot()
    graph_root = root.find(".//root")
    if graph_root is None:
        raise RuntimeError("drawio root not found")

    next_id = _next_id(root)
    donor_cells = _text_cells(donor)
    _apply_manual_text_geometry(donor_cells)
    extras, next_id = _manual_extra_text(next_id)
    donor_cells, removed_donor_text = _filter_overlapped_donor_text(donor_cells, extras)
    text_cells = donor_cells + extras

    boxes = _text_boxes(text_cells)
    removed = _remove_old_glyphs(root, boxes) if remove_glyphs else 0

    replaced_rect_stencils = 0
    native_rects = 0
    if add_native_primitives:
        replaced_rect_stencils, native_rects, next_id = _replace_rect_stencils(root, next_id)
        primitives, next_id = _native_primitive_cells(
            next_id,
            include_rects=False,
            include_extra_edges=add_extra_edges,
        )
        for cell in primitives:
            graph_root.append(cell)

    if add_erasers:
        sampler = _background_sampler(source)
        erasers, next_id = _eraser_cells(text_cells, next_id, sampler=sampler)
        for cell in erasers:
            graph_root.append(cell)
        erasers, next_id = _manual_eraser_cells(next_id)
        for cell in erasers:
            graph_root.append(cell)

    for cell in text_cells:
        cell.set("id", str(next_id))
        next_id += 1
        graph_root.append(cell)

    root.set("agent", "codex_native_text_rebuild")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="UTF-8", xml_declaration=True)
    print(
        f"wrote {output} ({len(text_cells)} text cells, removed {removed} old glyph stencils, "
        f"removed {removed_donor_text} donor text cells, "
        f"replaced {replaced_rect_stencils} box stencils with {native_rects} native rects)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/png_to_svg/codex_pure_vector_native_lines_v2.drawio")
    parser.add_argument("--donor", default="outputs/png_to_svg/codex_components_no_semantic.drawio")
    parser.add_argument("--output", default="outputs/png_to_svg/codex_drawio_native_text_rebuild_v4.drawio")
    parser.add_argument("--source", default="uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg")
    parser.add_argument("--remove-glyphs", action="store_true")
    parser.add_argument("--no-erasers", action="store_true")
    parser.add_argument("--no-native-primitives", action="store_true")
    parser.add_argument("--extra-edges", action="store_true")
    args = parser.parse_args()
    rebuild(
        Path(args.base),
        Path(args.donor),
        Path(args.output),
        source=Path(args.source) if args.source else None,
        remove_glyphs=args.remove_glyphs,
        add_erasers=not args.no_erasers,
        add_native_primitives=not args.no_native_primitives,
        add_extra_edges=args.extra_edges,
    )


if __name__ == "__main__":
    main()
