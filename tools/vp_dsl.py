#!/usr/bin/env python3
"""Compact DSL <-> vp_program for png->drawio SFT.

The model is trained to emit this DSL (the SFT TARGET); we parse it back to a
vp_program, which compiles deterministically to draw.io. Design:
  * JSON-Lines: one primitive per line -> robust parsing (skip a malformed line,
    keep the rest), unlike one giant JSON array.
  * Coordinates normalized to integer [0,1000] (matches Qwen-VL grounding
    convention). De-normalize at parse time using the target image W,H.
  * Short keys for compactness; defaults omitted.

serialize(program) -> str         (normalize by program canvas W,H)
parse(text, W, H) -> program       (de-normalize by target W,H)

Round-trip is intended to be loss-free enough that re-rendering the parsed
program reproduces the original's harness score.

Helpers return plain values; parse is tolerant by design (never raises on a bad line).
"""
from __future__ import annotations

import json
import os
from typing import Any

# Coordinate grid. 1000 matches Qwen-VL's pretrained grounding convention (best
# transfer, ~1.4px on a 1376px canvas). Raise (e.g. 4000) for finer targets at the
# cost of leaving the pretrained scale. RL refines precision below the grid later.
GRID = int(os.environ.get("VP_DSL_GRID", "1000"))

# Canonical task prompt shared by SFT export and inference/eval, so training and
# serving use identical instructions. Coordinates are normalized to [0,GRID].
INSTRUCTION = (
    "You are given a figure image. Reconstruct it as a draw.io-style diagram program. "
    "Output ONLY JSON-Lines (one compact JSON object per line, one per visual primitive), "
    "with integer coordinates normalized to [0,1000] (x by image width, y by image height; "
    "origin top-left). Emit panels/boxes first, then shapes, edges, then text labels. Schemas:\n"
    "region (filled box/panel): {\"k\":\"r\",\"b\":[x0,y0,x1,y1],\"fc\":\"#hex|none\",\"sc\":\"#hex\",\"rd\":0|1}\n"
    "shape (mark): {\"k\":\"s\",\"b\":[x0,y0,x1,y1],\"sh\":\"rectangle|ellipse|rhombus\",\"fc\":\"#hex|none\",\"sc\":\"#hex\"}\n"
    "edge (connector): {\"k\":\"e\",\"p\":[[x,y],...],\"sc\":\"#hex\",\"sw\":1.2,\"a\":[start,end]}\n"
    "text (label): {\"k\":\"t\",\"b\":[x0,y0,x1,y1],\"s\":fontsize,\"t\":\"string\",\"fc\":\"#hex\",\"bd\":0|1}\n"
    "No prose, no markdown fences."
)


def _n(v: float, span: int) -> int:
    return int(round(max(0.0, min(float(span), float(v))) / span * GRID))


def _d(v: float, span: int) -> float:
    return round(float(v) / GRID * span, 2)


def serialize(program: dict[str, Any]) -> str:
    canvas = program.get("canvas") or {}
    W = int(canvas.get("width") or 0)
    H = int(canvas.get("height") or 0)
    if W <= 0 or H <= 0:
        raise ValueError("program canvas must define positive width/height")
    lines: list[str] = []
    for p in program.get("primitives", []):
        t = p.get("type")
        st = p.get("style") or {}
        if t == "region":
            b = p["bbox"]
            o = {"k": "r", "b": [_n(b[0], W), _n(b[1], H), _n(b[2], W), _n(b[3], H)],
                 "fc": st.get("fill", "none"), "sc": st.get("stroke", "#808080")}
            if st.get("rounded"):
                o["rd"] = 1
        elif t == "shape":
            b = p["bbox"]
            o = {"k": "s", "b": [_n(b[0], W), _n(b[1], H), _n(b[2], W), _n(b[3], H)],
                 "sh": p.get("shape", "rectangle"),
                 "fc": st.get("fill", "none"), "sc": st.get("stroke", "#808080")}
        elif t == "edge":
            pts = p.get("path") or [p["bbox"][:2], p["bbox"][2:]]
            o = {"k": "e", "p": [[_n(x, W), _n(y, H)] for x, y in pts],
                 "sc": st.get("stroke", "#303030"),
                 "sw": round(float(st.get("stroke_width", 1.2)), 2)}
            if st.get("arrow_start") or st.get("arrow_end"):
                o["a"] = [1 if st.get("arrow_start") else 0, 1 if st.get("arrow_end") else 0]
        elif t == "text":
            b = p["bbox"]
            o = {"k": "t", "b": [_n(b[0], W), _n(b[1], H), _n(b[2], W), _n(b[3], H)],
                 "s": int(st.get("font_size", 11)), "t": str(p.get("text", "")),
                 "fc": st.get("font_color", "#101010")}
            if st.get("bold"):
                o["bd"] = 1
            if st.get("align") and st.get("align") != "center":
                o["al"] = st.get("align")
        else:
            continue
        lines.append(json.dumps(o, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def _iter_objs(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    # whole-array form
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                for o in arr:
                    if isinstance(o, dict):
                        yield o
                return
        except Exception:  # noqa: BLE001
            pass
    # JSON-Lines form (tolerant)
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(o, dict):
            yield o


def parse(text: str, W: int, H: int) -> dict[str, Any]:
    prims: list[dict[str, Any]] = []
    n = {"r": 0, "s": 0, "e": 0, "t": 0}

    def nid(k: str) -> str:
        n[k] += 1
        pref = {"r": "region", "s": "shape", "e": "edge", "t": "text"}[k]
        return f"m_{pref}_{n[k]:04d}"

    for o in _iter_objs(text):
        k = o.get("k")
        try:
            if k == "r" and len(o.get("b", [])) == 4:
                b = o["b"]
                prims.append({"id": nid("r"), "type": "region", "role": "panel_or_container",
                              "bbox": [_d(b[0], W), _d(b[1], H), _d(b[2], W), _d(b[3], H)],
                              "style": {"fill": o.get("fc", "none"), "stroke": o.get("sc", "#808080"),
                                        "rounded": bool(o.get("rd"))}, "source": "model"})
            elif k == "s" and len(o.get("b", [])) == 4:
                b = o["b"]
                prims.append({"id": nid("s"), "type": "shape", "role": "symbol_or_mark",
                              "shape": o.get("sh", "rectangle"),
                              "bbox": [_d(b[0], W), _d(b[1], H), _d(b[2], W), _d(b[3], H)],
                              "style": {"fill": o.get("fc", "none"), "stroke": o.get("sc", "#808080")},
                              "source": "model"})
            elif k == "e" and len(o.get("p", [])) >= 2:
                pts = [[_d(x, W), _d(y, H)] for x, y in o["p"] if True]
                xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
                a = o.get("a", [0, 0])
                prims.append({"id": nid("e"), "type": "edge", "role": "connector",
                              "bbox": [min(xs), min(ys), max(xs), max(ys)], "path": pts,
                              "style": {"stroke": o.get("sc", "#303030"),
                                        "stroke_width": float(o.get("sw", 1.2)),
                                        "arrow_start": bool(a[0]), "arrow_end": bool(a[1] if len(a) > 1 else 0)},
                              "source": "model"})
            elif k == "t" and len(o.get("b", [])) == 4 and str(o.get("t", "")).strip():
                b = o["b"]
                prims.append({"id": nid("t"), "type": "text", "role": "label",
                              "bbox": [_d(b[0], W), _d(b[1], H), _d(b[2], W), _d(b[3], H)],
                              "text": str(o["t"]),
                              "style": {"font_size": int(o.get("s", 11)), "bold": bool(o.get("bd")),
                                        "align": o.get("al", "center"), "font_color": o.get("fc", "#101010")},
                              "source": "model"})
        except Exception:  # noqa: BLE001
            continue
    return {
        "version": "dsl-1", "source": "model",
        "canvas": {"width": int(W), "height": int(H)}, "coordinate_space": "pixel",
        "counts": {"regions": n["r"], "shapes": n["s"], "edges": n["e"], "texts": n["t"],
                   "total": len(prims)},
        "primitives": prims, "metadata": {"from": "vp_dsl.parse"},
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="round-trip test a vp_program through the DSL")
    ap.add_argument("program")
    args = ap.parse_args()
    prog = json.load(open(args.program))
    W = prog["canvas"]["width"]; H = prog["canvas"]["height"]
    dsl = serialize(prog)
    back = parse(dsl, W, H)
    print(f"orig prims={len(prog['primitives'])}  dsl_lines={len(dsl.splitlines())}  "
          f"parsed prims={len(back['primitives'])}  dsl_chars={len(dsl)}")
    print("first 5 DSL lines:")
    for ln in dsl.splitlines()[:5]:
        print("  " + ln)
