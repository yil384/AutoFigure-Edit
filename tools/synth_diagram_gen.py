#!/usr/bin/env python3
"""Stage 1: procedural synthetic (png, vp_program) generator for png->drawio SFT.

Emits vp_programs of academic/technical-style diagrams (multi-panel layouts,
left-to-right pipelines, vertical architectures, trees) biased toward the dense,
labelled, box-and-arrow figures we ultimately target. Each program is rendered to
PNG via the existing, battle-tested compile_program_to_drawio + drawio CLI, so the
ground-truth program <-> image pairing is exact and free.

The vp_program is the model's TARGET DSL (compact, transpiles deterministically to
draw.io XML). This script is backbone-agnostic.

Usage:
  python3 tools/synth_diagram_gen.py --n 8 --outdir outputs/synth/preview --seed 7
  python3 tools/synth_diagram_gen.py --n 50000 --outdir data/synth/train --seed 1 --no-render  # bulk programs first

Helpers return (value, error): (result, None) on success, (None, 'msg') on failure.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import export_drawio_png  # noqa: E402

FONT_FAMILY = "Helvetica"

# Muted academic palettes: (fill, stroke) pairs.
NODE_PALETTES = [
    ("#d8e6ef", "#6f8190"), ("#eaf2f7", "#8f9cac"), ("#f0e6d8", "#b09a78"),
    ("#e6efe0", "#7fa06f"), ("#f3e0e6", "#b07f93"), ("#e8e4f0", "#8a7fb0"),
    ("#fdf3d8", "#c9b35a"), ("#dceeee", "#6fa0a0"), ("#ffffff", "#5a5a5a"),
]
PANEL_FILLS = ["#f7f9fb", "#f5f6f8", "#fbf8f3", "#f4f8f4", "#faf5f7", "none"]
TEXT_COLOR = "#101010"

NODE_WORDS = [
    "Encoder", "Decoder", "Attention", "Policy", "Reward", "Sampler", "Optimizer",
    "Filter", "Module", "Agent", "Memory", "Planner", "Verifier", "Embedding",
    "Tokenizer", "Classifier", "Generator", "Critic", "Buffer", "Scheduler",
    "Latent", "Prior", "Kernel", "Qubit", "Gate", "Circuit", "Noise", "Readout",
    "Hamiltonian", "Backbone", "Adapter", "Router", "Expert", "Decoder Head",
    "Feature Map", "Projection", "Aggregator", "Controller", "Estimator",
]
QUALIFIERS = ["", "Pre-", "Post-", "Self-", "Cross-", "Multi-", "Dense ", "Sparse ", "Hybrid ", "Deep "]
TITLES = [
    "System Architecture", "Training Pipeline", "Method Overview", "Inference Flow",
    "Data Flow", "Model Overview", "Framework", "Processing Stages", "Workflow",
    "Stage 1: Encoding", "Stage 2: Reasoning", "Stage 3: Decoding", "Module Layout",
]
EDGE_LABELS = ["", "", "", "yes", "no", "x_t", "h", "z", "loss", "grad", "->", "feedback", "sample"]


# --------------------------------------------------------------------------- #
def _node_label(rng: random.Random) -> str:
    w = rng.choice(NODE_WORDS)
    if rng.random() < 0.4:
        w = rng.choice(QUALIFIERS) + w
    if rng.random() < 0.25:
        w = f"{w} {rng.randint(1, 4)}"
    return w


def _border_point(bbox: list[float], side: str) -> list[float]:
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"r": [x1, cy], "l": [x0, cy], "t": [cx, y0], "b": [cx, y1]}[side]


class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.prims: list[dict[str, Any]] = []
        self.n = {"region": 0, "shape": 0, "edge": 0, "text": 0}

    def _id(self, t: str) -> str:
        self.n[t] += 1
        return f"syn_{t}_{self.n[t]:04d}"

    def region(self, bbox, fill=None, stroke="#8f9cac", rounded=True):
        self.prims.append({
            "id": self._id("region"), "type": "region", "role": "panel_or_container",
            "bbox": [round(v, 1) for v in bbox],
            "style": {"fill": fill or self.rng.choice(PANEL_FILLS), "stroke": stroke, "rounded": rounded},
            "source": "synth",
        })

    def shape(self, bbox, shape="rectangle", fill=None, stroke=None):
        f, s = self.rng.choice(NODE_PALETTES)
        self.prims.append({
            "id": self._id("shape"), "type": "shape", "role": "symbol_or_mark",
            "shape": shape, "bbox": [round(v, 1) for v in bbox],
            "style": {"fill": fill or f, "stroke": stroke or s},
            "source": "synth",
        })

    def edge(self, path, arrow_end=True, arrow_start=False, sw=None):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        self.prims.append({
            "id": self._id("edge"), "type": "edge", "role": "connector",
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "path": [[round(p[0], 1), round(p[1], 1)] for p in path],
            "style": {"stroke": "#303030", "stroke_width": sw or round(self.rng.uniform(1.0, 1.8), 2),
                      "arrow_start": arrow_start, "arrow_end": arrow_end},
            "source": "synth",
        })

    def text(self, bbox, txt, size=11, bold=False, align="center"):
        self.prims.append({
            "id": self._id("text"), "type": "text", "role": "label",
            "bbox": [round(v, 1) for v in bbox], "text": txt,
            "style": {"font_size": size, "bold": bold, "align": align, "font_color": TEXT_COLOR},
            "source": "synth",
        })

    def node(self, bbox, label=None, shape="rectangle", size=11):
        """A labelled box. Rectangles use a region (bottom layer) so the text
        label (layer 4) renders ON TOP; ellipse/rhombus use an outline-only shape
        so the label shows through (filled shapes on layer 7 would occlude text)."""
        label = label if label is not None else _node_label(self.rng)
        f, s = self.rng.choice(NODE_PALETTES)
        if shape == "rectangle":
            self.region(bbox, fill=f, stroke=s, rounded=self.rng.random() < 0.5)
        else:
            self.shape(bbox, shape=shape, fill="none", stroke=s)
        self.text([bbox[0], bbox[1], bbox[2], bbox[3]], label, size=size)

    def program(self, W, H) -> dict[str, Any]:
        return {
            "version": "synth-1",
            "source": "synthetic",
            "canvas": {"width": W, "height": H},
            "coordinate_space": "pixel",
            "counts": {k + "s" if not k.endswith("s") else k: v for k, v in self.n.items()} | {"total": len(self.prims)},
            "primitives": self.prims,
            "metadata": {"generator": "synth_diagram_gen"},
        }


# --------------------------------------------------------------------------- #
# Layout families
# --------------------------------------------------------------------------- #
def fam_horizontal_flow(b: Builder, W, H):
    rng = b.rng
    rows = rng.randint(1, 2)
    cols = rng.randint(3, 6)
    margin = 60
    if rng.random() < 0.7:
        b.text([margin, 12, W - margin, 40], rng.choice(TITLES), size=rng.randint(13, 16), bold=True)
    top = 70
    cell_w = (W - 2 * margin) / cols
    nh = rng.randint(48, 80)
    gap_y = rng.randint(90, 150)
    prev_row = None
    for r in range(rows):
        cy = top + r * gap_y + nh / 2
        nodes = []
        for c in range(cols):
            cx = margin + c * cell_w + cell_w / 2
            nw = cell_w * rng.uniform(0.55, 0.8)
            bbox = [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]
            shp = rng.choice(["rectangle", "rectangle", "rectangle", "ellipse", "rhombus"])
            b.node(bbox, shape=shp)
            nodes.append(bbox)
        for i in range(cols - 1):
            p0 = _border_point(nodes[i], "r")
            p1 = _border_point(nodes[i + 1], "l")
            b.edge([p0, p1])
            lab = rng.choice(EDGE_LABELS)
            if lab:
                mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
                b.text([mx - 22, my - 22, mx + 22, my - 6], lab, size=9)
        if prev_row is not None and rng.random() < 0.6:
            j = rng.randrange(cols)
            b.edge([_border_point(prev_row[j], "b"), _border_point(nodes[j], "t")])
        prev_row = nodes


def fam_vertical_stack(b: Builder, W, H):
    rng = b.rng
    layers = rng.randint(3, 6)
    margin = 80
    if rng.random() < 0.6:
        b.text([margin, 10, W - margin, 36], rng.choice(TITLES), size=rng.randint(13, 16), bold=True)
    top = 56
    avail = H - top - 30
    lh = avail / layers
    nh = lh * rng.uniform(0.5, 0.7)
    prev = None
    full_w = (W - 2 * margin)
    for i in range(layers):
        cy = top + i * lh + lh / 2
        nw = full_w * rng.uniform(0.45, 0.9)
        cx = W / 2
        bbox = [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]
        b.node(bbox)
        if prev is not None:
            b.edge([_border_point(prev, "b"), _border_point(bbox, "t")],
                   arrow_end=rng.random() < 0.85)
        prev = bbox


def fam_panel_grid(b: Builder, W, H):
    rng = b.rng
    pr, pc = rng.choice([(1, 2), (1, 3), (2, 2), (2, 1)])
    gap = 24
    pw = (W - gap * (pc + 1)) / pc
    ph = (H - gap * (pr + 1)) / pr
    for r in range(pr):
        for c in range(pc):
            x0 = gap + c * (pw + gap)
            y0 = gap + r * (ph + gap)
            b.region([x0, y0, x0 + pw, y0 + ph])
            b.text([x0 + 8, y0 + 6, x0 + pw - 8, y0 + 28],
                   f"({chr(97 + r * pc + c)}) " + rng.choice(TITLES), size=rng.randint(11, 13), bold=True, align="left")
            # small sub-flow inside panel (kept strictly within the panel bounds)
            n = rng.randint(2, 4)
            inner_top = y0 + 40
            inner_bot = y0 + ph - 14
            step = (inner_bot - inner_top) / n
            nh = min(46, step * 0.62)
            sub = []
            for k in range(n):
                cy = inner_top + k * step + step / 2
                nw = pw * rng.uniform(0.4, 0.72)
                cx = x0 + pw / 2
                bb = [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]
                b.node(bb, size=10)
                sub.append(bb)
            for k in range(n - 1):
                b.edge([_border_point(sub[k], "b"), _border_point(sub[k + 1], "t")])


def fam_tree(b: Builder, W, H):
    rng = b.rng
    margin = 60
    if rng.random() < 0.5:
        b.text([margin, 10, W - margin, 34], rng.choice(TITLES), size=14, bold=True)
    nh = 46
    root_w = rng.uniform(120, 180)
    root = [W / 2 - root_w / 2, 56, W / 2 + root_w / 2, 56 + nh]
    b.node(root)
    nchild = rng.randint(2, 4)
    cy = 56 + nh + rng.randint(70, 110)
    child_w = (W - 2 * margin) / nchild
    children = []
    for i in range(nchild):
        cx = margin + i * child_w + child_w / 2
        nw = child_w * rng.uniform(0.5, 0.75)
        bb = [cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2]
        b.node(bb, size=10)
        b.edge([_border_point(root, "b"), _border_point(bb, "t")])
        children.append(bb)
    # optional leaves
    if cy + nh + 90 < H and rng.random() < 0.7:
        ly = cy + nh + rng.randint(60, 90)
        for parent in children:
            if rng.random() < 0.6:
                nleaf = rng.randint(1, 2)
                for j in range(nleaf):
                    cx = (parent[0] + parent[2]) / 2 + (j - 0.5) * 70
                    bb = [cx - 40, ly - 20, cx + 40, ly + 20]
                    if bb[0] < 10 or bb[2] > W - 10:
                        continue
                    b.node(bb, size=9)
                    b.edge([_border_point(parent, "b"), _border_point(bb, "t")])


FAMILIES = [fam_horizontal_flow, fam_vertical_stack, fam_panel_grid, fam_tree]


def generate_one(seed: int) -> tuple[dict[str, Any], str]:
    rng = random.Random(seed)
    W = rng.choice([960, 1100, 1200, 1280, 1376])
    H = rng.choice([640, 720, 768, 820])
    b = Builder(rng)
    fam = rng.choice(FAMILIES)
    fam(b, W, H)
    return b.program(W, H), fam.__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--outdir", default="outputs/synth/preview")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    ok_render = 0
    for i in range(args.n):
        seed = args.seed * 1_000_000 + i
        program, fam = generate_one(seed)
        stem = outdir / f"sample_{i:05d}"
        (stem.with_suffix(".vp_program.json")).write_text(json.dumps(program, indent=1))
        rec = {"stem": str(stem), "family": fam, "canvas": program["canvas"],
               "counts": program["counts"], "rendered": False}
        if not args.no_render:
            try:
                compile_program_to_drawio(program, str(stem) + ".drawio", font_family=FONT_FAMILY)
                res = export_drawio_png(str(stem) + ".drawio", str(stem) + ".png")
                rec["rendered"] = bool(res.get("ok"))
                if rec["rendered"]:
                    ok_render += 1
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)[:200]
        manifest.append(rec)
        print(f"[{i:05d}] {fam:22s} {program['counts']['total']:3d} prims  rendered={rec['rendered']}", flush=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {args.n} programs, {ok_render} rendered -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
