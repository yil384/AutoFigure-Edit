# SVG → drawio (mxgraph) Converter

Convert a raster figure (PNG) + its vtracer-traced SVG into an editable
drawio (`.drawio` / mxgraph XML) file. This README captures the **lessons
learned, the architecture, and the open TODO** so the next iteration
doesn't repeat work that was already tried.

> **Status: WIP**. The pipeline produces a structurally rich, partially
> editable drawio file (~50 native shapes, ~20 bound connector edges,
> ~1k path stencils consumed/cleaned). It is NOT yet on par with a
> hand-built drawio file — many small labels remain as raw path
> stencils, pixel diff to the original PNG sits around 22–25 on a
> 1376×768 canvas, and the recognition coverage is brittle on figures
> with unusual layouts (rotated boxes, decorative arrows, multi-line
> nested labels). See the **Open TODO** section at the bottom.

---

## What this package does today

```
input:  source.svg  (vtracer or PDF→SVG of a paper figure)
        source.png  (the original raster — used for OCR and PNG-driven
                     CV fallbacks)
output: out.drawio  (mxgraph XML — openable in diagrams.net or the
                     drawio desktop app, partially editable)
```

Run:

```bash
python -m svg_to_drawio source.svg source.png -o out.drawio \
    --cluster-cache /tmp/clusters.json \
    --ocr-cache /tmp/ocr.json
```

Required environment:

- `CUDA_VISIBLE_DEVICES=<one-free-gpu>` (EasyOCR uses GPU)
- `CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...` *(only if you want the
  Claude-Haiku recovery / vibrant-container OCR passes — see
  caveats below)*
- Docker with the `drawio-export-cmu` image, for the render-diff
  line-repair loop (`docker run drawio-export-cmu --format png ...`).

Render to PNG for visual verification:

```bash
docker run --rm -v "$PWD/dir":/data drawio-export-cmu \
    --format png --scale 2 --transparent out.drawio
```

---

## Architecture (`svg_to_drawio/`)

```
svg_to_drawio/
├── __init__.py        # exports convert_pair
├── __main__.py        # CLI entry: python -m svg_to_drawio
├── convert.py         # main pipeline orchestrator
├── parse.py           # SVG path parsing (re-exports v2/v4/v5)
├── panels.py          # full-canvas bg + panel-bg detection
├── glyphs.py          # glyph candidates + clustering + outlier removal
│                      # + is_rectangle_silhouette + split_spatially_joined
├── ocr.py             # EasyOCR + Claude verify + 9 recovery passes
├── dedup.py           # text cluster dedup
├── edges.py           # line/arrowhead detection + snap_to_shape +
│                      # emit_edge_cell (with source/target binding)
├── icons.py           # icon clustering + atomic SVG image cells
├── colors.py          # hex parsing, dark detection (re-exports v5)
├── emit.py            # drawio cell builders (one function per z-order
│                      # layer) + emit_native_shape + emit_simple_line
├── border_detect.py   # PNG-driven: chroma rect + adaptive-threshold +
│                      # line-segment detectors (catches vtracer misses)
├── semantic.py        # match text clusters to container paths +
│                      # classify shape + emit ONE native shape with
│                      # editable text + vibrant container OCR
└── auth.py            # Anthropic client factory (api_key vs OAuth)
```

Legacy `svg_to_drawio_v{2,4,5,6,7,9,11,12,14,17}.py` at the repo root
are kept because the package re-exports specific helpers from them.

### Pipeline steps (numbered to match the printed log)

1. **Parse SVG paths** — extract path data, fill, fill-opacity
2. **Drop canvas-bg** — full-canvas white rect (paper background)
3. **Detect panel backgrounds** — large pale rects (quadrant panels)
4. **Cluster glyphs** — small dark paths → text-line clusters
5. **OCR** — EasyOCR full-image pass + per-cluster crop +
   Claude verify fallback + 9 recovery passes
6. **Dedup** — merge substring text fragments, hyphen continuation,
   spatial split, stacked multi-line merge
7. **Detect edges** — line-shaped paths + arrowheads → pair
8. **Cluster icons** — non-text non-edge paths into atomic icons
9. **Emit cells in z-order**:
   - canvas (invisible locked frame)
   - panel backgrounds
   - missing container rects (PNG chroma + adaptive-thresh detectors)
   - **semantic native shapes** (text fused into rounded\_rect/ellipse/
     cylinder/diamond/hexagon with editable `value=`)
   - edges (with `source`/`target` snapped to native shapes when possible)
   - non-icon singleton stencils
   - unattached arrowheads
   - atomic SVG icon cells
   - remaining text cells (clusters not absorbed by native shapes)
10. **Render-diff line repair** — render drawio via Docker, diff vs
    source PNG, find missing horizontal/vertical line segments, emit
    them as bound edges, repeat until convergence (3 iterations max)
11. **Write final drawio**

---

## What works (proven)

- **45–50 native drawio shapes** with `value="<text>"` and built-in
  drawio styles (`rounded=1`, `ellipse;`, `rhombus`, `shape=cylinder3`,
  `shape=hexagon`). User can double-click and edit text in place.
- **Edge ↔ shape binding**: ~20 connector edges have `source` /
  `target` IDs + `exitX`/`exitY`/`entryX`/`entryY` ratios. Drag a
  bound shape and arrows follow. Tested with manual XML +60 px shift.
- **PNG-driven container detection** catches rounded rects vtracer
  missed (chroma + adaptive threshold + line-pair detectors).
- **Multi-line label merging**:
  - hyphen continuation (`Bloch-` + `Interfero-` + `meters` →
    `Bloch-Interferometers`)
  - vertically-stacked OCR fragments (`Verilog SFT` + `corpus (87K)` →
    one cluster, then one native shape)
  - same-row spatial split (one cluster spanning two adjacent labels
    gets split, each gets its own native shape)
- **Vibrant-container OCR fallback** for white-text-on-color pills
  (Stage 0/1/2/3 headers, "Final 9B-V" orange, AT-GRPO banner) —
  EasyOCR per-crop because identify\_glyph\_candidates filters by
  `dark` and never reaches them.
- **Render-diff line repair** with text-bbox mask: catches missing
  panel-divider strokes (~30–45 per figure) without re-drawing through
  rendered text.
- **OAuth client factory** for Claude — works for Haiku, falls back
  silently when token is unset or rate-limited.

---

## What doesn't work yet — open TODO

These are the ranked gaps that block "human-quality drawio":

### 1. Coverage: ~30 boxes per figure still emit as raw stencils
The container-matching heuristic only fires when:
- The text cluster's bbox fits inside a "pale" path (RGB sum ≥ 600
  or vibrant 200–650 + chroma ≥ 40),
- AND the container is between 1.3× and 4× the text bbox size in
  each dimension,
- AND the path didn't already get classified as a panel-background.

Real-world failure modes:
- Containers with PURE WHITE interior (#ffffff) but only a thin gray
  stroke outline. vtracer encodes the stroke as TWO paths (outer +
  inner) and the outer is then classified as panel_bg, skipped.
- Containers with a faint gradient that vtracer splits into 5+ paths
  (e.g. shadowed buttons in modern figures).
- Small label boxes where the text bbox is wider than the container
  by 1–2 px due to OCR padding.

**Suggested fix**: replace the heuristic match with a learned
classifier — given a text bbox + 8-neighbor crop, predict which of
{rect, rounded\_rect, ellipse, diamond, cylinder, hexagon, none}.
A small CNN or a single Claude vision call per cluster batched in
groups of 10–20 would work. Avoids the "tuning thresholds forever"
trap we hit.

### 2. Edge endpoint detection is line-shape-only
`is_line_shaped` only catches paths whose bbox is 5× longer than
wide. CURVING connectors (the big "Iterative GNN Retraining" feedback
loop, the Final-arrow in CodeV's AT-GRPO panel) are stored by vtracer
as **filled polygons** that occupy a roughly square bbox. They get
clustered as icons instead of edges.

**Suggested fix**: detect arrowheads first (small triangles, already
done in `edges.py`), then for each arrowhead, ray-march back along
the most-likely direction (inferred from arrowhead orientation) and
follow a connected path of dark pixels to find the source endpoint.
This is what Arrow R-CNN / DAMO-YOLO do — output Arrow Start / Arrow
End / Arrow Body as three keypoint classes. See research notes in
the next section.

### 3. Vtracer is the wrong tool for the input we have
The "good" figures used as test inputs (`ccb48c…svg`, `codev_v11.svg`)
were generated by a PDF→SVG converter (probably `pdf2svg`/`cairosvg`)
NOT by vtracer. They use `fill-rule="evenodd"` and 13+ subpaths per
shape (letters with holes, panels with cutouts). When you run vtracer
on a figure-rendered-to-PNG, you get:
- 6700+ paths (vs 3000 in the PDF→SVG version)
- Single-subpath shapes with implicit nonzero fill
- Per-path `transform="translate(...)"` that our parser ignores
- Color quantization that flattens the brick stack's gradient bricks
  into 1–2 paths instead of 5

We tested CLAHE + vtracer with `filter_speckle=2,4,8` and
`color_precision=6,7,8` — every config either produced too many
noise singletons (~900 vs ~140 baseline) or lost the Hardware Agent
box entirely. **Conclusion: never re-trace with vtracer**. If a
better trace is needed, look at:
- The original source PDF (regenerate the SVG with `pdf2svg`)
- LIVE (Layer-wise Image Vectorization), DiffVG, or commercial
  vectorizers (Adobe Illustrator's Image Trace)
- A diffusion-based vectorizer (StarVector, Im2Vec) — heavy and slow
  but produces clean shape primitives directly

A pre-processing utility — `tmp/bake_translates.py` — is kept in
the repo for the case where you DO want to use sensitive-vtracer
output: it walks the SVG and bakes each path's `transform="translate"`
into its `d` attribute so `parse_svg_paths` sees correct absolute
coords.

### 4. Container detection misses inner sub-rects
Faint borders inside a panel (P1's "Validity Check", "GNN Surrogate
Filtering", "Classical Metrics Ranker" sub-boxes in the LDPC figure)
are too thin for adaptive-threshold contour detection. Tried:
- chroma mask + morph-close (3×3, 5×5, 9×9 kernels) → catches outer
  but not inner sub-boxes
- adaptive-threshold INV + contour + 4–12 vertex polygon → 20+
  candidates but they all overlap an existing path
- Sobel + run-length detection of vertical/horizontal segments → finds
  vertical lines at x=556 (HW Agent left) and x=766 (Ver Agent left)
  reliably but never finds matching horizontals (interior content
  edges drown them)

**Suggested fix**: combined LSD line detector + line-segment pairing
(SLSD-Net or LCNN), or just a VLM call: "list the bounding boxes of
all rounded rectangles you see in this image."

### 5. Pixel diff vs original is 22–25 mean
Even with everything working, the rendered drawio differs from the
source PNG by ~22 mean per channel. Sources:
- vtracer-traced bbox is 1–2 px off from the actual PNG border
  position (sub-pixel positioning) → added strokes don't perfectly
  align
- font metrics in drawio (DejaVu Sans / Helvetica) ≠ figure's
  original font; anti-aliasing differs
- color quantization flattens icon details (brick stack)

These are mostly cosmetic and the user said the BIG visible issues
are structural, not pixel-level. Not worth optimizing further.

### 6. Self-loops + spurious bindings
`snap_to_shape` rejects self-loops (source==target). But it still
binds some perimeter-stroke fragments that happen to start/end near
two adjacent shape edges. Result: a tiny edge from shape\_24's right
side to shape\_25's left side when there shouldn't be one.

**Suggested fix**: also require the line's bbox length to be ≥ the
sum of the two endpoints' snap distances + a margin. Truly external
connectors are LONG; perimeter strokes are short.

### 7. Vibrant pill OCR can bleed into adjacent labels
When EasyOCR crops a Stage 2 pill (280×31), it sometimes captures
text from the boxes below if they're within the crop padding. Saw
"Stage 2: Parallel Single-RL DAPO~V-SFT DAPO-V 9B" — Stage 2 text
+ adjacent labels concatenated.

**Suggested fix**: tighter crop (no padding) or filter OCR results
to those whose center is strictly inside the pill's bbox (not just
overlapping).

### 8. OAuth Claude limited
The `CLAUDE_CODE_OAUTH_TOKEN` works for Haiku-4.5 but:
- Sonnet/Opus return 429 (subscription limits)
- Periodically returns 403 "OAuth authentication is currently not
  allowed for this organization" — recovery passes silently skip.

**Suggested fix**: switch to a real `ANTHROPIC_API_KEY` for
production use. The `auth.py` factory already supports both.

---

## Lessons learned (the painful ones)

### "Just one more heuristic" doesn't converge
The earliest version was simple: vtracer → emit each path as a
stencil. Each iteration since added a heuristic that fixed one
visible bug but introduced a slightly different one elsewhere:

- "filter glyph-sized subpaths inside text bboxes" caught letter
  fragments inside container paths, but also dropped the agent
  header strips (74×35 light blue rects). Fixed with `is_pale_container`
  exemption.
- "drop letter-only icons" stopped the title 'Pipeline' letters from
  rendering as image cells, but the criterion (small + dark +
  multi-subpath) also matched some 'real' icon fragments. Fixed by
  using LARGEST-path criterion not all-paths.
- "iterative render-diff line repair" added 36 missing lines
  initially, but iters 2–3 kept finding the SAME lines because the
  emit position was sub-pixel-off from where the detector wanted.
  Fixed with dedupe by quantized coords.
- "vibrant container OCR" caught Stage pills but bled adjacent labels
  into the value. Fixed by lowercase-start filter + dup-bbox check.

**Lesson**: when each fix introduces a fix-for-the-fix, the
abstraction is wrong. We're playing whack-a-mole at the wrong layer.
The semantic recognition needs to be a single LEARNED classifier,
not a hand-tuned heuristic.

### Pixel diff is the wrong metric
For early phases we tracked mean pixel diff vs the source PNG. It
went 16.35 (v17) → 17.84 (v18 fillstroke) → 17.57 → 22.13 (v20 native
shapes) → 24.99 (v23 vibrant). Each "real improvement" raised the
metric because: more visible borders = more pixels different from
the original; native shapes use Helvetica not the figure's actual
font, etc.

**Lesson**: pixel diff measures fidelity, not usability. The user
explicitly said they want a usable drawio, not a pixel-perfect render.
The right metric is "number of cells that are not raw stencil
dumps" + "percentage of edges with source/target bindings" + "would
this open cleanly in drawio editor."

### vtracer is fragile to input
A clean PDF→SVG produces 3000 sensible paths. CLAHE + sensitive
vtracer on the same PNG produces 6700+ paths with worse structure.
The "more sensitive trace catches more detail" intuition is wrong —
sensitivity adds noise faster than signal.

**Lesson**: trust the source-of-truth vector data if you have it.
If you don't, the bottleneck is the trace, not the pipeline.

### Caches are landmines
`tmp/ocr.json` was loaded by `_load_full_ocr` from a previous run's
figure, polluting the current figure's OCR with phrases like "Figure
2: ML-Driven Codesign of QEC". Took 20 minutes to track down.

**Lesson**: caches must be namespaced by input file hash. The current
`<svg-stem>.{clusters,ocr}.json` convention is decent but the global
`tmp/ocr.json` fallback is dangerous; remove it.

### OCR fragments cause cascading bugs
"4-" next to "4-6" both with 1 glyph each defeated the dedup `<`
comparison; "7—" with em-dash didn't substring-match "7-9"; a partial
"Stage 2…" OCR with text bleed pretended to be a valid pill label.
Each of these required a one-off fix.

**Lesson**: OCR cache needs to be sanity-checked AFTER OCR by an LLM
that knows the figure's domain, not just per-cluster crops. A
"reasonableness pass" on the full OCR set would catch fragments and
typos at much lower cost than tuning each recovery heuristic.

### Z-order is load-bearing
Native shapes had to be emitted AFTER stencils+icons because of
draw-order, otherwise leftover inner-fill paths cover the text.
Edges with source/target work regardless of XML position, but
visible stencils don't. Took multiple iterations to get right.

**Lesson**: write down the z-order layers explicitly (canvas → panel
bg → missing rects → singletons → unattached arrows → icons → native
shapes → edges → text) and gate every emit by that table.

---

## Research notes (what to try next)

From the Dec 2025 / 2026 literature:

1. **Arrow-Guided VLM** (arXiv 2505.07864): fine-tune DAMO-YOLO with
   9 classes including "Arrow Start" and "Arrow End" as separate
   keypoint boxes → assign text by IoU → match arrow bodies to
   start/end by IoU → serialize as structured prompt → GPT-4o or
   Claude consumes the structured prompt and emits drawio XML. 80 →
   89% on flowchart QA.
2. **Flowchart2Mermaid** (arXiv 2512.02170): pure VLM (GPT-4.1 /
   Gemini-2.5-Flash) end-to-end on 200 FlowVQA images → Mermaid
   syntax, entity F1 ≈ 0.98, relationship F1 ≈ 0.97. No detector.
   Caveat: VLMs corrupt edge `source`/`target` IDs in raw drawio XML.
3. **GenAI-DrawIO-Creator** (arXiv 2601.05162): Claude 3.7 emits raw
   drawio XML but needs strong XML validation (missing `id="0"/"1"`
   roots, dangling edges).
4. Open source: `drawio-mcp`, `imagetodrawio.com` (paid).

**Recommendation for the next iteration of this package**:
- Stop adding heuristics. Replace `semantic.find_container_for_cluster`
  with a single per-figure Claude/Gemini vision call returning a JSON
  list of `{bbox, shape, text}`.
- Fine-tune YOLOv8 (already in repo: `yolov8l-world.pt`) on Roboflow's
  flowchart-etfvh dataset for arrow Start/End/Body detection.
- Keep the existing path-stencil layer as a fallback for everything
  the VLM doesn't recognize.

---

## File layout (this branch)

```
svg_to_drawio/                      ← the canonical package
svg_to_drawio_v{2,4,5,6,7,9,11,12,14,17}.py   ← legacy helpers re-exported
png_svg_to_drawio.py                ← friendly CLI wrapper
tmp/bake_translates.py              ← vtracer-output preprocessing tool
```

Test inputs and caches sit in `tmp/` and `outputs/` (gitignored).

---

## Test cases

Two figures the pipeline has been exercised against:

1. **LDPC / QEC figure** (1376×768, 4 quadrants). Best output:
   `outputs/v17_final/v18_borders.drawio` — has all 4 Agent column
   header strips visible, ZX colored circles, multi-line `Bloch-Messiah`
   / `Interferometers` correctly merged.
2. **CodeV-R1 Training Pipeline** (1376×768). Best output:
   `outputs/codev_v11_run/codev_v11_FINAL.drawio` — Stage 0/1/2/3
   colored pill headers as native shapes, DAPO-V → 9B-V-RL connector
   bound, Verilog SFT corpus boxes editable.

Neither matches a hand-built drawio yet. See the open TODO above.
