# CCB v142 — Agentic VLM + local-snap (Stage 0 of the learned-pivot)

First result from the NEW mechanism (not greedy sweeping). A frontier VLM (Claude
Opus 4.8 via OAuth) proposes missing primitives from (target, current-render); each
is added only if the harness score improves, with a local-search SNAP that fixes
VLM coordinate imprecision. See tools/agentic_render_loop.py.

- Base: v139 (0.879980, harness canonical convention).
- v142 final: **0.881532** (harness canonical, no expected_tokens) = **+0.001552**.
  (On the expected-tokens scale used during the run: 0.881132 -> 0.882668.)
- Added 3 text + 4 edge primitives that greedy tweaking could never add (670 -> 677).
- edge_f1 0.8021 -> 0.8041; ocr_semantic_f1 0.9085.

Significance: validated that VLM perception is correct but coordinate precision is the
bottleneck (snap converting a -0.0003 near-miss into the biggest +0.00105 gain proved
it). This is exactly what fine-tuning Qwen3-VL on coordinate data is expected to fix.
