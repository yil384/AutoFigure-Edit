# png→drawio learned pivot — server runbook (GPU box)

The CV + greedy render-and-compare pipeline plateaued at ~0.88 on one image. We pivot
to a fine-tuned VLM: **png → DSL (compact vp_program) → draw.io**, trained on synthetic
(png, program) pairs (SFT), then refined with render-and-compare RL.

**Backbone (data-verified, see `data/bench_results/`):** primary **Qwen3-VL-8B-Instruct**
for fast iteration; **Qwen3-VL-32B-Instruct** for the production run (its zero-shot
grounding prior is ~2.7× the 8B's, matches Qwen2.5-VL-72B at half size). Apache-2.0.
On dense figures, frontier models are *worse* at coordinates (GPT-5.2 0.012, Gemini-3.1
0.264, Qwen2.5-VL-72B 0.135 IoU) — this task is won by task-specialized fine-tuning + RL,
which require open weights. **All zero-shot models are poor → fine-tuning is the point.**

## 0. Environment
```bash
git clone git@github.com:yil384/AutoFigure-Edit.git && cd AutoFigure-Edit
pip install -r requirements.txt 2>/dev/null || pip install pillow opencv-python pytesseract numpy openai requests
# Training stack:
pip install "llamafactory[torch,metrics]" deepspeed   # or: git clone hiyouga/LLaMA-Factory && pip install -e .
pip install vllm                                       # for serving + eval
# Rendering + scoring deps (REQUIRED for eval — harness renders drawio and OCRs it):
#  - drawio CLI is Electron/headless: install drawio-desktop .deb/AppImage and run under xvfb
sudo apt-get install -y tesseract-ocr xvfb libgbm1 libasound2
#  - get drawio CLI, then wrap with xvfb-run; set DEFAULT_DRAWIO_CLI / qa.py to use `xvfb-run -a drawio`
```
**Gotcha:** `visual_primitives/qa.py:export_drawio_png` shells out to the drawio CLI. On a
headless server it MUST run under `xvfb-run` (no display otherwise). Verify with a single
render before mass jobs:
```bash
python3 -c "from visual_primitives.qa import export_drawio_png as e; print(e('artifacts/visual_primitives/ccb_v142_agentic_vlm_snap/ccb_pure_native_current_best.drawio','/tmp/t.png'))"
```

## 1. Generate synthetic training data (the workhorse)
Render throughput (drawio CLI ~2-3s/img) is the bottleneck — **shard across CPU cores**:
```bash
# 16 parallel shards × 2000 = 32k pairs. Each writes data/synth/shardNN/{sample_*.png,*.vp_program.json}
for i in $(seq 0 15); do
  xvfb-run -a python3 tools/synth_diagram_gen.py --n 2000 --seed $i --outdir data/synth/shard$i &
done; wait
```
Bias toward dense academic layouts as needed (edit FAMILIES / counts in `tools/synth_diagram_gen.py`).
Mine real `.drawio` from GitHub later for realism (see memory note / strategy report).

## 2. Export SFT dataset
```bash
python3 tools/export_sft_data.py --in-dirs data/synth/shard* --out-dir data/sft --val-frac 0.02
# Register in LLaMA-Factory: merge data/sft/dataset_info.json into LLaMA-Factory's data/dataset_info.json,
# and point its file_name to the absolute path of data/sft/png2drawio_synth_train.json
```

## 3. SFT (LoRA)
```bash
llamafactory-cli train configs/sft/qwen3vl_8b_lora.yaml
llamafactory-cli export ...   # merge LoRA -> saves/qwen3vl-8b-png2drawio-merged
```

## 4. Serve + evaluate (the REAL metric = render-and-compare)
```bash
vllm serve saves/qwen3vl-8b-png2drawio-merged --port 8000 --limit-mm-per-prompt image=2 &
# Headline reconstruction score (CCB is directly comparable to the 0.88 lineage):
xvfb-run -a python3 tools/eval_reconstruct.py --backend openai \
  --base-url http://localhost:8000/v1 --api-key-env NONE \
  --model saves/qwen3vl-8b-png2drawio-merged \
  --images uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg data/synth/shard0/sample_0000*.png
# Grounding IoU vs the frontier/Qwen baselines in data/bench_results/:
python3 tools/grounding_iou_bench.py --backend openai --base-url http://localhost:8000/v1 \
  --api-key-env NONE --model saves/qwen3vl-8b-png2drawio-merged \
  --manifest data/bench_results/eval_manifest.json
```
**Targets (year-one realistic, not pixel-lossless):** beat the 0.88 render score on CCB AND
generalize across held-out synthetic; shape/label-F1 ≥ 0.9, edge-connectivity-F1 0.7–0.8.

## 5. RL (next stage, after SFT works)
GRPO with reward = our harness score (`visual_primitives/variant_eval.score_variant`) on the
rendered DSL, gated on "drawio compiled OK". Precedents: RLRF (arXiv 2505.20793), RRVF
(2507.20766). Stable on-ramp first: **RAFT / rejection-sampling** — sample N DSLs per image,
render+score, SFT on the winners, repeat (reuses tools/eval_reconstruct scoring). Then GRPO.

## Key files
- `tools/synth_diagram_gen.py` — synthetic (png, vp_program) generator (4 layout families).
- `tools/vp_dsl.py` — DSL ↔ vp_program (serialize/parse, normalized [0,1000], `INSTRUCTION`). Round-trip validated.
- `tools/export_sft_data.py` — pairs → LLaMA-Factory ShareGPT SFT json.
- `tools/eval_reconstruct.py` — image→DSL→render→harness score (headline metric + RL reward).
- `tools/grounding_iou_bench.py` — coordinate-IoU benchmark (frontier vs Qwen baselines saved).
- `tools/agentic_render_loop.py` — Stage-0 VLM+snap loop (no-train baseline; produced v142=0.8815).
- `visual_primitives/{emit_drawio,variant_eval,qa}.py` — compile vp_program→drawio, score, render.
- `data/bench_results/` — the grounding-IoU comparison evidence.
- `artifacts/visual_primitives/ccb_v142_agentic_vlm_snap/` — current best (0.881532) + the v139 base.

## Gotchas learned
- **Font:** compile with `font_family="Helvetica"` (harness default). "Arial" silently drops OCR ~0.05.
- **Headless render:** drawio CLI needs `xvfb-run`.
- **Render throughput** is the data bottleneck — shard generation across cores.
- **Coord grid:** DSL uses [0,1000] (Qwen-native). RL refines sub-grid precision later.
