#!/usr/bin/env python3
"""End-to-end png->drawio reconstruction eval (the headline render-and-compare metric).

image -> VLM (emits DSL) -> vp_dsl.parse -> compile_program_to_drawio -> render ->
evaluate_drawio_variants(target_image, drawio) -> score. This is the SAME harness
score the whole project optimizes, so the CCB number is directly comparable to the
0.88 lineage, and it is the RL reward signal.

Backends: openai-compatible (vLLM / OpenRouter), or claude (CLI via OAuth).

Usage (server, against a local vLLM serving the fine-tuned model):
  python3 tools/eval_reconstruct.py --backend openai \
    --base-url http://localhost:8000/v1 --api-key-env NONE --model <ckpt-name> \
    --images uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg data/synth/val/*.png
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

from tools.vp_dsl import INSTRUCTION, parse  # noqa: E402
from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.variant_eval import evaluate_drawio_variants  # noqa: E402

FONT_FAMILY = "Helvetica"


def _data_uri(path: str) -> str:
    media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{media};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def call_openai(image: str, model: str, base_url: str, api_key: str, max_tokens: int):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    resp = client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": INSTRUCTION},
            {"type": "image_url", "image_url": {"url": _data_uri(image)}}]}])
    return resp.choices[0].message.content or ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="openai", choices=["openai"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key-env", default="NONE")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", nargs="+", required=True, help="target PNGs/JPGs (globs ok)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--outdir", default="outputs/recon_eval")
    args = ap.parse_args()

    paths = []
    for g in args.images:
        paths += sorted(glob.glob(g)) if any(c in g for c in "*?[") else [g]
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    api_key = "" if args.api_key_env in ("", "NONE") else os.environ.get(args.api_key_env, "")

    scores = []
    for i, img in enumerate(paths):
        W, H = Image.open(img).size
        try:
            dsl = call_openai(img, args.model, args.base_url, api_key, args.max_tokens)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {Path(img).name}: backend error {exc}", flush=True)
            continue
        prog = parse(dsl, W, H)
        stem = out / f"recon_{i:04d}"
        (stem.with_suffix(".dsl.txt")).write_text(dsl)
        try:
            compile_program_to_drawio(prog, str(stem) + ".drawio", font_family=FONT_FAMILY)
            row = evaluate_drawio_variants(img, [str(stem) + ".drawio"], export=True)[0]
            sc = float(row.get("score") or 0.0)
            m = row.get("metrics") or {}
            ef = (m.get("edge") or {}).get("f1", 0)
            of = max((m.get("ocr") or {}).get("f1", 0) or 0,
                     ((m.get("ocr") or {}).get("semantic") or {}).get("f1", 0) or 0)
            scores.append(sc)
            print(f"  [{i}] {Path(img).name:40s} score={sc:.4f} edge_f1={ef:.3f} ocr_f1={of:.3f} "
                  f"prims={prog['counts']['total']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {Path(img).name}: compile/render error {exc}", flush=True)

    if scores:
        mean = sum(scores) / len(scores)
        print(f"\n[recon] n={len(scores)} mean_score={mean:.4f} "
              f"min={min(scores):.4f} max={max(scores):.4f}")
        (out / "summary.json").write_text(json.dumps(
            {"model": args.model, "n": len(scores), "mean": mean, "scores": scores}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
