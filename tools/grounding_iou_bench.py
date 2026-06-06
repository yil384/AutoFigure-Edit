#!/usr/bin/env python3
"""Coordinate-grounding IoU benchmark for choosing the png->drawio backbone.

Decides between Qwen3-VL-8B / 32B / Qwen2.5-VL (and any frontier baseline) by
measuring how accurately a VLM can LOCATE a labelled element in a figure — the
exact capability our task depends on. We have exact ground-truth boxes from the
vp_programs (synthetic) and the real CCB program, so this is a clean test.

Backends (pluggable):
  --backend claude     : Claude via `claude -p` + CLAUDE_CODE_OAUTH_TOKEN (works here now)
  --backend openai     : any OpenAI-compatible server — set --base-url + --api-key-env + --model
                         * OpenRouter: --base-url https://openrouter.ai/api/v1 --model qwen/qwen3-vl-8b-instruct
                         * local vLLM: `vllm serve Qwen/Qwen3-VL-8B-Instruct` then
                           --base-url http://localhost:8000/v1 --api-key-env NONE --model Qwen/Qwen3-VL-8B-Instruct
                         * DashScope: --base-url https://dashscope.../compatible-mode/v1 --model qwen3-vl-8b-instruct

Metric: mean IoU vs ground-truth box + fraction with IoU>=0.5, broken down by
source (synthetic/ccb) and kind (text/box).

Helpers return (value, error): (result, None) on success, (None, 'msg') on failure.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LOCATE_PROMPT = (
    "This image is {W}x{H} pixels (origin top-left, x right, y down). "
    "Locate the single element that {what} \"{label}\". "
    "Return ONLY JSON: {{\"box\":[x0,y0,x1,y1]}} in PIXEL coordinates of this image "
    "(x0<x1, y0<y1). If the element is not present, return {{\"box\":null}}. No other text."
)


# --------------------------------------------------------------------------- #
# Eval-set construction (exact GT from vp_programs)
# --------------------------------------------------------------------------- #
def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _inside(c, b):
    return b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]


def build_items_from_pair(program: dict, image: str, source: str,
                          max_text: int, max_box: int, rng: random.Random) -> list[dict]:
    prims = program.get("primitives", [])
    W = int((program.get("canvas") or {}).get("width") or 0)
    H = int((program.get("canvas") or {}).get("height") or 0)
    texts = [p for p in prims if p.get("type") == "text"
             and len(str(p.get("text", "")).strip()) >= 4
             and any(c.isalpha() for c in str(p.get("text", "")))]
    regions = [p for p in prims if p.get("type") == "region"]
    items = []
    # text-grounding items
    chosen_t = texts[:] ; rng.shuffle(chosen_t)
    for t in chosen_t[:max_text]:
        items.append({"image": image, "W": W, "H": H, "source": source, "kind": "text",
                      "what": "is the text reading", "label": str(t["text"]).strip(),
                      "gt": [float(v) for v in t["bbox"]]})
    # box-grounding items: a region uniquely labelled by one interior text
    boxes = []
    for r in regions:
        labs = [t for t in texts if _inside(_center(t["bbox"]), r["bbox"])]
        if len(labs) == 1:
            boxes.append((labs[0], r))
    rng.shuffle(boxes)
    for lab, r in boxes[:max_box]:
        items.append({"image": image, "W": W, "H": H, "source": source, "kind": "box",
                      "what": "is the box/panel containing the label", "label": str(lab["text"]).strip(),
                      "gt": [float(v) for v in r["bbox"]]})
    return items


def build_manifest(args, rng) -> list[dict]:
    items: list[dict] = []
    # synthetic pairs
    for prog_path in sorted(Path(args.synth_dir).glob("*.vp_program.json"))[: args.synth_images]:
        png = prog_path.with_name(prog_path.name.replace(".vp_program.json", ".png"))
        if not png.exists():
            continue
        prog = json.loads(prog_path.read_text())
        items += build_items_from_pair(prog, str(png), "synthetic",
                                       args.synth_text, args.synth_box, rng)
    # real CCB pair (show the SOURCE image; GT is program-space == source dims)
    if args.ccb_program and Path(args.ccb_program).exists():
        prog = json.loads(Path(args.ccb_program).read_text())
        items += build_items_from_pair(prog, args.ccb_image, "ccb",
                                       args.ccb_text, args.ccb_box, rng)
    return items


# --------------------------------------------------------------------------- #
# Coordinate parsing
# --------------------------------------------------------------------------- #
def parse_box(text: str, W: int, H: int, coord_space: str):
    if not text:
        return None
    m = re.search(r"\{[^{}]*\"(?:box|bbox|bbox_2d)\"[^{}]*\}", text, re.DOTALL)
    blob = m.group(0) if m else (text[text.find("{"):text.rfind("}") + 1] if "{" in text else text)
    box = None
    try:
        d = json.loads(blob)
        box = d.get("box") if isinstance(d, dict) else None
        if box is None and isinstance(d, dict):
            box = d.get("bbox") or d.get("bbox_2d")
    except Exception:  # noqa: BLE001
        nums = re.findall(r"-?\d+\.?\d*", blob)
        if len(nums) >= 4:
            box = [float(x) for x in nums[:4]]
    if not box or (isinstance(box, list) and len(box) < 4):
        return None
    box = [float(v) for v in box[:4]]
    # normalize handling
    mx = max(abs(v) for v in box)
    if coord_space == "norm1000" or (coord_space == "auto" and mx <= 1000 and (W > 1000 or H > 1000)):
        box = [box[0] * W / 1000.0, box[1] * H / 1000.0, box[2] * W / 1000.0, box[3] * H / 1000.0]
    x0, y0, x1, y1 = box
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _data_uri(path: str) -> str:
    media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{media};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def locate_claude(prompt: str, image: str, model: str) -> tuple[str | None, str | None]:
    import subprocess
    media = "image/png" if image.lower().endswith(".png") else "image/jpeg"
    content = [{"type": "text", "text": prompt},
               {"type": "image", "source": {"type": "base64", "media_type": media,
                                            "data": base64.b64encode(Path(image).read_bytes()).decode()}}]
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    try:
        proc = subprocess.run(
            ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", model,
             "--disallowedTools", "Bash", "Edit", "Write", "Read", "Skill", "Agent", "WebSearch", "WebFetch"],
            input=json.dumps(msg) + "\n", text=True, capture_output=True, env=env, timeout=180)
    except Exception as exc:  # noqa: BLE001
        return None, f"claude failed: {exc}"
    out = None
    for line in proc.stdout.splitlines():
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if obj.get("type") == "result":
            out = obj.get("result")
        elif obj.get("type") == "assistant":
            for b in (obj.get("message") or {}).get("content", []):
                if b.get("type") == "text" and b.get("text", "").strip():
                    out = b["text"]
    return (out, None) if out else (None, "no claude text")


def locate_openai(prompt: str, image: str, model: str, base_url: str, api_key: str) -> tuple[str | None, str | None]:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_uri(image)}}]}])
        return resp.choices[0].message.content, None
    except Exception as exc:  # noqa: BLE001
        return None, f"openai backend failed: {exc}"


def run_item(item: dict, args, api_key: str) -> dict:
    prompt = LOCATE_PROMPT.format(W=item["W"], H=item["H"], what=item["what"], label=item["label"])
    if args.backend == "claude":
        text, err = locate_claude(prompt, item["image"], args.model)
    else:
        text, err = locate_openai(prompt, item["image"], args.model, args.base_url, api_key)
    # Fair scoring: try the requested coord space; for 'auto' try BOTH pixel and
    # norm1000 and keep the better, so no model is penalised for its coordinate frame.
    spaces = ["pixel", "norm1000"] if args.coord_space == "auto" else [args.coord_space]
    best_pred, best_score = None, 0.0
    if not err:
        for cs in spaces:
            pred = parse_box(text or "", item["W"], item["H"], cs)
            if not pred:
                continue
            sc = iou(pred, item["gt"])
            if best_pred is None or sc > best_score:
                best_pred, best_score = pred, sc
    return {**{k: item[k] for k in ("source", "kind", "label", "gt")},
            "pred": best_pred, "iou": round(best_score, 4), "err": err, "raw": (text or "")[:160]}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude", choices=["claude", "openai"])
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--coord-space", default="auto", choices=["auto", "pixel", "norm1000"])
    ap.add_argument("--synth-dir", default="outputs/synth/preview3")
    ap.add_argument("--synth-images", type=int, default=4)
    ap.add_argument("--synth-text", type=int, default=3)
    ap.add_argument("--synth-box", type=int, default=2)
    ap.add_argument("--ccb-program", default="artifacts/visual_primitives/ccb_v139_stroke_width_sweep_harness_v81/ccb_pure_native_current_best.vp_program.json")
    ap.add_argument("--ccb-image", default="uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg")
    ap.add_argument("--ccb-text", type=int, default=12)
    ap.add_argument("--ccb-box", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap total items (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="outputs/grounding_bench")
    ap.add_argument("--manifest", default="", help="reuse a saved eval_manifest.json instead of rebuilding")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if args.manifest and Path(args.manifest).exists():
        items = json.loads(Path(args.manifest).read_text())
    else:
        items = build_manifest(args, rng)
        (outdir / "eval_manifest.json").write_text(json.dumps(items, indent=1))
    if args.limit:
        items = items[: args.limit]
    api_key = "" if args.api_key_env in ("", "NONE") else os.environ.get(args.api_key_env, "")
    print(f"[bench] backend={args.backend} model={args.model} items={len(items)} "
          f"(synthetic={sum(1 for i in items if i['source']=='synthetic')}, ccb={sum(1 for i in items if i['source']=='ccb')})", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(lambda it: run_item(it, args, api_key), items):
            results.append(r)
            tag = f"{r['source']}/{r['kind']}"
            print(f"  IoU={r['iou']:.3f} {tag:16s} '{r['label'][:30]}'" + (f"  ERR {r['err']}" if r['err'] else ""), flush=True)

    def agg(sel):
        sub = [r for r in results if sel(r)]
        if not sub:
            return (0, 0.0, 0.0)
        miou = sum(r["iou"] for r in sub) / len(sub)
        hit = sum(1 for r in sub if r["iou"] >= 0.5) / len(sub)
        return (len(sub), miou, hit)

    report = {"backend": args.backend, "model": args.model, "n": len(results),
              "overall": agg(lambda r: True),
              "synthetic": agg(lambda r: r["source"] == "synthetic"),
              "ccb": agg(lambda r: r["source"] == "ccb"),
              "text": agg(lambda r: r["kind"] == "text"),
              "box": agg(lambda r: r["kind"] == "box")}
    (outdir / f"results_{args.backend}_{args.model.replace('/', '_')}.json").write_text(
        json.dumps({"report": report, "results": results}, indent=2))
    print("\n=== Grounding IoU report ===")
    for k in ("overall", "synthetic", "ccb", "text", "box"):
        n, miou, hit = report[k]
        print(f"  {k:10s} n={n:3d}  meanIoU={miou:.3f}  IoU>=0.5={hit*100:4.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
