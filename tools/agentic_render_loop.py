#!/usr/bin/env python3
"""Stage 0: agentic render-and-compare loop (no training).

Validates whether putting a frontier VLM inside a generate->render->diff->revise
loop can break the ~0.88 ceiling that greedy per-primitive tweaking hit.

Strategy: start from the current best vp_program, render it, show a frontier VLM
(Gemini 2.5 Pro) the TARGET image and our RECONSTRUCTION, ask what visual content
is present in the target but MISSING/WRONG in the reconstruction, add those
primitives, re-render, and keep them only if the harness score improves
(accept-if-better). This directly attacks the diagnosed gap that greedy cannot
touch: recall (missing edges ~17% of pixels, ~30 missing text tokens).

Every helper returns (value, error): (result, None) on success, (None, 'msg') on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import evaluate_drawio_variants  # noqa: E402

GEMINI_KEY = os.environ.get("GOOGLE_API_KEY", "")  # set GOOGLE_API_KEY to use the gemini proposer
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
# CRITICAL: harness default font is Helvetica (config font_family or "Helvetica").
# Compiling with Arial degrades OCR by ~0.05 and makes scores non-comparable.
FONT_FAMILY = os.environ.get("AGENT_FONT_FAMILY", "Helvetica")


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def load_expected_tokens(path: str | Path) -> tuple[list[str] | None, str | None]:
    try:
        data = json.loads(Path(path).read_text())
    except Exception as exc:  # noqa: BLE001
        return None, f"failed to read expected tokens: {exc}"
    if isinstance(data, list):
        return [str(t) for t in data], None
    if isinstance(data, dict):
        toks = [str(t) for t in (data.get("tokens") or [])]
        phrases = [str(t) for t in (data.get("phrases") or [])]
        return toks + phrases, None
    return [], None


def refresh_counts(program: dict[str, Any]) -> None:
    prims = program.get("primitives") or []
    program["counts"] = {
        "regions": sum(1 for p in prims if p.get("type") == "region"),
        "texts": sum(1 for p in prims if p.get("type") == "text"),
        "edges": sum(1 for p in prims if p.get("type") == "edge"),
        "shapes": sum(1 for p in prims if p.get("type") == "shape"),
        "total": len(prims),
    }


def score_program(
    program: dict[str, Any],
    source_image: str,
    expected_tokens: list[str] | None,
    workdir: Path,
    tag: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Emit program -> render -> score. Returns (row, None) or (None, err)."""
    drawio = workdir / f"{tag}.drawio"
    try:
        compile_program_to_drawio(program, drawio, font_family=FONT_FAMILY)
    except Exception as exc:  # noqa: BLE001
        return None, f"compile failed: {exc}"
    try:
        rows = evaluate_drawio_variants(
            source_image, [drawio], export=True, expected_tokens=expected_tokens or None
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"evaluate failed: {exc}"
    if not rows:
        return None, "evaluate returned no rows"
    return rows[0], None


def summarize(row: dict[str, Any]) -> dict[str, Any]:
    m = row.get("metrics") or {}
    edge = m.get("edge") or {}
    ocr = m.get("ocr") or {}
    sem = ocr.get("semantic") or {}
    tgt = ocr.get("target_semantic") or {}
    ocr_f1 = max(float(ocr.get("f1") or 0), float(sem.get("f1") or 0), float(tgt.get("f1") or 0))
    pc = m.get("program_cleanliness") or {}
    pure = row.get("native_purity") or {}
    return {
        "score": row.get("score"),
        "edge_f1": round(float(edge.get("f1") or 0), 4),
        "edge_recall": round(float(edge.get("recall") or 0), 4),
        "edge_precision": round(float(edge.get("precision") or 0), 4),
        "reference_edges": edge.get("reference_edges"),
        "rendered_edges": edge.get("rendered_edges"),
        "ocr_f1": round(ocr_f1, 4),
        "ocr_recall": round(float(ocr.get("recall") or 0), 4),
        "missing_sample": (ocr.get("missing_sample") or [])[:30],
        "residual_text_overlap": pc.get("residual_text_overlap_count"),
        "native_purity_ok": bool(pure.get("ok", True)),
    }


def acceptable(row: dict[str, Any]) -> bool:
    """Hard constraints mirror the harness acceptance gate."""
    pure = row.get("native_purity") or {}
    if not pure.get("ok", True):
        return False
    m = row.get("metrics") or {}
    pc = m.get("program_cleanliness") or {}
    if int(pc.get("residual_text_overlap_count") or 0) > 0:
        return False
    return True


# --------------------------------------------------------------------------- #
# VLM proposal
# --------------------------------------------------------------------------- #
PROMPT = """You are repairing a draw.io reconstruction of a technical figure.

IMAGE 1 = the TARGET figure (ground truth).
IMAGE 2 = our current RECONSTRUCTION, rendered from a diagram program.
They should look identical but the reconstruction is MISSING some content.

Both images are {W}x{H} pixels. Coordinates: origin top-left, x grows right, y grows down.

Your job: list visual content that is clearly present in the TARGET (image 1) but
MISSING or clearly WRONG in the RECONSTRUCTION (image 2). Focus on, in priority order:
1. Missing connector lines / arrows between boxes.
2. Missing text labels (give the EXACT text string).
3. Missing boxes / shapes.

STRICT RULES:
- Only propose content that is genuinely absent or wrong in the RECONSTRUCTION. Do NOT
  duplicate content already correctly shown in image 2.
- Do NOT place a text label on top of text that already exists in the reconstruction.
- Give coordinates in TARGET pixel space ({W}x{H}).
- Be precise about positions; approximate is acceptable but try to be within ~10px.
- Prefer the most confident, clearly-missing items. At most {maxn} items.

Known weak spots from automatic scoring (use as hints, not gospel):
- Edge recall is low (many thin connector lines missing).
- Missing OCR text tokens detected: {missing}

Return ONLY JSON of this exact form:
{{"additions": [
  {{"kind":"edge","path":[[x1,y1],[x2,y2]],"stroke":"#050505","stroke_width":1.2,"arrow_end":false,"arrow_start":false,"why":"..."}},
  {{"kind":"text","text":"EXACT","bbox":[x0,y0,x1,y1],"font_size":11,"bold":false,"align":"center","font_color":"#050505","why":"..."}},
  {{"kind":"shape","shape":"rectangle","bbox":[x0,y0,x1,y1],"fill":"none","stroke":"#6f8190","why":"..."}}
]}}
No prose, no markdown fences."""


CLAUDE_OAUTH = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")


def _parse_additions(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    # tolerate leading prose: grab the first {...} block
    if not text.startswith("{") and "{" in text:
        text = text[text.find("{"):]
        if "}" in text:
            text = text[: text.rfind("}") + 1]
    try:
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return None, f"json parse failed: {exc}; raw head: {text[:200]}"
    adds = data.get("additions") if isinstance(data, dict) else None
    if not isinstance(adds, list):
        return None, "no additions list in response"
    return adds, None


def _call_claude_vision(prompt: str, image_paths: list[str]) -> tuple[str | None, str | None]:
    import base64
    import subprocess
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        media = "image/png" if str(p).lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
    msg = {"type": "user", "message": {"role": "user", "content": content}}
    env = os.environ.copy()
    if CLAUDE_OAUTH:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = CLAUDE_OAUTH
    env.pop("ANTHROPIC_API_KEY", None)
    try:
        proc = subprocess.run(
            ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--model", CLAUDE_MODEL,
             "--disallowedTools", "Bash", "Edit", "Write", "Read", "Skill", "Agent", "WebSearch", "WebFetch"],
            input=json.dumps(msg) + "\n", text=True, capture_output=True, env=env, timeout=240,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"claude call failed: {exc}"
    result_text = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if obj.get("type") == "result":
            result_text = obj.get("result")
        elif obj.get("type") == "assistant":
            for b in (obj.get("message") or {}).get("content", []):
                if b.get("type") == "text" and b.get("text", "").strip():
                    result_text = b["text"]
    if not result_text:
        return None, f"no text in claude output (rc={proc.returncode})"
    return result_text, None


def propose_additions(
    proposer: str,
    target_path: str,
    render_path: str,
    missing_tokens: list[str],
    width: int,
    height: int,
    max_items: int,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    prompt = PROMPT.format(
        W=width, H=height, maxn=max_items,
        missing=", ".join(missing_tokens[:30]) or "(none)",
    )
    if proposer == "claude":
        text, err = _call_claude_vision(prompt, [target_path, render_path])
        if err:
            return None, err
        return _parse_additions(text)
    # gemini
    try:
        from google import genai
        from google.genai import types
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        return None, f"import failed: {exc}"
    try:
        client = genai.Client(api_key=GEMINI_KEY)
        target = Image.open(target_path).convert("RGB")
        render = Image.open(render_path).convert("RGB")
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, target, render],
            config=types.GenerateContentConfig(
                max_output_tokens=8192, temperature=0.3, response_mime_type="application/json",
            ),
        )
        text = (resp.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return None, f"gemini call failed: {exc}"
    return _parse_additions(text)


# --------------------------------------------------------------------------- #
# Candidate -> primitive
# --------------------------------------------------------------------------- #
def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def shift_primitive(prim: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Return a copy of prim translated by (dx, dy)."""
    p = json.loads(json.dumps(prim))
    bb = p.get("bbox")
    if bb and len(bb) >= 4:
        p["bbox"] = [bb[0] + dx, bb[1] + dy, bb[2] + dx, bb[3] + dy]
    if p.get("path"):
        p["path"] = [[pt[0] + dx, pt[1] + dy] for pt in p["path"] if len(pt) >= 2]
    return p


SNAP_GRID = [(dx, dy)
             for dx in (-24, -12, 0, 12, 24)
             for dy in (-24, -12, 0, 12, 24)
             if not (dx == 0 and dy == 0)]


def candidate_to_primitive(
    cand: dict[str, Any], idx: int, width: int, height: int
) -> dict[str, Any] | None:
    kind = str(cand.get("kind") or "").lower()
    try:
        if kind == "edge":
            path = cand.get("path") or []
            pts = [[_clip(p[0], 0, width), _clip(p[1], 0, height)] for p in path if len(p) >= 2]
            if len(pts) < 2:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return {
                "id": f"agent_edge_{idx:04d}",
                "type": "edge",
                "role": "connector",
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "path": pts,
                "style": {
                    "stroke": str(cand.get("stroke") or "#050505"),
                    "stroke_width": float(cand.get("stroke_width") or 1.2),
                    "arrow_start": bool(cand.get("arrow_start")),
                    "arrow_end": bool(cand.get("arrow_end")),
                },
                "source": "agent_render_loop",
            }
        if kind == "text":
            text = str(cand.get("text") or "").strip()
            bbox = cand.get("bbox") or []
            if not text or len(bbox) < 4:
                return None
            x0, y0, x1, y1 = (_clip(bbox[0], 0, width), _clip(bbox[1], 0, height),
                              _clip(bbox[2], 0, width), _clip(bbox[3], 0, height))
            return {
                "id": f"agent_text_{idx:04d}",
                "type": "text",
                "role": "label",
                "bbox": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                "text": text,
                "style": {
                    "font_size": int(cand.get("font_size") or 11),
                    "bold": bool(cand.get("bold")),
                    "align": str(cand.get("align") or "center"),
                    "font_color": str(cand.get("font_color") or "#050505"),
                },
                "source": "agent_render_loop",
            }
        if kind == "shape":
            bbox = cand.get("bbox") or []
            if len(bbox) < 4:
                return None
            x0, y0, x1, y1 = (_clip(bbox[0], 0, width), _clip(bbox[1], 0, height),
                              _clip(bbox[2], 0, width), _clip(bbox[3], 0, height))
            return {
                "id": f"agent_shape_{idx:04d}",
                "type": "shape",
                "role": "symbol_or_mark",
                "shape": str(cand.get("shape") or "rectangle"),
                "bbox": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                "style": {
                    "fill": str(cand.get("fill") or "none"),
                    "stroke": str(cand.get("stroke") or "#6f8190"),
                },
                "source": "agent_render_loop",
            }
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg")
    ap.add_argument("--program", default="artifacts/visual_primitives/ccb_v139_stroke_width_sweep_harness_v81/ccb_pure_native_current_best.vp_program.json")
    ap.add_argument("--expected-tokens", default="configs/visual_primitives/ccb_expected_text_prior_v1.json")
    ap.add_argument("--outdir", default="outputs/agentic_loop/stage0_run1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-items", type=int, default=14)
    ap.add_argument("--proposer", default="claude", choices=["claude", "gemini"])
    ap.add_argument("--snap", action="store_true", help="local position search to refine VLM-placed primitives")
    ap.add_argument("--snap-max", type=int, default=6, help="max near-miss candidates to snap per round")
    args = ap.parse_args()

    workdir = Path(args.outdir)
    workdir.mkdir(parents=True, exist_ok=True)

    expected, err = load_expected_tokens(args.expected_tokens)
    if err:
        print(f"[warn] {err}; proceeding without expected tokens", flush=True)
        expected = None

    program = load_program(args.program)
    width = int((program.get("canvas") or {}).get("width") or 1376)
    height = int((program.get("canvas") or {}).get("height") or 768)

    base_row, err = score_program(program, args.source, expected, workdir, "round0_baseline")
    if err:
        print(f"[fatal] baseline scoring failed: {err}", flush=True)
        return 1
    best_score = float(base_row.get("score") or -1e9)
    print(f"[baseline] score={best_score:.6f}", flush=True)
    print(f"           {json.dumps(summarize(base_row))}", flush=True)

    trajectory = [{"round": 0, "summary": summarize(base_row), "accepted_adds": 0}]
    add_counter = 0

    for rnd in range(1, args.rounds + 1):
        render_png = str(workdir / f"round{rnd-1}_baseline.drawio.png") if rnd == 1 \
            else str(workdir / f"round{rnd-1}_best.drawio.png")
        # ensure we point at the most recent render of current best
        cur_render = workdir / (f"round{rnd-1}_baseline.drawio.png" if rnd == 1 else f"round{rnd-1}_best.drawio.png")
        if not cur_render.exists():
            # fall back to baseline render
            cur_render = workdir / "round0_baseline.drawio.png"
        missing = summarize(base_row if rnd == 1 else best_row)["missing_sample"]

        proposer_name = CLAUDE_MODEL if args.proposer == "claude" else GEMINI_MODEL
        print(f"\n[round {rnd}] proposing additions via {args.proposer} ({proposer_name}) ...", flush=True)
        cands, err = propose_additions(
            args.proposer, args.source, str(cur_render), missing, width, height, args.max_items
        )
        if err:
            print(f"[round {rnd}] proposal failed: {err}", flush=True)
            trajectory.append({"round": rnd, "error": err})
            continue
        print(f"[round {rnd}] got {len(cands)} candidate additions", flush=True)

        accepted = 0
        near_misses = []  # (delta, kind, why) for candidates that scored just below best
        for cand in cands:
            add_counter += 1
            prim = candidate_to_primitive(cand, add_counter, width, height)
            if prim is None:
                continue
            existing_ids = {str(p.get("id")) for p in program.get("primitives", [])}
            if prim["id"] in existing_ids:
                continue
            trial = json.loads(json.dumps(program))
            trial.setdefault("primitives", []).append(prim)
            refresh_counts(trial)
            row, serr = score_program(trial, args.source, expected, workdir, f"round{rnd}_try{add_counter}")
            if serr:
                print(f"   [try {add_counter}] {prim['type']} score error: {serr}", flush=True)
                continue
            sc = float(row.get("score") or -1e9)
            delta = sc - best_score
            why = str(cand.get("why") or "")[:55]
            if sc > best_score and acceptable(row):
                program = trial
                best_score = sc
                best_row = row
                accepted += 1
                print(f"   [ACCEPT {prim['type']}] +{delta:.6f} -> {sc:.6f} | {why}", flush=True)
                continue
            if delta > -0.002:  # near miss: likely a real item placed imprecisely
                near_misses.append((round(delta, 6), prim, why))

        # Snap pass: local position search on the most promising near-misses.
        if args.snap and near_misses:
            near_misses.sort(key=lambda t: t[0], reverse=True)
            print(f"   near-misses: {[(d, p['type'], w) for d, p, w in near_misses[:8]]}", flush=True)
            for raw_delta, prim, why in near_misses[: args.snap_max]:
                best_local = (best_score, None, None)
                for dx, dy in SNAP_GRID:
                    sp = shift_primitive(prim, dx, dy)
                    trial = json.loads(json.dumps(program))
                    trial.setdefault("primitives", []).append(sp)
                    refresh_counts(trial)
                    add_counter += 1
                    row, serr = score_program(trial, args.source, expected, workdir, f"round{rnd}_snap{add_counter}")
                    if serr:
                        continue
                    sc = float(row.get("score") or -1e9)
                    if sc > best_local[0] and acceptable(row):
                        best_local = (sc, trial, row)
                if best_local[1] is not None:
                    program = best_local[1]
                    best_row = best_local[2]
                    delta = best_local[0] - best_score
                    best_score = best_local[0]
                    accepted += 1
                    print(f"   [SNAP-ACCEPT {prim['type']}] raw={raw_delta:+.6f} -> +{delta:.6f} -> {best_score:.6f} | {why}", flush=True)
        elif near_misses:
            near_misses.sort(key=lambda t: t[0], reverse=True)
            print(f"   near-misses: {[(d, p['type'], w) for d, p, w in near_misses[:8]]}", flush=True)

        # render current best for next round's diff
        best_row, err = score_program(program, args.source, expected, workdir, f"round{rnd}_best")
        if err is None:
            best_score = float(best_row.get("score") or best_score)
        save_program(program, workdir / f"round{rnd}_best.vp_program.json")
        s = summarize(best_row)
        print(f"[round {rnd}] accepted {accepted}; score={best_score:.6f} edge_f1={s['edge_f1']} ocr_f1={s['ocr_f1']}", flush=True)
        trajectory.append({"round": rnd, "summary": s, "accepted_adds": accepted})
        (workdir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    save_program(program, workdir / "final_best.vp_program.json")
    print(f"\n[done] baseline={trajectory[0]['summary']['score']:.6f} -> final={best_score:.6f} "
          f"(delta {best_score - float(trajectory[0]['summary']['score']):+.6f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
