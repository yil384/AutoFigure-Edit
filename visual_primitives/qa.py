"""Render-time QA for pure-native draw.io visual primitive programs."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops


DEFAULT_DRAWIO_CLI = "/Applications/draw.io.app/Contents/MacOS/draw.io"
FORBIDDEN_DRAWIO_TOKENS = (
    "shape=image",
    "data:image",
    "<image",
    "base64",
    "shape=stencil",
    "stencil(",
)


def validate_pure_native_drawio(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(errors="ignore")
    hits = [token for token in FORBIDDEN_DRAWIO_TOKENS if token in text]
    return {
        "ok": not hits,
        "forbidden_hits": hits,
        "forbidden_tokens": list(FORBIDDEN_DRAWIO_TOKENS),
    }


def export_drawio_png(drawio_path: str | Path,
                      png_path: str | Path,
                      drawio_cli: str = DEFAULT_DRAWIO_CLI,
                      scale: float = 1.0,
                      attempts: int = 3) -> dict[str, Any]:
    cmd = [
        drawio_cli, "-x", "-f", "png", "-e", "-b", "0",
        "-o", str(png_path),
    ]
    if scale != 1.0:
        cmd.extend(["-s", str(scale)])
    cmd.append(str(drawio_path))
    env = os.environ.copy()
    env.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    attempt_reports = []
    for attempt in range(1, max(1, attempts) + 1):
        if attempt > 1:
            time.sleep(float(attempt))
        result = subprocess.run(cmd, text=True, capture_output=True, env=env)
        ok = result.returncode == 0 and Path(png_path).exists()
        attempt_reports.append({
            "attempt": attempt,
            "ok": ok,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        })
        if ok:
            return {
                "ok": True,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "command": cmd,
                "attempts": attempt_reports,
            }
        if result.returncode in {-6, -9, 134, 137}:
            _cleanup_drawio_cli_process(drawio_cli)
    last = attempt_reports[-1]
    return {
        "ok": False,
        "returncode": last["returncode"],
        "stdout": last["stdout"],
        "stderr": last["stderr"],
        "command": cmd,
        "attempts": attempt_reports,
    }


def _cleanup_drawio_cli_process(drawio_cli: str) -> None:
    """Best-effort cleanup for draw.io Desktop crashes between batch exports."""
    patterns = []
    name = Path(drawio_cli).name
    if name:
        patterns.append(name)
    if "draw.io" not in patterns:
        patterns.append("draw.io")
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                text=True,
                capture_output=True,
                timeout=2,
            )
        except Exception:
            continue


def make_side_by_side(reference_image: str | Path,
                      rendered_image: str | Path,
                      output_path: str | Path) -> dict[str, Any]:
    ref = Image.open(reference_image).convert("RGB")
    ren = Image.open(rendered_image).convert("RGB")
    h = max(ref.height, ren.height)
    canvas = Image.new("RGB", (ref.width + ren.width, h), "white")
    canvas.paste(ref, (0, 0))
    canvas.paste(ren, (ref.width, 0))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return {
        "ok": True,
        "path": str(out),
        "width": canvas.width,
        "height": canvas.height,
    }


def make_quadrant_crops(image_path: str | Path,
                        output_stem: str | Path) -> dict[str, str]:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    boxes = {
        "top_left": (0, 0, w // 2, h // 2),
        "top_right": (w // 2, 0, w, h // 2),
        "bottom_left": (0, h // 2, w // 2, h),
        "bottom_right": (w // 2, h // 2, w, h),
    }
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, box in boxes.items():
        out = Path(f"{stem}.{name}.png")
        img.crop(box).save(out)
        paths[name] = str(out)
    return paths


def compute_render_metrics(reference_image: str | Path,
                           rendered_image: str | Path) -> dict[str, Any]:
    ref = Image.open(reference_image).convert("RGB")
    ren = Image.open(rendered_image).convert("RGB")
    w = min(ref.width, ren.width)
    h = min(ref.height, ren.height)
    ref = ref.crop((0, 0, w, h))
    ren = ren.crop((0, 0, w, h))
    ref_arr = np.asarray(ref).astype(np.float32)
    ren_arr = np.asarray(ren).astype(np.float32)
    mae = float(np.mean(np.abs(ref_arr - ren_arr)))
    diff = ImageChops.difference(ref, ren).convert("L")
    changed_ratio = float((np.asarray(diff) > 30).mean())
    edge = _edge_metrics(ref_arr.astype(np.uint8), ren_arr.astype(np.uint8))
    ocr = _ocr_token_metrics(ref, ren)
    return {
        "canvas": {"width": w, "height": h},
        "pixel_mae": round(mae, 4),
        "changed_pixel_ratio_t30": round(changed_ratio, 6),
        "edge": edge,
        "ocr": ocr,
    }


def write_qa_report(report_path: str | Path,
                    *,
                    image_path: str | Path,
                    drawio_path: str | Path,
                    rendered_png: str | Path | None,
                    compare_png: str | Path | None,
                    ledger_path: str | Path,
                    program_path: str | Path,
                    pure_check: dict[str, Any],
                    export_result: dict[str, Any] | None,
                    metrics: dict[str, Any] | None,
                    program: dict[str, Any]) -> None:
    lines = [
        "# Visual Primitive QA Report",
        "",
        "## Artifacts",
        f"- source_image: `{image_path}`",
        f"- drawio: `{drawio_path}`",
        f"- rendered_png: `{rendered_png}`",
        f"- compare_png: `{compare_png}`",
        f"- primitive_ledger: `{ledger_path}`",
        f"- visual_primitive_program: `{program_path}`",
        "",
        "## Native Purity",
        f"- ok: `{pure_check.get('ok')}`",
        f"- forbidden_hits: `{pure_check.get('forbidden_hits')}`",
        "",
        "## Counts",
        "```json",
        json.dumps(program.get("counts", {}), indent=2),
        "```",
        "",
        "## Metrics",
        "```json",
        json.dumps(metrics or {}, indent=2),
        "```",
        "",
        "## Heuristic Issues",
    ]
    for issue in _heuristic_issues(pure_check, export_result, metrics, program):
        lines.append(f"- {issue}")
    lines.extend([
        "",
        "## VLM Refinement Prompt",
        "Use this prompt with the side-by-side image and the JSON program.",
        "",
        "```text",
        build_vlm_refinement_prompt(
            image_path=image_path,
            compare_png=compare_png,
            program_path=program_path,
            metrics=metrics,
            program=program,
        ),
        "```",
        "",
    ])
    Path(report_path).write_text("\n".join(lines))


def build_vlm_refinement_prompt(*,
                                image_path: str | Path,
                                compare_png: str | Path | None,
                                program_path: str | Path,
                                metrics: dict[str, Any] | None,
                                program: dict[str, Any]) -> str:
    counts = program.get("counts", {})
    counts_text = json.dumps(counts, indent=2)
    metric_text = json.dumps(metrics or {}, indent=2)
    return f"""You are refining a pure-native draw.io reconstruction.

Reference image: {image_path}
Side-by-side image: {compare_png}
Visual primitive program JSON: {program_path}

Hard constraints:
- Do not use a source-image overlay.
- Do not use raster crops, data:image, base64, image cells, or stencils.
- All edits must be native draw.io primitives: text, rect, ellipse, rhombus,
  triangle, line, polyline, edge, group, style, z-order.
- Think with visual primitives. Every critique must point to a primitive id or
  a concrete coordinate box/path.

Current primitive counts:
{counts_text}
Current render metrics:
{metric_text}

Return JSON only:
{{
  "visual_diagnosis": [
    {{
      "issue": "...",
      "region": [x0, y0, x1, y1],
      "primitive_ids": ["..."],
      "severity": "high|medium|low"
    }}
  ],
  "patch_plan": [
    {{
      "op": "add|delete|update|split|merge|restyle|reroute",
      "target_ids": ["..."],
      "new_primitive": {{}},
      "reason": "..."
    }}
  ],
  "stop_or_continue": "continue"
}}
"""


def _edge_metrics(ref_rgb: np.ndarray, ren_rgb: np.ndarray) -> dict[str, float]:
    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    ren_gray = cv2.cvtColor(ren_rgb, cv2.COLOR_RGB2GRAY)
    ref_edges = cv2.Canny(ref_gray, 55, 150) > 0
    ren_edges = cv2.Canny(ren_gray, 55, 150) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ref_d = cv2.dilate(ref_edges.astype(np.uint8), kernel) > 0
    ren_d = cv2.dilate(ren_edges.astype(np.uint8), kernel) > 0
    tp_ref = int((ren_edges & ref_d).sum())
    tp_ren = int((ref_edges & ren_d).sum())
    pred = int(ren_edges.sum())
    gt = int(ref_edges.sum())
    precision = tp_ref / max(1, pred)
    recall = tp_ren / max(1, gt)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "reference_edges": gt,
        "rendered_edges": pred,
    }


def _ocr_token_metrics(ref: Image.Image, ren: Image.Image) -> dict[str, Any]:
    try:
        import pytesseract
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    ref_tokens = _ocr_tokens(ref, pytesseract)
    ren_tokens = _ocr_tokens(ren, pytesseract)
    raw = _token_set_metrics(ref_tokens, ren_tokens)
    semantic = _token_set_metrics(
        [_semantic_ocr_token(token) for token in ref_tokens],
        [_semantic_ocr_token(token) for token in ren_tokens],
    )
    return {
        "available": True,
        "reference_token_count": raw["reference_token_count"],
        "rendered_token_count": raw["rendered_token_count"],
        "overlap_count": raw["overlap_count"],
        "precision": raw["precision"],
        "recall": raw["recall"],
        "f1": raw["f1"],
        "missing_sample": raw["missing_sample"],
        "extra_sample": raw["extra_sample"],
        "semantic": semantic,
    }


def _ocr_tokens(img: Image.Image, pytesseract_module) -> list[str]:
    data = pytesseract_module.image_to_data(
        img, output_type=pytesseract_module.Output.DICT, config="--psm 11")
    tokens = []
    for raw, conf in zip(data.get("text", []), data.get("conf", [])):
        text = _normalize_token(raw)
        if not text:
            continue
        try:
            c = float(conf)
        except Exception:
            c = -1.0
        if c < 45:
            continue
        tokens.append(text)
    return tokens


def _token_set_metrics(ref_tokens: list[str], ren_tokens: list[str]) -> dict[str, Any]:
    ref_set = {token for token in ref_tokens if token}
    ren_set = {token for token in ren_tokens if token}
    overlap = ref_set & ren_set
    precision = len(overlap) / max(1, len(ren_set))
    recall = len(overlap) / max(1, len(ref_set))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "reference_token_count": len(ref_set),
        "rendered_token_count": len(ren_set),
        "overlap_count": len(overlap),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "missing_sample": sorted(ref_set - ren_set)[:30],
        "extra_sample": sorted(ren_set - ref_set)[:30],
    }


def _normalize_token(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9+\-/]+", "", text)
    if len(text) <= 1 and not text.isdigit():
        return ""
    return text


def _semantic_ocr_token(token: str) -> str:
    corrections = {
        "al-driven": "ai-driven",
        "al-enabled": "ai-enabled",
        "al-enhanced": "ai-enhanced",
        "drand": "brand",
        "metrocs": "metrics",
        "hrrc": "hmc",
        "layout)": "layout",
        "bencnmarking": "benchmarking",
        "benenmarking": "benchmarking",
        "dittusion": "diffusion",
        "fiter": "filter",
    }
    return corrections.get(token, token)


def _heuristic_issues(pure_check, export_result, metrics, program) -> list[str]:
    issues = []
    if not pure_check.get("ok"):
        issues.append("native purity failed; forbidden raster/stencil tokens are present")
    if export_result and not export_result.get("ok"):
        issues.append("draw.io export failed; inspect stderr before trusting visual metrics")
    if metrics:
        edge = metrics.get("edge", {})
        edge_f1 = edge.get("f1", 0.0)
        edge_recall = edge.get("recall", 0.0)
        if edge_f1 < 0.35:
            issues.append("edge F1 is low; connector/topology reconstruction is weak")
        elif edge_recall < 0.65:
            issues.append("edge recall is low; many source strokes/connectors are missing")
        ocr_f1 = metrics.get("ocr", {}).get("f1")
        ocr_recall = metrics.get("ocr", {}).get("recall")
        if isinstance(ocr_f1, float) and ocr_f1 < 0.45:
            issues.append("OCR overlap is low; text content/placement needs VLM attention")
        elif isinstance(ocr_recall, float) and ocr_recall < 0.65:
            issues.append("OCR recall is low; many source text tokens are missing")
    duplicate_regions = _count_high_overlap_primitives(program, "region", threshold=0.82)
    if duplicate_regions:
        issues.append(
            f"{duplicate_regions} region pairs overlap heavily; panel deduplication needs refinement")
    source_counts = program.get("source_ledger_counts", {})
    if source_counts.get("contour_paths", 0):
        issues.append("experimental contour paths are present; visually inspect for dirty outlines")
    if program.get("counts", {}).get("shapes", 0) > 2 * max(1, program.get("counts", {}).get("regions", 0)):
        issues.append("many symbol/mark shapes relative to regions; check for speckle noise")
    if not issues:
        issues.append("no automatic blocker found; use VLM visual critique for qualitative errors")
    return issues


def _count_high_overlap_primitives(program: dict[str, Any],
                                   primitive_type: str,
                                   threshold: float) -> int:
    primitives = [
        p for p in program.get("primitives", [])
        if p.get("type") == primitive_type and "bbox" in p
    ]
    count = 0
    for i, a in enumerate(primitives):
        for b in primitives[i + 1:]:
            if _bbox_overlap_fraction(a["bbox"], b["bbox"]) >= threshold:
                count += 1
    return count


def _bbox_overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / area
