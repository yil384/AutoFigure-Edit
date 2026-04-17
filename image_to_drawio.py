"""
PNG -> Editable draw.io XML Converter

Transforms an input PNG figure (or multi-figure grid) into a natively editable
.drawio file. Each visual element (boxes, arrows, text, icons) becomes an
independent mxCell in the output.

Pipeline:
  1. split_image_grid()           - split multi-figure grids into panels
  2. segment_with_sam3()          - detect icons via Roboflow SAM3 API  [from autofigure2]
  3. regenerate_icons_with_gemini - LLM regenerates clean icon PNGs
     extract_icons_from_sheet     - split icon sheet into individuals
  4. generate_drawio_template()   - two-image multimodal prompt -> draw.io XML
  5. optimize_drawio_with_llm()   - SSIM-guided iterative refinement
  6. replace_icons_in_drawio()    - insert base64 PNGs into mxCell placeholders
  7. assemble multi-page .drawio  - combine pages into single file

Usage (CLI):
    python image_to_drawio.py \\
        --input_image figure.png \\
        --output_dir outputs/test \\
        --provider gemini \\
        --api_key <key>
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import requests

from icon_extractor import extract_icons, auto_detect_boxes, IconInfo, BBox

# ---------------------------------------------------------------------------
# Real-ESRGAN 4x super-resolution (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_esrgan_model = None  # lazy singleton


def _get_esrgan_model():
    """Get or create Real-ESRGAN model singleton. Returns (model, device, None) or (None, None, error)."""
    global _esrgan_model
    if _esrgan_model is not None:
        return (*_esrgan_model, None)
    try:
        import torch
        import spandrel

        model_dir = os.environ.get("ESRGAN_MODEL_DIR", os.path.expanduser("~/.cache/realesrgan"))
        model_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")
        if not os.path.exists(model_path):
            return (None, None, f"Model not found: {model_path}")

        print(f"  Loading Real-ESRGAN model...")
        model = spandrel.ModelLoader().load_from_file(model_path).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _esrgan_model = (model, device)
        print(f"  Real-ESRGAN loaded (device={device})")
        return (model, device, None)
    except Exception as e:
        return (None, None, f"Real-ESRGAN unavailable: {e}")


def upscale_icon_esrgan(image_path: str, output_path: str) -> str:
    """Upscale an icon image 4x using Real-ESRGAN. Returns output path."""
    import torch

    model, device, err = _get_esrgan_model()
    if err:
        raise RuntimeError(err)

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)

    out_arr = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_arr = (out_arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(out_arr).save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# RMBG-2.0 background removal (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_rmbg_remover = None  # lazy singleton


def _get_rmbg_remover(output_dir: str = "./output/icons"):
    """Get or create RMBG-2.0 remover singleton. Returns (remover, None) or (None, error)."""
    global _rmbg_remover
    if _rmbg_remover is not None:
        _rmbg_remover.output_dir = Path(output_dir)
        _rmbg_remover.output_dir.mkdir(parents=True, exist_ok=True)
        return (_rmbg_remover, None)
    try:
        import torch
        from torchvision import transforms as tv_transforms
        from transformers import AutoModelForImageSegmentation

        # Load HF token from env or .env file
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            env_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_path):
                for line in open(env_path):
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.strip().split("=", 1)[1]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading RMBG-2.0 model (device={device})...")

        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0",
            trust_remote_code=True,
            token=hf_token,
        ).eval().to(device)

        transform = tv_transforms.Compose([
            tv_transforms.Resize((1024, 1024)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        class _RMBGRemover:
            def __init__(self):
                self.model = model
                self.device = device
                self.transform = transform
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)

            def remove_background(self, image_path: str, output_path: str) -> str:
                """Remove background from image, save RGBA PNG to output_path."""
                image = Image.open(image_path).convert("RGB")
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds = self.model(input_tensor)[-1].sigmoid().cpu()
                pred = preds[0].squeeze()
                mask = tv_transforms.ToPILImage()(pred).resize(image.size)
                out = image.copy()
                out.putalpha(mask)
                out.save(output_path)
                return output_path

        _rmbg_remover = _RMBGRemover()
        print(f"  RMBG-2.0 loaded successfully")
        return (_rmbg_remover, None)

    except Exception as e:
        return (None, f"RMBG-2.0 unavailable: {e}")


# ---------------------------------------------------------------------------
# Provider config (standalone, no autofigure2 dependency)
# ---------------------------------------------------------------------------
ProviderType = Literal["openrouter", "bianxie", "gemini"]

PROVIDER_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    },
    "bianxie": {
        "base_url": "https://api.bianxie.ai/v1",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
}

SAM3_ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL",
    "https://serverless.roboflow.com/sam3/concept_segment",
)
SAM3_API_TIMEOUT = 300


# ---------------------------------------------------------------------------
# Standalone Gemini API helpers
# ---------------------------------------------------------------------------

def _get_gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _build_gemini_text_config(max_tokens: int, temperature: float):
    from google.genai import types
    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


def _gemini_call_with_retry(client, model: str, contents, config, max_retries: int = 3):
    """Call Gemini generate_content with exponential backoff retry.

    Retries on 429 (rate limit), 503 (unavailable), timeouts, and
    connection errors. Non-retryable errors are raised immediately.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            retryable = any(k in err_str for k in [
                "429", "503", "resource_exhausted", "unavailable",
                "timeout", "timed out", "deadline", "connection",
                "reset by peer", "broken pipe", "server error", "500",
            ])
            if retryable and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                print(f"    Retryable error (attempt {attempt+1}/{max_retries}), waiting {wait}s: {e}")
                time.sleep(wait)
                continue
            raise
    raise last_err


def _extract_gemini_text(response) -> Optional[str]:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    parts = getattr(response, "parts", None) or []
    extracted = []
    for part in parts:
        pt = getattr(part, "text", None)
        if isinstance(pt, str) and pt.strip():
            extracted.append(pt)
    if extracted:
        return "\n".join(extracted)
    for cand in (getattr(response, "candidates", None) or []):
        for part in (getattr(getattr(cand, "content", None), "parts", None) or []):
            pt = getattr(part, "text", None)
            if isinstance(pt, str) and pt.strip():
                extracted.append(pt)
    return "\n".join(extracted) if extracted else None


def call_llm_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Gemini multimodal API (standalone, no autofigure2 dependency)."""
    if provider != "gemini":
        # For non-gemini, try autofigure2 import
        try:
            import autofigure2
            return autofigure2.call_llm_multimodal(
                contents, api_key, model, base_url, provider, max_tokens=max_tokens, temperature=temperature
            )
        except ImportError:
            raise RuntimeError(f"Provider '{provider}' requires autofigure2 with full dependencies")

    client = _get_gemini_client(api_key)
    try:
        response = _gemini_call_with_retry(
            client,
            model=model,
            contents=contents,
            config=_build_gemini_text_config(max_tokens, temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        last_err = e
    raise last_err


def call_llm_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Gemini text API (standalone)."""
    return call_llm_multimodal([prompt], api_key, model, base_url, provider, max_tokens=max_tokens, temperature=temperature)


# ---------------------------------------------------------------------------
# Standalone SAM3 Roboflow API
# ---------------------------------------------------------------------------

def _image_to_base64_file(image_path: str) -> str:
    img = Image.open(image_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _polygon_to_bbox(points, width, height):
    xs = [p[0] for p in points if len(p) >= 2]
    ys = [p[1] for p in points if len(p) >= 2]
    if not xs or not ys:
        return None
    x1, x2 = max(0, int(min(xs))), min(width, int(max(xs)))
    y1, y2 = max(0, int(min(ys))), min(height, int(max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _extract_roboflow_detections(response_json, image_size):
    width, height = image_size
    detections = []
    prompt_results = response_json.get("prompt_results") if isinstance(response_json, dict) else None
    if not isinstance(prompt_results, list):
        return detections
    for pr in prompt_results:
        if not isinstance(pr, dict):
            continue
        for pred in (pr.get("predictions") or []):
            if not isinstance(pred, dict):
                continue
            confidence = pred.get("confidence")
            for mask in (pred.get("masks") or []):
                points = []
                if isinstance(mask, list) and mask:
                    if isinstance(mask[0], (list, tuple)) and len(mask[0]) >= 2 and isinstance(mask[0][0], (int, float)):
                        points = mask
                    elif isinstance(mask[0], (list, tuple)):
                        for sub in mask:
                            if isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(sub[0], (int, float)):
                                points.append(sub)
                            elif isinstance(sub, (list, tuple)) and sub and isinstance(sub[0], (list, tuple)):
                                for pt in sub:
                                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                        points.append(pt)
                if not points:
                    continue
                xyxy = _polygon_to_bbox(points, width, height)
                if xyxy:
                    detections.append({"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": confidence})
    return detections


def _call_sam3_roboflow(image_b64, prompt, api_key, min_score):
    payload = {
        "image": {"type": "base64", "value": image_b64},
        "prompts": [{"type": "text", "text": prompt}],
        "format": "polygon",
        "output_prob_thresh": min_score,
    }
    url = f"{SAM3_ROBOFLOW_API_URL}?api_key={api_key}"
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=SAM3_API_TIMEOUT)
            if resp.status_code != 200:
                raise Exception(f"SAM3 API error: {resp.status_code} - {resp.text[:500]}")
            result = resp.json()
            if isinstance(result, dict) and "error" in result:
                raise Exception(f"SAM3 API error: {result.get('error')}")
            return result
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise
    raise RuntimeError("SAM3 Roboflow request failed")


def calculate_overlap_ratio(box1, box2):
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    return inter / min(area1, area2) if min(area1, area2) > 0 else 0.0


def merge_overlapping_boxes(boxes, overlap_threshold=0.9):
    """Deduplicate overlapping boxes using NMS-style suppression.

    Instead of expanding boxes (which causes chain-merging), keeps the
    highest-scoring box from each overlapping cluster.
    """
    if len(boxes) <= 1:
        return boxes

    # Sort by score descending so we keep the best detections
    scored = sorted(boxes, key=lambda b: b.get("score", 0), reverse=True)
    keep = []
    suppressed = set()

    for i, box_i in enumerate(scored):
        if i in suppressed:
            continue
        keep.append(box_i)
        for j in range(i + 1, len(scored)):
            if j in suppressed:
                continue
            ratio = calculate_overlap_ratio(box_i, scored[j])
            if ratio >= overlap_threshold:
                suppressed.add(j)

    # Also filter out boxes that are too large (>60% of image) — likely false positives
    # and boxes that are tiny (<5px in any dimension)
    filtered = []
    for b in keep:
        w = b.get("x2", 0) - b.get("x1", 0)
        h = b.get("y2", 0) - b.get("y1", 0)
        if w < 5 or h < 5:
            continue
        filtered.append(b)

    for i, box in enumerate(filtered):
        box["id"] = i
        box["label"] = f"<AF>{i + 1:02d}"
        box["width"] = box.get("x2", 0) - box.get("x1", 0)
        box["height"] = box.get("y2", 0) - box.get("y1", 0)
    return filtered


def segment_with_sam3(
    image_path: str,
    output_dir: str,
    text_prompts: str = "icon",
    min_score: float = 0.5,
    merge_threshold: float = 0.9,
    sam_backend: str = "roboflow",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
) -> tuple:
    """Standalone SAM3 segmentation via Roboflow API (no torch/transformers needed)."""
    print("\n" + "=" * 60)
    print("Step 2: SAM3 Segmentation")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_path)
    w, h = image.size
    print(f"Image size: {w} x {h}")

    prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    print(f"Prompts: {prompt_list}")

    if sam_backend != "roboflow":
        raise ValueError(f"Standalone mode only supports roboflow backend, got: {sam_backend}")

    api_key = sam_api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    image_b64 = _image_to_base64_file(image_path)

    all_boxes = []
    for prompt in prompt_list:
        print(f"\n  Detecting: '{prompt}'")
        response = _call_sam3_roboflow(image_b64, prompt, api_key, min_score)
        detections = _extract_roboflow_detections(response, (w, h))
        count = 0
        for det in detections:
            score = float(det.get("score") or 0)
            if score >= min_score:
                all_boxes.append({**det, "score": score, "prompt": prompt})
                count += 1
                print(f"    Object {count}: ({det['x1']},{det['y1']},{det['x2']},{det['y2']}) score={score:.3f}")
        print(f"  '{prompt}' detected {count} objects")

    # Assign labels
    valid_boxes = []
    for i, box in enumerate(all_boxes):
        valid_boxes.append({
            "id": i, "label": f"<AF>{i+1:02d}",
            "x1": box["x1"], "y1": box["y1"], "x2": box["x2"], "y2": box["y2"],
            "score": box["score"], "prompt": box.get("prompt", ""),
            "width": box["x2"] - box["x1"], "height": box["y2"] - box["y1"],
        })

    # Filter out boxes that are too large (>40% of image area) or too tiny
    area_limit = w * h * 0.4
    valid_boxes = [
        b for b in valid_boxes
        if 5 < b["width"] and 5 < b["height"]
        and b["width"] * b["height"] < area_limit
    ]

    if merge_threshold > 0 and len(valid_boxes) > 1:
        original_count = len(valid_boxes)
        valid_boxes = merge_overlapping_boxes(valid_boxes, merge_threshold)
        print(f"  Deduplicated: {original_count} -> {len(valid_boxes)}")

    # Draw samed.png
    from PIL import ImageDraw, ImageFont
    samed = image.copy()
    draw = ImageDraw.Draw(samed)
    for box in valid_boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = box["label"]
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(10, min(20, (x2-x1)//4)))
        except (OSError, IOError):
            font = ImageFont.load_default()
        try:
            draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
        except TypeError:
            draw.text((cx - 20, cy - 8), label, fill="white", font=font)

    samed_path = os.path.join(output_dir, "samed.png")
    samed.save(samed_path)

    boxlib_data = {
        "image_size": {"width": w, "height": h},
        "prompts_used": prompt_list,
        "boxes": valid_boxes,
    }
    boxlib_path = os.path.join(output_dir, "boxlib.json")
    with open(boxlib_path, "w", encoding="utf-8") as f:
        json.dump(boxlib_data, f, indent=2, ensure_ascii=False)

    print(f"  samed.png saved: {samed_path}")
    print(f"  boxlib.json saved: {boxlib_path}")
    return (samed_path, boxlib_path, valid_boxes)

# ---------------------------------------------------------------------------
# Gemini Vision icon detection (supplements SAM3)
# ---------------------------------------------------------------------------

def detect_icons_with_gemini(
    image_path: str,
    api_key: str,
    model: str = None,
) -> Tuple[List[Dict], Optional[str]]:
    """Use Gemini Vision to detect all icon/image elements in a figure.

    Sends the image to Gemini and asks it to identify bounding boxes
    for all visual elements (icons, symbols, charts, etc.).

    Returns (list_of_boxes, error_string_or_None).
    Each box: {x1, y1, x2, y2, score, prompt}.
    """
    img = Image.open(image_path)
    w, h = img.size
    model = model or "gemini-2.5-flash"

    prompt = f"""This is a scientific/academic figure image ({w}x{h} pixels).

Identify ALL visual elements that are icons, logos, symbols, small images,
mini-charts, mini-graphs, or any non-text decorative visual element.

Include:
- Robot/brain/chip/gear/person icons
- Small colored circles or dots
- Mini bar charts, line graphs, pie charts
- Circuit diagrams or schematic elements
- Stacked paper/document icons
- Arrow decorations (NOT connecting arrows between boxes)
- Any visual element that is NOT plain text and NOT a simple rectangular box border

Do NOT include:
- Plain text labels
- Simple rectangular box outlines with only text inside
- Connecting arrows/lines between components

For each element, provide its bounding box in PIXEL coordinates.
Return ONLY a valid JSON array, no markdown fences, no other text:
[{{"x1":10,"y1":20,"x2":50,"y2":60,"label":"robot icon"}},...]

Be thorough. Typical academic figures have 10-30 visual elements."""

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = _gemini_call_with_retry(
            client,
            model=model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                max_output_tokens=8000,
                temperature=0.2,
            ),
        )
        text = _extract_gemini_text(response)
        if not text:
            return ([], "No response from Gemini")

        # Extract JSON from response (may be wrapped in markdown fences)
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()

        boxes_raw = json.loads(text)
        if not isinstance(boxes_raw, list):
            return ([], f"Expected list, got {type(boxes_raw)}")

        result = []
        for box in boxes_raw:
            if not isinstance(box, dict):
                continue
            try:
                x1 = max(0, min(w, int(box["x1"])))
                y1 = max(0, min(h, int(box["y1"])))
                x2 = max(0, min(w, int(box["x2"])))
                y2 = max(0, min(h, int(box["y2"])))
            except (KeyError, ValueError, TypeError):
                continue
            bw, bh = x2 - x1, y2 - y1
            if bw < 5 or bh < 5:
                continue
            if bw * bh > w * h * 0.4:
                continue
            result.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "score": 0.8,
                "prompt": "gemini_vision",
                "label": box.get("label", ""),
            })

        return (result, None)

    except json.JSONDecodeError as e:
        return ([], f"JSON parse error: {e}")
    except Exception as e:
        return ([], f"Gemini detection error: {e}")


def merge_sam3_and_gemini_boxes(
    sam3_boxes: List[Dict],
    gemini_boxes: List[Dict],
    iou_threshold: float = 0.5,
) -> List[Dict]:
    """Merge SAM3 and Gemini detections, preferring SAM3 for overlapping regions.

    For each Gemini box, if it overlaps significantly with any SAM3 box,
    discard it (SAM3 already has that detection). Otherwise, add it as new.
    """
    merged = list(sam3_boxes)

    for gbox in gemini_boxes:
        overlaps = False
        for sbox in merged:
            ratio = calculate_overlap_ratio(gbox, sbox)
            if ratio >= iou_threshold:
                overlaps = True
                break
        if not overlaps:
            merged.append(gbox)

    # Re-assign IDs and labels
    for i, box in enumerate(merged):
        box["id"] = i
        box["label"] = f"<AF>{i + 1:02d}"
        box["width"] = box.get("x2", 0) - box.get("x1", 0)
        box["height"] = box.get("y2", 0) - box.get("y1", 0)

    return merged


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRAWIO_CLI = "/Applications/draw.io.app/Contents/MacOS/draw.io"
DEFAULT_TARGET_SSIM = 0.90
DEFAULT_MAX_ITERATIONS = 4
DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
DEFAULT_IMAGE_GEN_MODEL = "gemini-3-pro-image-preview"  # supports image output


# ============================================================================
# Step 1: Image Grid Splitting
# ============================================================================

def split_image_grid(
    image_path: str,
    output_dir: str,
    grid: str = "auto",
) -> Tuple[List[str], Optional[str]]:
    """Split a multi-figure image into individual panel PNGs.

    Args:
        image_path: Path to input image
        output_dir: Directory to save panel PNGs
        grid: Grid spec - "auto", "1x1", "2x2", "1x2", "2x1", etc.

    Returns:
        (list_of_panel_paths, error_string_or_None)
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        return (None, f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    if grid == "1x1" or grid == "1":
        # Single figure, just copy
        out_path = os.path.join(output_dir, "panel_1.png")
        cv2.imwrite(out_path, img)
        return ([out_path], None)

    if grid == "auto":
        rows, cols = _detect_grid(img)
    else:
        parts = grid.lower().split("x")
        rows, cols = int(parts[0]), int(parts[1])

    if rows == 1 and cols == 1:
        out_path = os.path.join(output_dir, "panel_1.png")
        cv2.imwrite(out_path, img)
        return ([out_path], None)

    panels = []
    panel_h = h // rows
    panel_w = w // cols

    for r in range(rows):
        for c in range(cols):
            y1 = r * panel_h
            y2 = (r + 1) * panel_h if r < rows - 1 else h
            x1 = c * panel_w
            x2 = (c + 1) * panel_w if c < cols - 1 else w
            panel = img[y1:y2, x1:x2]
            idx = r * cols + c + 1
            out_path = os.path.join(output_dir, f"panel_{idx}.png")
            cv2.imwrite(out_path, panel)
            panels.append(out_path)

    print(f"Split image into {rows}x{cols} = {len(panels)} panels")
    return (panels, None)


def _detect_grid(img: np.ndarray) -> Tuple[int, int]:
    """Auto-detect grid layout using projection analysis."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    def find_midpoint_gaps(proj, length, min_gap_width=10):
        threshold = np.mean(proj) * 0.3
        is_gap = proj < threshold
        gaps = []
        in_gap = False
        gap_start = 0
        margin = length * 0.2
        for i in range(int(margin), int(length - margin)):
            if is_gap[i] and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_gap[i] and in_gap:
                if i - gap_start >= min_gap_width:
                    gaps.append((gap_start + i) // 2)
                in_gap = False
        return gaps

    # Check for horizontal split (rows)
    row_proj = np.mean(gray, axis=1)
    row_gaps = find_midpoint_gaps(row_proj, h)
    rows = len(row_gaps) + 1

    # Check for vertical split (cols)
    col_proj = np.mean(gray, axis=0)
    col_gaps = find_midpoint_gaps(col_proj, w)
    cols = len(col_gaps) + 1

    rows = min(rows, 4)
    cols = min(cols, 4)

    print(f"Auto-detected grid: {rows}x{cols}")
    return (rows, cols)


# ============================================================================
# Step 3: Gemini Icon Regeneration
# ============================================================================

def regenerate_icons_with_gemini(
    image_path: str,
    output_dir: str,
    api_key: str,
    model: str = None,
    provider: ProviderType = "gemini",
    base_url: str = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Use Gemini to regenerate clean icons from the reference image.

    Uses an image-generation-capable model (DEFAULT_IMAGE_GEN_MODEL) since
    text-only models like gemini-2.5-pro/flash cannot output images.

    Returns:
        (icon_sheet_path, error_string_or_None)
    """
    os.makedirs(output_dir, exist_ok=True)

    if base_url is None:
        base_url = PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")

    # Always use image-generation model for this step
    image_model = model or DEFAULT_IMAGE_GEN_MODEL

    img = Image.open(image_path)

    prompt = """Look at this scientific diagram. Regenerate ALL icons/symbols you see as individual clean icons on a white background.
Arrange them in a grid layout. Each icon should be:
- Clean, high-quality, standalone
- No flowchart elements, no arrows, no text labels
- Just the individual visual icons/symbols
Output a single image with all icons arranged in a grid.
Label each icon with a number (1, 2, 3...) below it for identification."""

    # Use Gemini image generation with image-capable model
    print(f"  Using image generation model: {image_model}")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = _gemini_call_with_retry(
            client,
            model=image_model,
            contents=[prompt, img],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        # Extract generated image
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                img_data = part.inline_data.data
                icon_sheet_path = os.path.join(output_dir, "gemini_icons.png")
                with open(icon_sheet_path, "wb") as f:
                    f.write(img_data)
                print(f"Gemini icon sheet saved: {icon_sheet_path}")
                return (icon_sheet_path, None)

        # If no image returned, try text-only response
        return (None, "Gemini did not return an image for icon regeneration")

    except Exception as e:
        return (None, f"Gemini icon regeneration failed: {e}")


def extract_icons_from_sheet(
    icon_sheet_path: str,
    output_dir: str,
) -> Tuple[Optional[List[IconInfo]], Optional[str]]:
    """Split a Gemini-generated icon sheet into individual icon PNGs.

    Returns:
        (list_of_IconInfo, error_string_or_None)
    """
    icons, err = extract_icons(
        image_path=icon_sheet_path,
        output_dir=output_dir,
        bg_mode="auto",
        make_svg=False,
    )
    if err:
        return (None, err)

    print(f"Extracted {len(icons)} individual icons from sheet")
    return (icons, None)


def crop_icons_from_original(
    image_path: str,
    sam_boxes: list,
    output_dir: str,
) -> List[Dict[str, Any]]:
    """Crop each SAM3-detected icon region from the original image.

    Returns a list of dicts sorted by (y1, x1) with keys:
        label, label_clean, crop_path, x1, y1, width, height
    """
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)

    sorted_boxes = sorted(
        sam_boxes, key=lambda b: (b.get("y1", 0), b.get("x1", 0))
    )

    crops = []
    for i, box in enumerate(sorted_boxes):
        label = box.get("label", f"<AF>{i+1:02d}")
        label_clean = label.replace("<", "").replace(">", "")
        x1, y1 = box.get("x1", 0), box.get("y1", 0)
        x2 = box.get("x2", x1 + box.get("width", 60))
        y2 = box.get("y2", y1 + box.get("height", 60))

        crop_path = os.path.join(output_dir, f"crop_{i+1:02d}_{label_clean}.png")
        cropped = img.crop((x1, y1, x2, y2))
        cropped.save(crop_path)

        crops.append({
            "label": label,
            "label_clean": label_clean,
            "crop_path": crop_path,
            "idx": i,
            "x1": x1, "y1": y1,
            "width": x2 - x1, "height": y2 - y1,
        })

    print(f"  Cropped {len(crops)} icon regions from original")
    return crops


def _make_crop_reference_sheet(crops: List[Dict], output_path: str) -> str:
    """Arrange cropped icons into a numbered reference sheet image.

    Each icon is placed in a grid cell with its index number below it.
    """
    n = len(crops)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 200, 220  # each cell: 200x200 icon + 20px label
    sheet_w = cols * cell_w + 20
    sheet_h = rows * cell_h + 20

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    for i, crop_info in enumerate(crops):
        col = i % cols
        row = i // cols
        cx = 10 + col * cell_w
        cy = 10 + row * cell_h

        icon = Image.open(crop_info["crop_path"])
        # Fit icon into 180x180 preserving aspect ratio
        icon.thumbnail((180, 180), Image.LANCZOS)
        # Center in cell
        ox = cx + (cell_w - icon.width) // 2
        oy = cy + (200 - icon.height) // 2
        sheet.paste(icon, (ox, oy))

        # Draw index number below icon
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(sheet)
            label = str(i + 1)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            except (IOError, OSError):
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            draw.text((cx + (cell_w - tw) // 2, cy + 195), label, fill="black", font=font)
        except ImportError:
            pass

    sheet.save(output_path)
    return output_path


def regenerate_icons_hd(
    crops: List[Dict[str, Any]],
    original_image_path: str,
    output_dir: str,
    api_key: str,
    provider: ProviderType = "gemini",
    base_url: str = None,
    tiny_threshold: int = 400,       # area < 20x20 = skip HD
    batch_threshold: int = 2500,     # area < 50x50 = batch together
    batch_size: int = 8,             # max icons per batch call
) -> List[Dict[str, Any]]:
    """Regenerate HD versions of cropped icons via Gemini with smart dispatch.

    Three tiers based on icon area (width * height):
      - Tiny  (< tiny_threshold):  skip HD, use original crop directly
      - Small (< batch_threshold): batch multiple icons in one Gemini call
      - Large (>= batch_threshold): send individually for best quality

    Returns the crops list with 'hd_path' and 'icon_path' set.
    """
    os.makedirs(output_dir, exist_ok=True)

    if base_url is None:
        base_url = PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")

    # Classify icons into tiers
    tiny, small, large = [], [], []
    for i, crop in enumerate(crops):
        area = crop.get("width", 0) * crop.get("height", 0)
        crop["_idx"] = i  # track original index
        if area < tiny_threshold:
            tiny.append(crop)
        elif area < batch_threshold:
            small.append(crop)
        else:
            large.append(crop)

    print(f"  Smart dispatch: {len(tiny)} tiny (skip), {len(small)} small (batch), {len(large)} large (individual)")
    api_calls = len(large) + (len(small) + batch_size - 1) // batch_size if small else len(large)
    print(f"  Estimated Gemini API calls: ~{api_calls}")

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"  Cannot init Gemini client: {e}")
        for crop in crops:
            crop["icon_path"] = crop["crop_path"]
        return crops

    # --- Tier 1: Tiny icons — skip HD, use crop directly ---
    for crop in tiny:
        print(f"  Icon {crop['_idx']+1}: tiny ({crop.get('width',0)}x{crop.get('height',0)}), skip HD")

    # --- Tier 2: Small icons — batch together ---
    if small:
        for batch_start in range(0, len(small), batch_size):
            batch = small[batch_start:batch_start + batch_size]
            batch_nums = [c["_idx"] + 1 for c in batch]
            print(f"  Batch HD: icons {batch_nums} ({len(batch)} icons)...")

            # Build batch reference sheet
            batch_sheet_path = os.path.join(output_dir, f"batch_sheet_{batch_start}.png")
            _make_crop_reference_sheet(batch, batch_sheet_path)
            batch_sheet_img = Image.open(batch_sheet_path)

            prompt = f"""You are given a reference sheet showing {len(batch)} cropped icons numbered 1-{len(batch)}.

Regenerate ALL {len(batch)} icons as clean, high-quality versions:
- Output exactly {len(batch)} icons arranged in a single row or grid, numbered 1-{len(batch)} in the SAME order
- Each icon should be clean with white background
- Preserve the visual content and style of each original icon
- Number each icon clearly below it"""

            try:
                response = _gemini_call_with_retry(
                    client,
                    model=DEFAULT_IMAGE_GEN_MODEL,
                    contents=[prompt, batch_sheet_img],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        hd_sheet_path = os.path.join(output_dir, f"hd_batch_{batch_start}.png")
                        with open(hd_sheet_path, "wb") as f:
                            f.write(part.inline_data.data)

                        hd_icons, err = extract_icons(
                            image_path=hd_sheet_path,
                            output_dir=output_dir,
                            bg_mode="auto",
                            make_svg=False,
                        )
                        if hd_icons and len(hd_icons) == len(batch):
                            hd_sorted = sorted(hd_icons, key=lambda ic: (ic.y, ic.x))
                            for j, crop in enumerate(batch):
                                crop["hd_path"] = hd_sorted[j].png_path
                            print(f"    Batch OK: {len(hd_icons)} HD icons")
                        elif hd_icons:
                            print(f"    Batch count mismatch: got {len(hd_icons)}, expected {len(batch)}, falling back")
                            # Fall back to individual for this batch
                            for crop in batch:
                                if crop.get("hd_path"):
                                    continue
                                large.append(crop)  # move to individual queue
                        else:
                            print(f"    Batch extraction failed, falling back to individual")
                            for crop in batch:
                                large.append(crop)
                        break
                else:
                    print(f"    No image returned, falling back to individual")
                    large.extend(batch)
            except Exception as e:
                print(f"    Batch failed ({e}), falling back to individual")
                large.extend(batch)

            time.sleep(1.5)

    # --- Tier 3: Large icons — individual calls ---
    for crop in large:
        if crop.get("hd_path"):
            continue

        idx = crop["_idx"]
        crop_img = Image.open(crop["crop_path"])
        single_prompt = (
            "Regenerate this icon as a clean, high-quality version with white background. "
            "Output ONLY the single icon image, preserving its visual content and style."
        )

        try:
            response = _gemini_call_with_retry(
                client,
                model=DEFAULT_IMAGE_GEN_MODEL,
                contents=[single_prompt, crop_img],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    hd_path = os.path.join(output_dir, f"hd_{idx+1:02d}.png")
                    with open(hd_path, "wb") as f:
                        f.write(part.inline_data.data)
                    crop["hd_path"] = hd_path
                    print(f"  Icon {idx+1}/{len(crops)}: HD generated")
                    break
            else:
                print(f"  Icon {idx+1}/{len(crops)}: no image returned, using crop")
        except Exception as e:
            print(f"  Icon {idx+1}/{len(crops)}: failed after retries ({e}), using crop")

        time.sleep(1.5)

    # Set intermediate icon_path: prefer HD, fall back to crop
    for crop in crops:
        crop["icon_path"] = crop.get("hd_path") or crop["crop_path"]

    # --- RMBG-2.0 background removal ---
    remover, rmbg_err = _get_rmbg_remover(output_dir)
    if rmbg_err:
        print(f"  RMBG-2.0 skipped: {rmbg_err}")
    else:
        print(f"  Removing backgrounds with RMBG-2.0...")
        for crop in crops:
            src = crop["icon_path"]
            nobg_path = src.rsplit(".", 1)[0] + "_nobg.png"
            try:
                remover.remove_background(src, nobg_path)
                crop["icon_path"] = nobg_path
            except Exception as e:
                print(f"    RMBG failed for {os.path.basename(src)}: {e}")
        rmbg_count = sum(1 for c in crops if c["icon_path"].endswith("_nobg.png"))
        print(f"  RMBG-2.0: {rmbg_count}/{len(crops)} icons processed")

    hd_count = sum(1 for c in crops if c.get("hd_path"))
    skip_count = len(tiny)
    print(f"  Final: {hd_count}/{len(crops)} HD, {skip_count} tiny (skipped), {len(crops)-hd_count-skip_count} crop fallback")
    return crops


def _crop_similarity(original: np.ndarray, box: dict, icon_path: str) -> float:
    """Compute visual similarity between an original-image crop and an icon.

    Uses histogram correlation in HSV space (robust to small shifts/resizing).
    Returns a score in [-1, 1]; higher is more similar.
    """
    x1, y1 = box.get("x1", 0), box.get("y1", 0)
    x2 = box.get("x2", x1 + box.get("width", 60))
    y2 = box.get("y2", y1 + box.get("height", 60))
    crop = original[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return -1.0

    icon_img = cv2.imread(icon_path, cv2.IMREAD_COLOR)
    if icon_img is None:
        return -1.0

    # Resize icon to crop dimensions for fair comparison
    icon_resized = cv2.resize(icon_img, (crop.shape[1], crop.shape[0]))

    # HSV histogram comparison (more robust than RGB)
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    icon_hsv = cv2.cvtColor(icon_resized, cv2.COLOR_BGR2HSV)
    h_bins, s_bins = 16, 16
    ranges = [0, 180, 0, 256]
    h1 = cv2.calcHist([crop_hsv], [0, 1], None, [h_bins, s_bins], ranges)
    h2 = cv2.calcHist([icon_hsv], [0, 1], None, [h_bins, s_bins], ranges)
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def match_icons_to_placeholders(
    regen_icons: List[IconInfo],
    sam_boxes: list,
    image_path: str,
) -> List[Dict[str, Any]]:
    """Match regenerated icons to SAM3-detected placeholder positions.

    Uses template matching (histogram correlation) to find the best icon for
    each AF placeholder, instead of naive index-based matching.

    Returns:
        List of dicts with keys: label, label_clean, icon_path, x1, y1, width, height
    """
    matched = []

    # Sort SAM boxes by position (top-to-bottom, left-to-right)
    sorted_boxes = sorted(
        sam_boxes, key=lambda b: (b.get("y1", 0), b.get("x1", 0))
    )

    # Load original image for template matching
    original = cv2.imread(image_path, cv2.IMREAD_COLOR) if image_path else None

    # Collect valid icon paths
    icon_paths = []
    for ic in regen_icons:
        p = ic.png_path if ic.png_path and os.path.exists(ic.png_path) else None
        icon_paths.append(p)

    # --- Build similarity matrix and do greedy best-match ---
    used_icons = set()  # indices into regen_icons already assigned

    if original is not None and icon_paths:
        n_boxes = len(sorted_boxes)
        n_icons = len(icon_paths)
        sim = np.full((n_boxes, n_icons), -1.0)

        print("  Computing icon–placeholder similarity matrix...")
        for i, box in enumerate(sorted_boxes):
            for j, ip in enumerate(icon_paths):
                if ip:
                    sim[i][j] = _crop_similarity(original, box, ip)

        # Greedy assignment: pick best (box, icon) pair by score, repeat
        for _ in range(min(n_boxes, n_icons)):
            best_val = sim.max()
            if best_val < 0:
                break
            bi, bj = np.unravel_index(sim.argmax(), sim.shape)
            bi, bj = int(bi), int(bj)

            label = sorted_boxes[bi].get("label", f"<AF>{bi+1:02d}")
            label_clean = label.replace("<", "").replace(">", "")
            matched.append({
                "label": label,
                "label_clean": label_clean,
                "x1": sorted_boxes[bi].get("x1", 0),
                "y1": sorted_boxes[bi].get("y1", 0),
                "width": sorted_boxes[bi].get("width", 60),
                "height": sorted_boxes[bi].get("height", 60),
                "icon_path": icon_paths[bj],
            })
            used_icons.add(bj)
            print(f"  {label_clean}: matched icon_{bj+1:02d} (score={best_val:.3f})")
            # Eliminate this box-row and icon-column
            sim[bi, :] = -2.0
            sim[:, bj] = -2.0

    # --- Handle unmatched boxes: crop from original ---
    matched_labels = {m["label_clean"] for m in matched}
    for i, box in enumerate(sorted_boxes):
        label = box.get("label", f"<AF>{i+1:02d}")
        label_clean = label.replace("<", "").replace(">", "")
        if label_clean in matched_labels:
            continue

        icon_path = None
        if image_path:
            try:
                x1, y1 = box.get("x1", 0), box.get("y1", 0)
                x2 = box.get("x2", x1 + box.get("width", 60))
                y2 = box.get("y2", y1 + box.get("height", 60))
                crop_dir = os.path.join(os.path.dirname(image_path), "icons_crop")
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, f"crop_{label_clean}.png")
                img = Image.open(image_path)
                cropped = img.crop((x1, y1, x2, y2))
                cropped.save(crop_path)
                icon_path = crop_path
                print(f"  {label_clean}: cropped from original (no icon matched)")
            except Exception as e:
                print(f"  {label_clean}: crop fallback failed: {e}")

        matched.append({
            "label": label,
            "label_clean": label_clean,
            "x1": box.get("x1", 0),
            "y1": box.get("y1", 0),
            "width": box.get("width", 60),
            "height": box.get("height", 60),
            "icon_path": icon_path,
        })

    print(f"Matched {len(matched)} icons to placeholders "
          f"({len(used_icons)} by template, "
          f"{len(matched) - len(used_icons)} cropped)")
    return matched


# ============================================================================
# Step 4: draw.io Template Generation
# ============================================================================

def _extract_svg_code(content: str) -> Optional[str]:
    """Extract <svg>...</svg> from LLM response."""
    m = re.search(r'(<svg[\s\S]*?</svg>)', content, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'```(?:svg|xml)?\s*([\s\S]*?)```', content)
    if m:
        code = m.group(1).strip()
        if code.startswith("<svg"):
            return code
    if content.strip().startswith("<svg"):
        return content.strip()
    return None


def _validate_svg_syntax(svg_code: str) -> Tuple[bool, List[str]]:
    """Basic SVG XML syntax check."""
    errors = []
    try:
        ET.fromstring(svg_code)
    except ET.ParseError as e:
        errors.append(f"SVG parse error: {e}")
        return (False, errors)
    if "<svg" not in svg_code:
        errors.append("No <svg> root element found")
        return (False, errors)
    return (True, [])


def _fix_svg_with_llm(svg_code, errors, api_key, model, base_url, provider, max_attempts=2):
    """Try to fix SVG syntax errors via LLM."""
    current = svg_code
    for attempt in range(max_attempts):
        err_text = "; ".join(errors)
        prompt = (
            f"The following SVG has syntax errors:\n{err_text}\n\n"
            f"Fix the errors and output ONLY the corrected SVG (starting with <svg, ending with </svg>).\n\n"
            f"{current}"
        )
        resp = call_llm_multimodal(
            contents=[prompt], api_key=api_key, model=model,
            base_url=base_url, provider=provider, max_tokens=50000,
        )
        if resp:
            fixed = _extract_svg_code(resp)
            if fixed:
                ok, new_errors = _validate_svg_syntax(fixed)
                if ok:
                    return (fixed, None)
                errors = new_errors
                current = fixed
    return (current, f"SVG still has errors after {max_attempts} fix attempts")


def generate_drawio_template(
    figure_path: str,
    samed_path: str,
    boxlib_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    base_url: str = None,
    provider: ProviderType = "gemini",
    no_icon_mode: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Generate draw.io XML template via SVG-first approach.

    Strategy: LLM generates SVG (high quality) → deterministic conversion to draw.io XML.
    This leverages the LLM's strong SVG knowledge while producing editable draw.io output.

    Returns:
        (output_path, error_string_or_None)
    """
    from svg_to_drawio import svg_to_drawio

    print("\n" + "=" * 60)
    print("Step 4: Generate draw.io Template (SVG-first approach)")
    print("=" * 60)

    if base_url is None:
        base_url = PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")

    figure_img = Image.open(figure_path)
    samed_img = Image.open(samed_path)
    fig_w, fig_h = figure_img.size
    print(f"Figure dimensions: {fig_w} x {fig_h}")

    # --- Step 4a: Generate SVG template (LLM excels at this) ---
    print("  [4a] Generating SVG template via LLM...")

    # Common SVG format constraints to ensure flat, parseable output
    svg_format_rules = """
SVG FORMAT RULES (MUST FOLLOW):
- Use INLINE style attributes only (fill="...", stroke="...", etc.)
- Do NOT use CSS classes, <style> blocks, or class attributes
- Use ABSOLUTE coordinates for ALL elements (no transform="translate(...)")
- Do NOT nest elements inside <g> groups (except for <AF> icon placeholders)
- Each element must have explicit x, y, width, height (or cx, cy, r for circles)
- Use simple SVG elements: <rect>, <text>, <path>, <line>, <circle>, <image>
- Arrows: use <path> with marker-end="url(#arrow)" and define markers in <defs>
- Text: use text-anchor, font-family, font-size, font-weight as inline attributes"""

    if no_icon_mode:
        prompt = f"""编写 SVG 代码来尽可能像素级复现这张图片。

当前 SAM3 没有检测到任何有效图标，因此这是一个无图标回退模式任务：
- 不要添加任何灰色矩形占位符
- 不要添加任何 <AF>01 / <AF>02 标签
- 所有可见内容都应直接用 SVG 元素复现

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {fig_w} x {fig_h} pixels
- Your SVG MUST use these EXACT dimensions:
  - Set viewBox="0 0 {fig_w} {fig_h}"
  - Set width="{fig_w}" height="{fig_h}"
- DO NOT scale or resize the SVG
{svg_format_rules}

Image 1 is the original target figure.
Image 2 is the SAM reference image.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""
    else:
        prompt = f"""编写svg代码来实现像素级别的复现这张图片（除了图标用相同大小的矩形占位符填充之外其他文字和组件都要保持一致（即灰色矩形覆盖的内容就是图标））

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {fig_w} x {fig_h} pixels
- Your SVG MUST use these EXACT dimensions to ensure accurate icon placement:
  - Set viewBox="0 0 {fig_w} {fig_h}"
  - Set width="{fig_w}" height="{fig_h}"
- DO NOT scale or resize the SVG
{svg_format_rules}

ARROW/CONNECTION RULES (非常重要):
- 仔细观察原图中每条箭头/连线的起点和终点，确保SVG中每条连线连接的是正确的两个元素
- 不要凭想象添加原图中不存在的连线
- 不要遗漏原图中存在的连线
- 保持箭头方向一致（箭头朝向哪端）
- 对于有弯折的连线，使用<path>准确复现弯折路径，不要简化为直线
- 对于虚线箭头，使用 stroke-dasharray 属性
- 先画所有矩形/文字/图标占位，最后画所有连线，确保连线端点精确对准目标元素的边界

PLACEHOLDER STYLE REQUIREMENT:
Look at the second image (samed.png) - each icon area is marked with a gray rectangle (#808080), black border, and a centered label like <AF>01, <AF>02, etc.

Your SVG placeholders MUST match this exact style:
- Rectangle with fill="#808080" and stroke="black" stroke-width="2"
- Centered white text showing the same label (<AF>01, <AF>02, etc.)
- Wrap each placeholder in a <g> element with id matching the label (e.g., id="AF01")

Example placeholder structure:
<g id="AF01">
  <rect x="100" y="50" width="80" height="80" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="140" y="90" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="14">&lt;AF&gt;01</text>
</g>

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    contents = [prompt, figure_img, samed_img]

    print(f"  Sending multimodal request to {provider} ({model})...")
    content = call_llm_multimodal(
        contents=contents,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        max_tokens=50000,
        temperature=0.4,
    )

    if not content:
        return (None, "LLM returned empty response for SVG template")

    svg_code = _extract_svg_code(content)
    if not svg_code:
        return (None, "Could not extract SVG code from LLM response")

    # Validate and fix SVG
    ok, errors = _validate_svg_syntax(svg_code)
    if not ok:
        print(f"  SVG has syntax errors, attempting fix...")
        svg_code, fix_err = _fix_svg_with_llm(
            svg_code, errors, api_key, model, base_url, provider,
        )
        if fix_err:
            print(f"  Warning: {fix_err}")

    # Save intermediate SVG
    svg_path = output_path.replace(".drawio", ".svg")
    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"  SVG template saved: {svg_path}")

    # --- Step 4b: Convert SVG → draw.io XML (deterministic) ---
    print("  [4b] Converting SVG → draw.io XML...")
    drawio_path, conv_err = svg_to_drawio(svg_path, output_path)
    if conv_err:
        return (None, f"SVG→draw.io conversion failed: {conv_err}")

    print(f"  draw.io template saved: {drawio_path}")
    return (drawio_path, None)


def extract_drawio_code(content: str) -> Optional[str]:
    """Extract <mxGraphModel>...</mxGraphModel> from LLM response."""
    # Try mxGraphModel first
    pattern = r'(<mxGraphModel[\s\S]*?</mxGraphModel>)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try mxfile wrapper
    pattern = r'(<mxfile[\s\S]*?</mxfile>)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        # Extract inner mxGraphModel
        inner = re.search(r'(<mxGraphModel[\s\S]*?</mxGraphModel>)', match.group(1))
        if inner:
            return inner.group(1)
        return match.group(1)

    # Try code block
    pattern = r'```(?:xml|drawio)?\s*([\s\S]*?)```'
    match = re.search(pattern, content)
    if match:
        code = match.group(1).strip()
        if '<mxGraphModel' in code:
            inner = re.search(r'(<mxGraphModel[\s\S]*?</mxGraphModel>)', code)
            if inner:
                return inner.group(1)

    # Raw XML starting with <mxGraphModel
    if content.strip().startswith('<mxGraphModel'):
        return content.strip()

    return None


def validate_drawio_syntax(xml_code: str) -> Tuple[bool, List[str]]:
    """Validate draw.io XML syntax."""
    errors = []
    try:
        root = ET.fromstring(xml_code)
    except ET.ParseError as e:
        return (False, [f"XML parse error: {e}"])

    # Check for root cells
    tag = root.tag
    if tag == "mxfile":
        diagrams = root.findall(".//mxGraphModel")
        if not diagrams:
            errors.append("No mxGraphModel found inside mxfile")
            return (False, errors)
        root = diagrams[0]
    elif tag != "mxGraphModel":
        errors.append(f"Root element is '{tag}', expected 'mxGraphModel' or 'mxfile'")
        return (False, errors)

    # Check for id=0 and id=1 cells
    all_cells = root.findall(".//mxCell")
    ids = {cell.get("id") for cell in all_cells}
    if "0" not in ids:
        errors.append("Missing root cell with id='0'")
    if "1" not in ids:
        errors.append("Missing default parent cell with id='1'")

    if errors:
        return (False, errors)
    return (True, [])


def fix_drawio_with_llm(
    xml_code: str,
    errors: List[str],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_attempts: int = 3,
) -> Tuple[str, Optional[str]]:
    """Fix draw.io XML syntax errors using LLM."""
    current = xml_code

    for attempt in range(max_attempts):
        error_text = "\n".join(errors)
        prompt = f"""Fix the following draw.io XML that has syntax errors.

ERRORS:
{error_text}

CURRENT XML:
{current}

Rules:
- Output ONLY the fixed XML starting with <mxGraphModel and ending with </mxGraphModel>
- Ensure <mxCell id="0"/> and <mxCell id="1" parent="0"/> exist in <root>
- No XML comments
- All attribute values must be properly quoted
- All special characters must be escaped (&amp; &lt; &gt; &quot;)"""

        response = call_llm_text(
            prompt=prompt,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            max_tokens=50000,
            temperature=0.2,
        )

        if not response:
            continue

        fixed = extract_drawio_code(response)
        if not fixed:
            continue

        is_valid, new_errors = validate_drawio_syntax(fixed)
        if is_valid:
            print(f"  draw.io XML fixed after {attempt + 1} attempt(s)")
            return (fixed, None)
        errors = new_errors

    return (current, f"Could not fix XML after {max_attempts} attempts: {errors}")


def check_and_fix_drawio(
    xml_code: str,
    api_key: str,
    model: str,
    base_url: str = None,
    provider: ProviderType = "gemini",
    max_fix_attempts: int = 3,
) -> Tuple[str, Optional[str]]:
    """Validate draw.io XML and fix if needed."""
    if base_url is None:
        base_url = PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")

    is_valid, errors = validate_drawio_syntax(xml_code)
    if is_valid:
        print("  draw.io XML validation: OK")
        return (xml_code, None)

    print(f"  draw.io XML has {len(errors)} error(s), attempting fix...")
    return fix_drawio_with_llm(
        xml_code=xml_code,
        errors=errors,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        max_attempts=max_fix_attempts,
    )


def wrap_in_mxfile(mxgraph_xml: str, page_name: str = "Figure") -> str:
    """Wrap mxGraphModel XML in mxfile container."""
    # If already wrapped, return as-is
    if mxgraph_xml.strip().startswith("<mxfile"):
        return mxgraph_xml

    return f"""<mxfile host="app" type="device">
  <diagram name="{page_name}" id="page1">
    {mxgraph_xml}
  </diagram>
</mxfile>"""


def combine_pages(page_xmls: List[str], page_names: List[str] = None) -> str:
    """Combine multiple mxGraphModel XMLs into a single multi-page .drawio file."""
    import uuid
    pages = []
    for i, xml in enumerate(page_xmls):
        name = page_names[i] if page_names and i < len(page_names) else f"Figure {i+1}"
        # Extract mxGraphModel if wrapped in mxfile
        inner = extract_drawio_code(xml)
        if inner is None:
            inner = xml
        uid = uuid.uuid4().hex[:20]
        pages.append(f'  <diagram name="{name}" id="{uid}">\n    {inner}\n  </diagram>')

    return '<mxfile host="app.diagrams.net" type="device">\n' + "\n".join(pages) + "\n</mxfile>"


def combine_pages_single(
    page_xmls: List[str],
    grid: str = "2x2",
    panel_widths: List[float] = None,
    panel_heights: List[float] = None,
) -> str:
    """Merge multiple panel XMLs into ONE page with grid layout.

    Each panel's elements are offset so they sit side-by-side in a grid,
    matching the original multi-figure image layout.

    Returns a single-page .drawio XML string.
    """
    import uuid

    cols, rows = 2, 2
    if grid == "1x2":
        cols, rows = 2, 1
    elif grid == "2x1":
        cols, rows = 1, 2
    elif grid == "1x1":
        cols, rows = 1, 1

    # Parse each panel to extract cells and page dimensions
    all_cells = []
    pw_default, ph_default = 688, 384

    for idx, xml in enumerate(page_xmls):
        inner = extract_drawio_code(xml)
        if inner is None:
            inner = xml
        try:
            root = ET.fromstring(inner)
        except ET.ParseError:
            continue

        # Get panel dimensions from mxGraphModel attributes
        pw = float(root.get("pageWidth", panel_widths[idx] if panel_widths else pw_default))
        ph = float(root.get("pageHeight", panel_heights[idx] if panel_heights else ph_default))

        # Grid position
        col = idx % cols
        row = idx // cols
        off_x = col * pw
        off_y = row * ph

        # Iterate over mxCell elements (skip id=0 and id=1)
        root_elem = root.find("root")
        if root_elem is None:
            continue

        for cell in root_elem:
            cid = cell.get("id", "")
            if cid in ("0", "1"):
                continue

            # Make ID unique across panels
            cell.set("id", f"p{idx+1}_{cid}")

            # Fix parent references (except parent="0" and parent="1")
            parent = cell.get("parent", "1")
            if parent not in ("0", "1"):
                cell.set("parent", f"p{idx+1}_{parent}")

            # Offset geometry for vertices
            geom = cell.find("mxGeometry")
            if geom is not None and cell.get("vertex") == "1":
                gx = float(geom.get("x", "0"))
                gy = float(geom.get("y", "0"))
                geom.set("x", str(gx + off_x))
                geom.set("y", str(gy + off_y))

            # Offset source/target points for edges
            if geom is not None and cell.get("edge") == "1":
                for pt_tag in ("sourcePoint", "targetPoint"):
                    # mxPoint stored as child with as="sourcePoint"/"targetPoint"
                    pass
                for child in geom:
                    as_attr = child.get("as", "")
                    if as_attr in ("sourcePoint", "targetPoint"):
                        px = float(child.get("x", "0"))
                        py = float(child.get("y", "0"))
                        child.set("x", str(px + off_x))
                        child.set("y", str(py + off_y))
                    elif as_attr == "points":
                        # Array of waypoints
                        for mxp in child:
                            px = float(mxp.get("x", "0"))
                            py = float(mxp.get("y", "0"))
                            mxp.set("x", str(px + off_x))
                            mxp.set("y", str(py + off_y))

            all_cells.append(cell)

    # Build combined mxGraphModel
    total_w = cols * pw_default
    total_h = rows * ph_default
    if panel_widths:
        total_w = cols * max(panel_widths)
    if panel_heights:
        total_h = rows * max(panel_heights)

    combined_root = ET.Element("mxGraphModel", {
        "dx": "0", "dy": "0", "grid": "0", "gridSize": "1",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(int(total_w)), "pageHeight": str(int(total_h)),
        "math": "0", "shadow": "0",
    })
    root_el = ET.SubElement(combined_root, "root")
    ET.SubElement(root_el, "mxCell", {"id": "0"})
    ET.SubElement(root_el, "mxCell", {"id": "1", "parent": "0"})

    for cell in all_cells:
        root_el.append(cell)

    xml_str = ET.tostring(combined_root, encoding="unicode")

    uid = uuid.uuid4().hex[:20]
    return (
        '<mxfile host="app.diagrams.net" type="device">\n'
        f'  <diagram name="Figure" id="{uid}">\n'
        f'    {xml_str}\n'
        f'  </diagram>\n'
        '</mxfile>'
    )


# ============================================================================
# draw.io CLI Rendering & SSIM
# ============================================================================

def drawio_to_png(
    drawio_path: str,
    output_path: str,
    page: int = 1,
    scale: float = 2.0,
    border: int = 10,
) -> Tuple[Optional[str], Optional[str]]:
    """Render a .drawio file to PNG using the draw.io CLI.

    Returns:
        (output_png_path, error_string_or_None)
    """
    if not os.path.exists(DRAWIO_CLI):
        return (None, f"draw.io CLI not found at {DRAWIO_CLI}")

    cmd = [
        DRAWIO_CLI,
        "-x",           # export mode
        "-f", "png",    # format
        "-b", str(border),
        "-s", str(scale),
        "-p", str(page - 1),  # 0-indexed in CLI
        "-o", output_path,
        drawio_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return (None, f"draw.io CLI error: {result.stderr}")
        if os.path.exists(output_path):
            return (output_path, None)
        return (None, "draw.io CLI produced no output file")
    except subprocess.TimeoutExpired:
        return (None, "draw.io CLI timed out")
    except Exception as e:
        return (None, f"draw.io CLI failed: {e}")


def compute_ssim(
    reference_path: str,
    rendered_path: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute SSIM between reference and rendered images.

    Returns:
        (ssim_score, error_string_or_None)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return (None, "scikit-image not installed: pip install scikit-image")

    ref = cv2.imread(reference_path)
    ren = cv2.imread(rendered_path)

    if ref is None:
        return (None, f"Cannot read reference: {reference_path}")
    if ren is None:
        return (None, f"Cannot read rendered: {rendered_path}")

    # Resize reference to match rendered dimensions
    ref_resized = cv2.resize(ref, (ren.shape[1], ren.shape[0]))
    gray_ref = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
    gray_ren = cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY)

    score = ssim(gray_ref, gray_ren)
    return (score, None)


# ============================================================================
# Step 5: SSIM-Guided Optimization Loop
# ============================================================================

def optimize_drawio_with_llm(
    figure_path: str,
    samed_path: str,
    drawio_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    base_url: str = None,
    provider: ProviderType = "gemini",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    target_ssim: float = DEFAULT_TARGET_SSIM,
    no_icon_mode: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Optimize draw.io output using SVG-space SSIM-guided refinement.

    Strategy: Optimize the intermediate SVG (where LLM excels), then
    re-convert to draw.io XML after each iteration.

    Each iteration:
    1. Render current SVG → PNG (via cairosvg)
    2. Compute SSIM vs reference
    3. Send images + SVG code to LLM for refinement
    4. Convert improved SVG → draw.io XML
    5. Early exit if SSIM >= target

    Returns:
        (optimized_drawio_path, error_string_or_None)
    """
    from svg_to_drawio import svg_to_drawio

    print("\n" + "=" * 60)
    print("Step 5: SVG-Space SSIM-Guided Optimization")
    print("=" * 60)
    print(f"Target SSIM: {target_ssim}")
    print(f"Max iterations: {max_iterations}")

    if base_url is None:
        base_url = PROVIDER_CONFIGS.get(provider, {}).get("base_url", "")

    if max_iterations == 0:
        shutil.copy(drawio_path, output_path)
        print("  Iterations = 0, skipping optimization")
        return (output_path, None)

    # Find the intermediate SVG (saved by generate_drawio_template)
    svg_path = drawio_path.replace(".drawio", ".svg")
    if not os.path.exists(svg_path):
        # No SVG available, fall back to copying draw.io as-is
        print(f"  No intermediate SVG found at {svg_path}, skipping optimization")
        shutil.copy(drawio_path, output_path)
        return (output_path, None)

    with open(svg_path, "r", encoding="utf-8") as f:
        current_svg = f.read()

    output_dir = os.path.dirname(drawio_path)
    best_ssim = 0.0
    best_svg = current_svg

    try:
        import cairosvg
        has_cairosvg = True
    except (ImportError, OSError):
        has_cairosvg = False

    for iteration in range(max_iterations):
        print(f"\n  Iteration {iteration + 1}/{max_iterations}")
        print("  " + "-" * 50)

        # Render SVG → PNG
        iter_svg = os.path.join(output_dir, f"temp_iter_{iteration}.svg")
        iter_png = os.path.join(output_dir, f"temp_iter_{iteration}.png")

        with open(iter_svg, "w", encoding="utf-8") as f:
            f.write(current_svg)

        render_ok = False
        if has_cairosvg:
            try:
                cairosvg.svg2png(
                    url=iter_svg, write_to=iter_png,
                    output_width=1376, output_height=768,
                )
                render_ok = True
            except Exception as e:
                print(f"  CairoSVG render failed: {e}")

        if not render_ok:
            # Fallback: convert SVG → draw.io → PNG via draw.io CLI
            temp_drawio = os.path.join(output_dir, f"temp_iter_{iteration}.drawio")
            svg_to_drawio(iter_svg, temp_drawio)
            _, render_err = drawio_to_png(temp_drawio, iter_png)
            if render_err:
                print(f"  Render failed: {render_err}")
                continue

        # Compute SSIM
        ssim_score, ssim_err = compute_ssim(figure_path, iter_png)
        if ssim_err:
            print(f"  SSIM computation failed: {ssim_err}")
            ssim_score = 0.0
        else:
            print(f"  SSIM: {ssim_score:.4f}")

        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_svg = current_svg

        if ssim_score >= target_ssim:
            print(f"  Target SSIM {target_ssim} reached! (actual: {ssim_score:.4f})")
            break

        # Prepare optimization prompt (LLM optimizes SVG, not draw.io XML)
        figure_img = Image.open(figure_path)
        samed_img = Image.open(samed_path)
        rendered_img = Image.open(iter_png)

        placeholder_note = ""
        if not no_icon_mode:
            placeholder_note = """- Keep all icon placeholder <g> elements intact (id="AF01", "AF02", etc.)
- Placeholders must keep fill="#808080" stroke="black" style"""

        prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and improve the SVG code.

Current SSIM score: {ssim_score:.4f} (target: {target_ssim})

I provide 3 images:
1. Image 1 (original figure): The target we want to reproduce
2. Image 2 (SAM reference): Icon positions marked with gray rectangles
3. Image 3 (current rendering): Current SVG rendered as PNG

Carefully compare Image 1 and Image 3. Fix these aspects:
- Position: Move elements to match exact positions in original
- Size: Adjust widths/heights of boxes and elements
- Colors: Match fill, stroke colors exactly
- Text: Fix font sizes, positions, text content
- Arrows/paths: Fix routing, start/end points, marker styles
- Layout: Fix spacing, alignment
{placeholder_note}

CURRENT SVG CODE:
```svg
{current_svg}
```

IMPORTANT:
- Output ONLY the improved SVG code (starting with <svg, ending with </svg>)
- Keep the same viewBox and width/height
- No markdown formatting
- Preserve all element structure"""

        contents = [prompt, figure_img, samed_img, rendered_img]

        try:
            print("  Sending SVG optimization request...")
            response = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=50000,
                temperature=0.3,
            )

            if not response:
                print("  Empty response")
                continue

            optimized_svg = _extract_svg_code(response)
            if not optimized_svg:
                print("  Could not extract SVG from response")
                continue

            ok, errors = _validate_svg_syntax(optimized_svg)
            if not ok:
                print(f"  SVG has errors, attempting fix...")
                optimized_svg, fix_err = _fix_svg_with_llm(
                    optimized_svg, errors, api_key, model, base_url, provider,
                )

            current_svg = optimized_svg
            print("  SVG optimization iteration complete")

        except Exception as e:
            print(f"  Optimization error: {e}")
            continue

        # Cleanup temp files
        for tmp in [iter_svg, iter_png]:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    # Save optimized SVG
    opt_svg_path = os.path.join(output_dir, "optimized_template.svg")
    with open(opt_svg_path, "w", encoding="utf-8") as f:
        f.write(best_svg if best_ssim > 0 else current_svg)
    print(f"  Optimized SVG saved: {opt_svg_path}")

    # Convert optimized SVG → draw.io
    drawio_path_out, conv_err = svg_to_drawio(opt_svg_path, output_path)
    if conv_err:
        print(f"  SVG→draw.io conversion failed: {conv_err}")
        shutil.copy(drawio_path, output_path)

    # Final render for preview
    final_png = output_path.replace(".drawio", ".png")
    drawio_to_png(output_path, final_png)

    print(f"\n  Best SSIM: {best_ssim:.4f}")
    print(f"  Optimized draw.io saved: {output_path}")
    return (output_path, None)


# ============================================================================
# Step 6: Icon Replacement in draw.io XML
# ============================================================================

def _find_nearest_gray_cell(xml_content: str, target_x: float, target_y: float,
                            target_w: float, target_h: float,
                            used_ids: set) -> Optional[re.Match]:
    """Find the mxCell with gray fill closest to (target_x, target_y).

    Gray placeholders use fillColor=#808080 or similar gray tones.
    Returns the regex match object for the closest cell, or None.
    """
    # Match vertex mxCells with geometry (both self-closing and with children)
    cell_pat = re.compile(
        r'(<mxCell\s[^>]*vertex="1"[^>]*>[\s\S]*?</mxCell>|'
        r'<mxCell\s[^>]*vertex="1"[^/]*/\s*>)',
        re.IGNORECASE
    )
    geom_pat = re.compile(
        r'<mxGeometry\s+x="([\d.]+)"\s+y="([\d.]+)"\s+'
        r'width="([\d.]+)"\s+height="([\d.]+)"'
    )
    id_pat = re.compile(r'id="([^"]*)"')

    best_match = None
    best_dist = float("inf")

    for m in cell_pat.finditer(xml_content):
        cell_xml = m.group(0)
        id_m = id_pat.search(cell_xml)
        if not id_m or id_m.group(1) in used_ids:
            continue
        cid = id_m.group(1)
        if cid in ("0", "1"):
            continue

        # Check if this is a gray placeholder (fillColor=#808080 or similar)
        style = re.search(r'style="([^"]*)"', cell_xml)
        if not style:
            continue
        s = style.group(1).lower()
        is_gray = "fillcolor=#808080" in s or "fillcolor=#8" in s or "fillcolor=#7" in s
        if not is_gray:
            continue

        gm = geom_pat.search(cell_xml)
        if not gm:
            continue
        cx, cy = float(gm.group(1)), float(gm.group(2))
        # Distance based on center points
        dist = abs(cx - target_x) + abs(cy - target_y)
        if dist < best_dist:
            best_dist = dist
            best_match = (m, cid)

    return best_match


def replace_icons_in_drawio(
    template_path: str,
    icon_infos: List[Dict[str, Any]],
    output_path: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Replace placeholder mxCells with base64 PNG icon images.

    Matching strategy (in priority order):
    1. By cell ID matching AF label (e.g. id="AF01")
    2. By nearest gray-filled cell to the icon's target position
    3. Append as new cell at the SAM3-detected position

    Returns:
        (output_path, error_string_or_None)
    """
    print("\n" + "=" * 60)
    print("Step 6: Icon Replacement in draw.io XML")
    print("=" * 60)

    with open(template_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    replaced_count = 0
    used_cell_ids = set()  # track which cells we've already replaced

    for info in icon_infos:
        icon_path = info.get("icon_path") or info.get("nobg_path")
        if not icon_path or not os.path.exists(icon_path):
            label = info.get("label", "unknown")
            print(f"  {label}: no icon file, skipping")
            continue

        label_clean = info.get("label_clean", "")
        if not label_clean:
            continue

        # Read icon and encode as base64
        with open(icon_path, "rb") as f:
            icon_b64 = base64.b64encode(f.read()).decode("utf-8")

        target_x = float(info.get("x1", 0))
        target_y = float(info.get("y1", 0))
        target_w = float(info.get("width", 60))
        target_h = float(info.get("height", 60))

        replaced = False

        # Strategy 1: ID match
        placeholder_pattern = rf'(<mxCell[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^>]*>[\s\S]*?</mxCell>|<mxCell[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^/]*/\s*>)'
        match = re.search(placeholder_pattern, xml_content, re.IGNORECASE)

        if match:
            old_cell = match.group(0)
            geom_match = re.search(
                r'<mxGeometry\s+x="([\d.]+)"\s+y="([\d.]+)"\s+width="([\d.]+)"\s+height="([\d.]+)"',
                old_cell
            )
            if geom_match:
                x, y, w, h = geom_match.groups()
            else:
                x, y, w, h = target_x, target_y, target_w, target_h

            new_cell = f'<mxCell id="{label_clean}" value="" style="shape=image;imageAspect=0;perimeter=none;image=data:image/png,{icon_b64}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
            xml_content = xml_content.replace(old_cell, new_cell)
            used_cell_ids.add(label_clean)
            print(f"  {label_clean}: replaced (id match)")
            replaced = True

        # Strategy 2: nearest gray cell by position
        if not replaced:
            result = _find_nearest_gray_cell(
                xml_content, target_x, target_y, target_w, target_h, used_cell_ids
            )
            if result:
                m, cid = result
                old_cell = m.group(0)
                geom_match = re.search(
                    r'<mxGeometry\s+x="([\d.]+)"\s+y="([\d.]+)"\s+width="([\d.]+)"\s+height="([\d.]+)"',
                    old_cell
                )
                if geom_match:
                    x, y, w, h = geom_match.groups()
                else:
                    x, y, w, h = target_x, target_y, target_w, target_h

                new_cell = f'<mxCell id="{label_clean}" value="" style="shape=image;imageAspect=0;perimeter=none;image=data:image/png,{icon_b64}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
                xml_content = xml_content.replace(old_cell, new_cell)
                used_cell_ids.add(cid)
                print(f"  {label_clean}: replaced gray cell {cid} (position match)")
                replaced = True

        # Strategy 3: append at SAM3 position
        if not replaced:
            new_cell = f'<mxCell id="icon_{label_clean}" value="" style="shape=image;imageAspect=0;perimeter=none;image=data:image/png,{icon_b64}" vertex="1" parent="1"><mxGeometry x="{target_x}" y="{target_y}" width="{target_w}" height="{target_h}" as="geometry"/></mxCell>'
            xml_content = xml_content.replace("</root>", f"    {new_cell}\n  </root>")
            print(f"  {label_clean}: appended at ({target_x},{target_y})")

        replaced_count += 1

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"  Total: {replaced_count}/{len(icon_infos)} icons processed")
    print(f"  Final draw.io saved: {output_path}")
    return (output_path, None)


# ============================================================================
# Main Orchestrator
# ============================================================================

def image_to_drawio(
    image_path: str,
    output_dir: str = "./output",
    api_key: str = None,
    base_url: str = None,
    provider: ProviderType = "gemini",
    model: str = None,
    grid: str = "auto",
    sam_prompts: str = "icon,symbol,logo,diagram,image,picture,chart,graph",
    min_score: float = 0.3,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "roboflow",
    sam_api_key: str = None,
    sam_max_masks: int = 64,
    merge_threshold: float = 0.9,
    target_ssim: float = DEFAULT_TARGET_SSIM,
    optimize_iterations: int = DEFAULT_MAX_ITERATIONS,
    stop_after: int = 7,
    skip_icons: bool = False,
) -> Tuple[Optional[Dict], Optional[str]]:
    """Main pipeline: PNG image -> editable .drawio file.

    Steps:
      1. Split grid (if multi-figure)
      2. SAM3 detection per panel
      3. Gemini icon regeneration
      4. draw.io template generation
      5. SSIM-guided optimization
      6. Icon replacement
      7. Multi-page assembly

    Returns:
        (result_dict, error_string_or_None)
    """
    if not api_key:
        return (None, "api_key is required")

    config = PROVIDER_CONFIGS.get(provider, {})
    if base_url is None:
        base_url = config.get("base_url", "")
    if model is None:
        model = DEFAULT_GEMINI_MODEL

    if sam_api_key is None:
        sam_api_key = os.environ.get("ROBOFLOW_API_KEY", "XP6i1kouBfkRylSQ12Kw")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PNG -> Editable draw.io XML Pipeline")
    print("=" * 60)
    print(f"Input: {image_path}")
    print(f"Output: {output_dir}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Grid: {grid}")
    print(f"Target SSIM: {target_ssim}")
    print(f"Optimize iterations: {optimize_iterations}")
    print("=" * 60)

    # ---- Step 1: Split grid ----
    print("\n[Step 1] Splitting image grid...")
    panels_dir = str(output_dir / "panels")
    panels, split_err = split_image_grid(image_path, panels_dir, grid)
    if split_err:
        return (None, f"Grid split failed: {split_err}")
    print(f"  {len(panels)} panel(s)")

    if stop_after == 1:
        return ({"panels": panels}, None)

    # ---- Process each panel ----
    page_xmls = []
    page_names = []
    all_results = []

    for panel_idx, panel_path in enumerate(panels):
        panel_name = f"Figure {panel_idx + 1}"
        panel_dir = str(output_dir / f"panel_{panel_idx + 1}")
        os.makedirs(panel_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing {panel_name}: {panel_path}")
        print(f"{'=' * 60}")

        # ---- Step 2: Icon detection (SAM3 + Gemini Vision) ----
        print(f"\n[Step 2] Icon detection for {panel_name}...")

        # 2a: SAM3 detection
        print(f"  [2a] SAM3 detection...")
        try:
            samed_path, boxlib_path, valid_boxes = segment_with_sam3(
                image_path=panel_path,
                output_dir=panel_dir,
                text_prompts=sam_prompts,
                min_score=min_score,
                merge_threshold=merge_threshold,
                sam_backend=sam_backend,
                sam_api_key=sam_api_key,
                sam_max_masks=sam_max_masks,
            )
        except Exception as e:
            print(f"  SAM3 failed: {e}")
            samed_path = panel_path
            boxlib_path = None
            valid_boxes = []
        print(f"  SAM3 found {len(valid_boxes)} icons")

        # 2b: Gemini Vision supplementary detection
        print(f"  [2b] Gemini Vision detection...")
        gemini_boxes, gemini_err = detect_icons_with_gemini(
            image_path=panel_path,
            api_key=api_key,
        )
        if gemini_err:
            print(f"  Gemini detection error: {gemini_err}")
        else:
            print(f"  Gemini found {len(gemini_boxes)} icons")

        # 2c: Merge SAM3 + Gemini detections
        if gemini_boxes:
            before = len(valid_boxes)
            valid_boxes = merge_sam3_and_gemini_boxes(valid_boxes, gemini_boxes)
            print(f"  Merged: SAM3({before}) + Gemini({len(gemini_boxes)}) -> {len(valid_boxes)} total")

            # Re-draw samed.png with merged boxes
            img = Image.open(panel_path)
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            for box in valid_boxes:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(10, min(20, (x2 - x1) // 4)))
                except (OSError, IOError):
                    font = ImageFont.load_default()
                try:
                    draw.text((cx, cy), box["label"], fill="white", anchor="mm", font=font)
                except TypeError:
                    draw.text((cx - 20, cy - 8), box["label"], fill="white", font=font)
            samed_path = os.path.join(panel_dir, "samed.png")
            img.save(samed_path)

            # Update boxlib.json
            w_img, h_img = Image.open(panel_path).size
            boxlib_data = {
                "image_size": {"width": w_img, "height": h_img},
                "prompts_used": [p.strip() for p in sam_prompts.split(",") if p.strip()] + ["gemini_vision"],
                "boxes": valid_boxes,
            }
            boxlib_path = os.path.join(panel_dir, "boxlib.json")
            with open(boxlib_path, "w", encoding="utf-8") as f:
                json.dump(boxlib_data, f, indent=2, ensure_ascii=False)

        no_icon_mode = len(valid_boxes) == 0 or skip_icons
        print(f"  Final: {len(valid_boxes)} icons (no_icon_mode={no_icon_mode})")

        if stop_after == 2:
            all_results.append({
                "panel": panel_name,
                "figure_path": panel_path,
                "samed_path": samed_path,
                "boxlib_path": boxlib_path,
            })
            continue

        # ---- Step 3: Crop-first icon pipeline ----
        icon_infos = []
        if not no_icon_mode:
            print(f"\n[Step 3] Icon pipeline for {panel_name}...")
            icons_dir = os.path.join(panel_dir, "icons")

            # 3a: Crop icons from original at detected positions
            crops = crop_icons_from_original(panel_path, valid_boxes, icons_dir)

            # 3b: Real-ESRGAN 4x upscale
            esrgan_model, esrgan_dev, esrgan_err = _get_esrgan_model()
            if esrgan_err:
                print(f"  Real-ESRGAN skipped: {esrgan_err}")
            else:
                print(f"  Upscaling {len(crops)} icons with Real-ESRGAN (4x)...")
                for crop in crops:
                    src = crop["crop_path"]
                    hd_path = src.rsplit(".", 1)[0] + "_hd.png"
                    try:
                        upscale_icon_esrgan(src, hd_path)
                        crop["hd_path"] = hd_path
                    except Exception as e:
                        print(f"    Upscale failed for {os.path.basename(src)}: {e}")
                hd_ok = sum(1 for c in crops if c.get("hd_path"))
                print(f"  Real-ESRGAN: {hd_ok}/{len(crops)} icons upscaled")

            # 3c: RMBG-2.0 background removal (on HD version if available)
            remover, rmbg_err = _get_rmbg_remover(icons_dir)
            if rmbg_err:
                print(f"  RMBG-2.0 skipped: {rmbg_err}")
                for crop in crops:
                    crop["icon_path"] = crop.get("hd_path") or crop["crop_path"]
            else:
                print(f"  Removing backgrounds with RMBG-2.0...")
                for crop in crops:
                    src = crop.get("hd_path") or crop["crop_path"]
                    nobg_path = src.rsplit(".", 1)[0] + "_nobg.png"
                    try:
                        remover.remove_background(src, nobg_path)
                        crop["icon_path"] = nobg_path
                    except Exception as e:
                        print(f"    RMBG failed for {os.path.basename(src)}: {e}")
                        crop["icon_path"] = src
                rmbg_ok = sum(1 for c in crops if c["icon_path"].endswith("_nobg.png"))
                print(f"  RMBG-2.0: {rmbg_ok}/{len(crops)} icons processed")

            icon_infos = crops

        if stop_after == 3:
            all_results.append({
                "panel": panel_name,
                "figure_path": panel_path,
                "samed_path": samed_path,
                "icon_infos": icon_infos,
            })
            continue

        # ---- Step 4: draw.io template generation ----
        print(f"\n[Step 4] Generating draw.io template for {panel_name}...")
        template_path = os.path.join(panel_dir, "template.drawio")
        _, template_err = generate_drawio_template(
            figure_path=panel_path,
            samed_path=samed_path,
            boxlib_path=boxlib_path or "",
            output_path=template_path,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            no_icon_mode=no_icon_mode,
        )

        if template_err:
            print(f"  Template generation failed: {template_err}")
            all_results.append({"panel": panel_name, "error": template_err})
            continue

        if stop_after == 4:
            all_results.append({
                "panel": panel_name,
                "template_path": template_path,
            })
            continue

        # ---- Step 5: SSIM-guided optimization ----
        print(f"\n[Step 5] Optimizing {panel_name}...")
        optimized_path = os.path.join(panel_dir, "optimized.drawio")
        _, opt_err = optimize_drawio_with_llm(
            figure_path=panel_path,
            samed_path=samed_path,
            drawio_path=template_path,
            output_path=optimized_path,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            max_iterations=optimize_iterations,
            target_ssim=target_ssim,
            no_icon_mode=no_icon_mode,
        )

        working_path = optimized_path if os.path.exists(optimized_path) else template_path

        if stop_after == 5:
            all_results.append({
                "panel": panel_name,
                "optimized_path": working_path,
            })
            continue

        # ---- Step 6: Icon replacement ----
        final_panel_path = os.path.join(panel_dir, "final.drawio")
        if no_icon_mode or not icon_infos:
            shutil.copy(working_path, final_panel_path)
            print(f"  No icons to replace for {panel_name}")
        else:
            print(f"\n[Step 6] Replacing icons for {panel_name}...")
            replace_icons_in_drawio(
                template_path=working_path,
                icon_infos=icon_infos,
                output_path=final_panel_path,
            )

        # Read final XML for multi-page assembly
        with open(final_panel_path, "r", encoding="utf-8") as f:
            page_xml = f.read()
        page_xmls.append(page_xml)
        page_names.append(panel_name)

        all_results.append({
            "panel": panel_name,
            "figure_path": panel_path,
            "samed_path": samed_path,
            "template_path": template_path,
            "optimized_path": optimized_path if os.path.exists(optimized_path) else None,
            "final_path": final_panel_path,
            "icon_count": len(icon_infos),
        })

    if stop_after <= 6 or not page_xmls:
        return ({"panels": all_results}, None)

    # ---- Step 7: Assemble .drawio ----
    print("\n" + "=" * 60)
    print("Step 7: Assembling .drawio file")
    print("=" * 60)

    final_drawio_path = str(output_dir / "final.drawio")
    if len(page_xmls) == 1:
        shutil.copy(all_results[0]["final_path"], final_drawio_path)
    else:
        # Merge all panels into a single page with grid layout
        combined = combine_pages_single(page_xmls, grid=grid)
        with open(final_drawio_path, "w", encoding="utf-8") as f:
            f.write(combined)

    print(f"Final draw.io file: {final_drawio_path}")

    # Final SSIM check
    final_png = str(output_dir / "final_rendered.png")
    png_path, _ = drawio_to_png(final_drawio_path, final_png, page=1)
    if png_path:
        ssim_score, _ = compute_ssim(panels[0], png_path)
        if ssim_score is not None:
            print(f"Final SSIM (page 1): {ssim_score:.4f}")

    result = {
        "final_drawio_path": final_drawio_path,
        "panels": all_results,
        "page_count": len(page_xmls),
    }
    return (result, None)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert PNG figure to editable draw.io XML"
    )
    parser.add_argument(
        "--input_image", required=True,
        help="Path to input PNG image"
    )
    parser.add_argument(
        "--output_dir", default="./output",
        help="Output directory"
    )
    parser.add_argument(
        "--provider", default="gemini",
        choices=["gemini", "openrouter", "bianxie"],
        help="LLM API provider"
    )
    parser.add_argument(
        "--api_key", default=None,
        help="API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_GEMINI_MODEL,
        help="LLM model name"
    )
    parser.add_argument(
        "--grid", default="auto",
        help="Grid layout: auto, 1x1, 2x2, 1x2, 2x1"
    )
    parser.add_argument(
        "--sam_prompts", default="icon,symbol,logo,diagram,image,picture,chart,graph",
        help="SAM3 text prompts (comma-separated)"
    )
    parser.add_argument(
        "--sam_backend", default="roboflow",
        choices=["local", "fal", "roboflow", "api"],
        help="SAM3 backend"
    )
    parser.add_argument(
        "--sam_api_key", default=None,
        help="SAM3 API key (Roboflow)"
    )
    parser.add_argument(
        "--target_ssim", type=float, default=DEFAULT_TARGET_SSIM,
        help="Target SSIM score for optimization"
    )
    parser.add_argument(
        "--optimize_iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
        help="Max optimization iterations (0 to skip)"
    )
    parser.add_argument(
        "--merge_threshold", type=float, default=0.9,
        help="SAM3 box merge threshold"
    )
    parser.add_argument(
        "--min_score", type=float, default=0.3,
        help="SAM3 minimum confidence score"
    )
    parser.add_argument(
        "--sam_max_masks", type=int, default=64,
        help="SAM3 max masks per prompt"
    )
    parser.add_argument(
        "--stop_after", type=int, default=7,
        help="Stop after step N (1-7)"
    )
    parser.add_argument(
        "--skip_icons", action="store_true",
        help="Skip icon regeneration (structure only)"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: --api_key or GEMINI_API_KEY env var required")
        sys.exit(1)

    result, err = image_to_drawio(
        image_path=args.input_image,
        output_dir=args.output_dir,
        api_key=api_key,
        provider=args.provider,
        model=args.model,
        grid=args.grid,
        sam_prompts=args.sam_prompts,
        sam_backend=args.sam_backend,
        sam_api_key=args.sam_api_key,
        target_ssim=args.target_ssim,
        optimize_iterations=args.optimize_iterations,
        merge_threshold=args.merge_threshold,
        min_score=args.min_score,
        sam_max_masks=args.sam_max_masks,
        stop_after=args.stop_after,
        skip_icons=args.skip_icons,
    )

    if err:
        print(f"\nERROR: {err}")
        sys.exit(1)

    print("\nPipeline complete!")
    if result and "final_drawio_path" in result:
        print(f"Output: {result['final_drawio_path']}")


if __name__ == "__main__":
    main()
