"""Icon extraction: auto-detect bounding boxes from icon sheets, extract RGBA PNGs.

Usage (Python):
    from icon_extractor import extract_icons
    icons, err = extract_icons("input.png", "output_dir/")

Usage (CLI):
    python icon_extractor.py input.png [output_dir]
"""

import os
import re
import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypeVar

import cv2
import numpy as np
from PIL import Image

try:
    import vtracer
    HAS_VTRACER = True
except ImportError:
    HAS_VTRACER = False

# ---------------------------------------------------------------------------
# Inline error helpers (replaces figtool.errors)
# ---------------------------------------------------------------------------
T = TypeVar("T")
Result = Tuple[Optional[T], Optional[str]]


def ok(value):
    return (value, None)


def fail(msg):
    return (None, msg)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class IconInfo:
    """Extracted icon metadata."""
    name: str
    x: int
    y: int
    w: int
    h: int
    png_path: str = ""
    svg_path: str = ""
    thumbnail_b64: str = ""


@dataclass
class BBox:
    """Bounding box for an icon region."""
    x: int
    y: int
    w: int
    h: int
    name: str = ""


UPSCALE = 3


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------
def _detect_icon_grid(gray: np.ndarray, min_cells: int = 4) -> Optional[List[BBox]]:
    """Detect if image is a regular grid of icons. Returns grid cells if found."""
    h, w = gray.shape

    content = (255 - gray).astype(float)
    content[content < 30] = 0

    row_proj = np.sum(content, axis=1)
    col_proj = np.sum(content, axis=0)

    ks = max(3, min(h, w) // 100)
    if ks % 2 == 0:
        ks += 1
    row_smooth = np.convolve(row_proj, np.ones(ks) / ks, mode='same')
    col_smooth = np.convolve(col_proj, np.ones(ks) / ks, mode='same')

    def find_splits(proj, length):
        best = []
        for gap_thresh in [0.05, 0.10, 0.15, 0.20]:
            threshold = np.max(proj) * gap_thresh
            is_gap = proj < threshold
            raw_gaps = []
            in_gap = False
            gap_start = 0
            for i in range(length):
                if is_gap[i] and not in_gap:
                    gap_start = i
                    in_gap = True
                elif not is_gap[i] and in_gap:
                    raw_gaps.append((gap_start, i))
                    in_gap = False
            merge_dist = length * 0.08
            merged = []
            for s, e in raw_gaps:
                if merged and s - merged[-1][1] < merge_dist:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            splits = [(s + e) // 2 for s, e in merged if e - s >= 3]
            if len(splits) > len(best):
                best = splits
        return best

    row_splits = find_splits(row_smooth, h)
    col_splits = find_splits(col_smooth, w)

    if len(col_splits) >= 2 and len(row_splits) < 1:
        col_bounds_tmp = [0] + col_splits + [w]
        avg_cell_w = (col_bounds_tmp[-1] - col_bounds_tmp[0]) / len(col_splits)
        expected_rows = max(1, round(h / avg_cell_w))
        if expected_rows >= 2:
            row_step = h / expected_rows
            row_splits = [int(row_step * i) for i in range(1, expected_rows)]

    if len(row_splits) < 1 or len(col_splits) < 1:
        return None

    def build_bounds(splits, total):
        bounds = [0] + splits + [total]
        min_size = total / (len(splits) + 1) * 0.3
        while len(bounds) > 2 and bounds[1] - bounds[0] < min_size:
            bounds = bounds[1:]
        while len(bounds) > 2 and bounds[-1] - bounds[-2] < min_size:
            bounds = bounds[:-1]
        return bounds

    row_bounds = build_bounds(row_splits, h)
    col_bounds = build_bounds(col_splits, w)
    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1

    if n_rows * n_cols < min_cells:
        return None

    row_sizes = [row_bounds[i + 1] - row_bounds[i] for i in range(n_rows)]
    col_sizes = [col_bounds[i + 1] - col_bounds[i] for i in range(n_cols)]
    if max(row_sizes) > 2.5 * min(row_sizes) or max(col_sizes) > 2.5 * min(col_sizes):
        return None

    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            x = col_bounds[c]
            y = row_bounds[r]
            cw = col_bounds[c + 1] - x
            ch = row_bounds[r + 1] - y
            cell_content = content[y:y + ch, x:x + cw]
            if np.sum(cell_content > 0) > 100:
                cells.append(BBox(x=x, y=y, w=cw, h=ch))

    return cells if len(cells) >= min_cells else None


# ---------------------------------------------------------------------------
# Bounding box detection
# ---------------------------------------------------------------------------
def auto_detect_boxes(
    image_path: str,
    min_area: int = 2500,
    min_dim: int = 30,
    dilation_kernel: int = 10,
    dilation_iters: int = 2,
    white_thresh: int = 230,
) -> Result[List[BBox]]:
    """Auto-detect icon bounding boxes from an image with white background."""
    img = cv2.imread(image_path)
    if img is None:
        return fail(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grid_boxes = _detect_icon_grid(gray)
    if grid_boxes is not None:
        for i, b in enumerate(grid_boxes):
            b.name = f"icon_{i+1:02d}"
        return ok(grid_boxes)

    _, thresh = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=dilation_iters)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area and w >= min_dim and h >= min_dim:
            boxes.append(BBox(x=x, y=y, w=w, h=h))

    if boxes:
        avg_h = sum(b.h for b in boxes) / len(boxes)
        row_tolerance = avg_h * 0.5
        boxes.sort(key=lambda b: (round(b.y / row_tolerance), b.x))

    for i, b in enumerate(boxes):
        b.name = f"icon_{i+1:02d}"

    return ok(boxes)


# ---------------------------------------------------------------------------
# Icon extraction
# ---------------------------------------------------------------------------
def extract_icon_rgba(
    img_bgr: np.ndarray,
    x: int, y: int, w: int, h: int,
    padding: int = 6,
    bg_mode: str = "white",
) -> Optional[Image.Image]:
    """Extract an icon region as RGBA with transparent background."""
    ih, iw = img_bgr.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(iw, x + w + padding)
    y2 = min(ih, y + h + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    roi = img_bgr[y1:y2, x1:x2].copy()
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb).convert("RGBA")
    data = np.array(pil_img)

    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    if bg_mode == "auto":
        corners = [data[0, 0, :3], data[0, -1, :3], data[-1, 0, :3], data[-1, -1, :3]]
        avg_corner = np.mean(corners, axis=0)
        if avg_corner[1] > avg_corner[0] and avg_corner[1] > avg_corner[2] and avg_corner[1] > 200:
            bg_mode = "green"
        else:
            bg_mode = "white"

    if bg_mode == "green":
        bg_mask = (r > 200) & (g > 210) & (b > 190) & (g >= r) & (g >= b)
    else:
        bg_mask = (r > 240) & (g > 240) & (b > 240)

    data[:, :, 3] = np.where(bg_mask, 0, 255)

    if bg_mode == "white":
        near_bg = (r > 215) & (g > 215) & (b > 215) & ~bg_mask
        avg = (r.astype(int) + g.astype(int) + b.astype(int)) / 3
        alpha = np.clip((240 - avg) * (255 / 25), 0, 255).astype(np.uint8)
        data[:, :, 3] = np.where(near_bg, alpha, data[:, :, 3])

    return Image.fromarray(data)


def trim_transparent(img: Image.Image) -> Image.Image:
    """Trim transparent borders."""
    data = np.array(img)
    alpha = data[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - 1)
    cmin = max(0, cmin - 1)
    rmax = min(data.shape[0] - 1, rmax + 1)
    cmax = min(data.shape[1] - 1, cmax + 1)
    return Image.fromarray(data[rmin:rmax + 1, cmin:cmax + 1])


# ---------------------------------------------------------------------------
# SVG conversion (optional)
# ---------------------------------------------------------------------------
def convert_to_svg(png_path: str, svg_path: str, upscale: int = UPSCALE) -> Result[str]:
    """Convert PNG to SVG using vtracer."""
    if not HAS_VTRACER:
        return fail("vtracer not installed: pip install vtracer")

    img = Image.open(png_path).convert("RGBA")
    w, h = img.size

    big = img.resize((w * upscale, h * upscale), Image.LANCZOS)
    bg = Image.new("RGBA", big.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(bg, big)
    tmp_png = png_path.replace(".png", "_vtmp.png")
    composite.convert("RGB").save(tmp_png)

    try:
        vtracer.convert_image_to_svg_py(
            tmp_png, svg_path,
            colormode='color',
            hierarchical='stacked',
            mode='spline',
            filter_speckle=4,
            color_precision=6,
            corner_threshold=60,
            splice_threshold=45,
            path_precision=2,
        )
        _fix_svg(svg_path, w, h, upscale)
        return ok(svg_path)
    except Exception as e:
        return fail(str(e))
    finally:
        if os.path.exists(tmp_png):
            os.remove(tmp_png)


def _fix_svg(svg_path: str, orig_w: int, orig_h: int, upscale: int):
    """Fix SVG viewBox and remove white background paths."""
    with open(svg_path) as f:
        content = f.read()

    up_w, up_h = orig_w * upscale, orig_h * upscale
    content = re.sub(r'viewBox="0 0 \d+ \d+"', f'viewBox="0 0 {up_w} {up_h}"', content)
    content = re.sub(r'width="\d+"', f'width="{orig_w}"', content)
    content = re.sub(r'height="\d+"', f'height="{orig_h}"', content)
    content = re.sub(r'<path[^>]*fill="#[fF]{6}"[^>]*/>', '', content)
    content = re.sub(r'<path[^>]*fill="#[fF][eE][fF][eE][fF][eE]"[^>]*/>', '', content)
    content = re.sub(
        r'<path[^>]*fill="#f[a-f][f][a-f][f][a-f]"[^>]*/>', '', content,
        flags=re.IGNORECASE,
    )

    with open(svg_path, 'w') as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------
def make_thumbnail_b64(img: Image.Image, max_size: int = 128) -> str:
    """Create a base64-encoded PNG thumbnail."""
    thumb = img.copy()
    thumb.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------
def extract_icons(
    image_path: str,
    output_dir: str,
    boxes: Optional[List[BBox]] = None,
    bg_mode: str = "auto",
    make_svg: bool = False,
) -> Result[List[IconInfo]]:
    """Extract icons from an image.

    Args:
        image_path: Path to source image
        output_dir: Directory for output PNG files
        boxes: Manual bounding boxes (auto-detect if None)
        bg_mode: Background removal mode ("white", "green", "auto")
        make_svg: Whether to generate SVG via vtracer (default False)

    Returns:
        (list of IconInfo, None) on success, (None, error_string) on failure
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        return fail(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    if boxes is None:
        boxes, err = auto_detect_boxes(image_path)
        if err:
            return fail(err)

    icons = []
    for i, box in enumerate(boxes):
        bx = max(0, min(box.x, w - 1))
        by = max(0, min(box.y, h - 1))
        bw = min(box.w, w - bx)
        bh = min(box.h, h - by)

        name = box.name or f"icon_{i+1:02d}"

        rgba = extract_icon_rgba(img, bx, by, bw, bh, padding=6, bg_mode=bg_mode)
        if rgba is None:
            continue

        rgba = trim_transparent(rgba)

        png_path = os.path.join(output_dir, f"{name}.png")
        rgba.save(png_path)

        thumb_b64 = make_thumbnail_b64(rgba)

        svg_path = ""
        if make_svg:
            svg_out = os.path.join(output_dir, f"{name}.svg")
            result, err = convert_to_svg(png_path, svg_out)
            if result:
                svg_path = svg_out

        icons.append(IconInfo(
            name=name,
            x=bx, y=by, w=rgba.size[0], h=rgba.size[1],
            png_path=png_path,
            svg_path=svg_path,
            thumbnail_b64=thumb_b64,
        ))

    return ok(icons)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python icon_extractor.py <image> [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "icons_output"

    print(f"Input:  {img_path}")
    print(f"Output: {out_dir}")
    print()

    print("Detecting icons...")
    boxes, err = auto_detect_boxes(img_path)
    if err:
        print(f"ERROR: {err}")
        sys.exit(1)
    print(f"Found {len(boxes)} icon regions")

    print("Extracting PNGs...")
    icons, err = extract_icons(img_path, out_dir, boxes, make_svg=False)
    if err:
        print(f"ERROR: {err}")
        sys.exit(1)

    print(f"\nExtracted {len(icons)} icons:")
    for ic in icons:
        print(f"  {ic.name}: {ic.w}x{ic.h}")
