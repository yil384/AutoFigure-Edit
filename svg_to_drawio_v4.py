"""svg_to_drawio_v4.py — PNG+SVG → drawio with:
 1. Claude/Gemini multimodal OCR (precise bboxes + font size + bold/italic)
 2. Subpath-level hole filtering: drop evenodd text holes from canvas/container
    paths when their corresponding text glyphs are replaced by drawio text cells.
    Fixes the "cut-out" appearance where transparent holes showed through after
    v3 removed text glyph fills.
 3. Native drawio text cells using OCR font metadata

Usage:
    python svg_to_drawio_v4.py input.svg input.jpg [-o out.drawio]
      [--ocr ocr.json]       # cached or precomputed OCR results
      [--ocr-mode claude|easyocr]
"""
from __future__ import annotations

import os
import re
import sys
import json
import base64
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_drawio_v2 import (
    parse_path, expand_path, path_bbox, segments_to_stencil, encode_stencil,
    _strip_ns,
)


# ----------------------------------------------------------------------------
# Subpath utilities
# ----------------------------------------------------------------------------

def split_subpaths(segments):
    """Split a flat segment list into subpaths (each starts with M)."""
    subs = []
    cur = []
    for seg in segments:
        if seg[0] == 'M' and cur:
            subs.append(cur)
            cur = [seg]
        else:
            cur.append(seg)
    if cur:
        subs.append(cur)
    return subs


def subpath_bbox(sub):
    return path_bbox(sub)  # tuple (x0,y0,w,h)


# ----------------------------------------------------------------------------
# OCR fetchers
# ----------------------------------------------------------------------------

def load_ocr(ocr_path: str, mode: str = 'claude',
             img_path: str | None = None) -> list:
    """Load OCR JSON. If not present, run OCR via the selected backend."""
    if Path(ocr_path).exists():
        data = json.loads(Path(ocr_path).read_text())
        # Ensure standard fields
        for r in data:
            r.setdefault('font_size', None)
            r.setdefault('bold', False)
            r.setdefault('italic', False)
        return data
    if img_path is None:
        raise RuntimeError(f'OCR cache not found at {ocr_path} and no image given')
    if mode == 'claude':
        return _run_claude(img_path, ocr_path)
    elif mode == 'easyocr':
        return _run_easy(img_path, ocr_path)
    else:
        raise ValueError(f'unknown ocr mode {mode!r}')


def _run_claude(img_path: str, cache: str) -> list:
    import anthropic
    from PIL import Image
    client = anthropic.Anthropic()
    img = Image.open(img_path); W, H = img.size
    img_b64 = base64.standard_b64encode(Path(img_path).read_bytes()).decode('ascii')
    prompt = (
        f'This image is {W} pixels wide and {H} pixels tall.\n\n'
        'Extract EVERY visible text region as a JSON array. For each text region:\n'
        '- "text": exact text content (preserve case, punctuation, symbols)\n'
        f'- "x1","y1","x2","y2": tight pixel-coord bbox in the {W}x{H} image\n'
        '- "font_size": estimated cap height in pixels\n'
        '- "bold": true/false\n'
        '- "italic": true/false\n\n'
        'Rules: include EVERY text (titles, labels, numbers, captions, symbols). '
        'One entry per visual line. Tight bboxes. Return ONLY the JSON array.'
    )
    msg = client.messages.create(
        model='claude-opus-4-5',
        max_tokens=16000,
        messages=[{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64',
             'media_type': 'image/jpeg', 'data': img_b64}},
            {'type': 'text', 'text': prompt},
        ]}],
    )
    txt = msg.content[0].text.strip()
    if txt.startswith('```'):
        txt = txt.split('\n', 1)[1]
        if txt.endswith('```'):
            txt = txt.rsplit('```', 1)[0]
        txt = txt.rstrip()
        if txt.startswith('json\n'):
            txt = txt[5:]
    data = json.loads(txt)
    Path(cache).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def _run_easy(img_path: str, cache: str) -> list:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    raw = reader.readtext(img_path)
    out = []
    for bbox, text, conf in raw:
        if conf < 0.25 or not text.strip():
            continue
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        out.append({
            'text': text, 'conf': float(conf),
            'x1': int(min(xs)), 'y1': int(min(ys)),
            'x2': int(max(xs)), 'y2': int(max(ys)),
            'font_size': None, 'bold': False, 'italic': False,
        })
    Path(cache).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _parse_hex(fill: str):
    fill = fill.strip()
    if not fill.startswith('#'): return None
    if len(fill) == 4:
        return tuple(int(fill[i]*2, 16) for i in (1, 2, 3))
    if len(fill) == 7:
        return tuple(int(fill[i:i+2], 16) for i in (1, 3, 5))
    return None


def _is_dark(fill: str) -> bool:
    rgb = _parse_hex(fill)
    if rgb is None: return False
    return sum(rgb) / 3 < 120


def _bbox_intersect_ratio(a, b):
    """Return intersection / area-of-a."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0: return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / aa


def _sample_text_color(img, box, default='#333333'):
    import numpy as np
    x0, y0, x1, y1 = box
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0: return default
    roi = img[y0:y1, x0:x1]
    gray = roi.mean(axis=-1)
    mask = gray < 130
    if mask.sum() < 3:
        mask = gray < gray.mean()
        if mask.sum() < 3: return default
    c = roi[mask].mean(axis=0)
    return '#{:02X}{:02X}{:02X}'.format(int(c[0]), int(c[1]), int(c[2]))


# ----------------------------------------------------------------------------
# Main conversion
# ----------------------------------------------------------------------------

def convert(svg_path: str, png_path: str, drawio_path: str,
            ocr_cache: str | None = None, ocr_mode: str = 'claude',
            stencil_size: int = 1000,
            font_family: str = 'Helvetica,Arial,sans-serif') -> dict:
    import numpy as np
    from PIL import Image

    # 1. OCR
    print('[1/5] loading OCR...')
    ocr_cache = ocr_cache or str(Path(svg_path).with_suffix('.ocr.json'))
    ocr = load_ocr(ocr_cache, mode=ocr_mode, img_path=png_path)
    print(f'      {len(ocr)} text regions')
    # Pad OCR boxes more vertically — Claude's bboxes tend to be tight to
    # x-height, missing descenders (g, p, y) which are ~3-4 px below baseline
    ocr_bboxes_padded = [(r['x1'] - 3, r['y1'] - 2,
                          r['x2'] + 3, r['y2'] + 6)
                         for r in ocr]

    # 2. Parse SVG paths, split into subpaths
    print('[2/5] parsing SVG + splitting subpaths...')
    tree = ET.parse(svg_path); root = tree.getroot()
    w_attr = root.get('width', '1376'); h_attr = root.get('height', '768')
    def strip_unit(s):
        m = re.match(r'^\s*([0-9.]+)', s); return float(m.group(1)) if m else 0.0
    W = int(strip_unit(w_attr)); H = int(strip_unit(h_attr))
    if W == 0 or H == 0:
        vb = root.get('viewBox', '').split()
        if len(vb) == 4: W, H = int(float(vb[2])), int(float(vb[3]))

    all_paths = []  # list of {'fill', 'fill_op', 'subpaths': [...]}
    for elem in root.iter():
        if _strip_ns(elem.tag) != 'path': continue
        d = elem.get('d', '')
        if not d.strip(): continue
        fill = elem.get('fill', '#000000')
        if fill == 'none': continue
        try:
            fill_op = float(elem.get('fill-opacity', '1') or 1)
        except ValueError:
            fill_op = 1.0
        segs = parse_path(d)
        if not segs: continue
        expanded = expand_path(segs)
        subs = split_subpaths(expanded)
        all_paths.append({'fill': fill, 'fill_op': fill_op, 'subpaths': subs})
    total_subs = sum(len(p['subpaths']) for p in all_paths)
    print(f'      {len(all_paths)} SVG paths, {total_subs} total subpaths')

    # 3. Filter subpaths:
    #   - Drop entire path: if all its subpaths are small, dark, and in OCR box (text-only path)
    #   - Drop inner subpaths: if the path is light-colored (canvas/container) and has
    #     subpaths (holes) contained in OCR boxes — those are text-shaped holes that
    #     need to be filled to avoid "cut-out" artifacts
    print('[3/5] filtering subpaths...')
    kept_paths = []   # list of {'fill', 'fill_op', 'subpaths': [filtered]}
    dropped_subs_total = 0
    dropped_paths_total = 0
    for p in all_paths:
        is_dark = _is_dark(p['fill'])
        new_subs = []
        for sub in p['subpaths']:
            x0, y0, w, h = subpath_bbox(sub)
            sbb = (x0, y0, x0 + w, y0 + h)
            # Does this subpath belong to any OCR text box?
            in_text = any(_bbox_intersect_ratio(sbb, ob) >= 0.45 for ob in ocr_bboxes_padded)
            if in_text and w < 100 and h < 36:
                # Skip this subpath. If path is dark: it's a text glyph; drop.
                # If path is light: it's a text-shaped hole in a filled bg; drop the hole
                # (makes the canvas solid where text used to be cut out).
                dropped_subs_total += 1
                continue
            new_subs.append(sub)
        if not new_subs:
            dropped_paths_total += 1
            continue
        kept_paths.append({'fill': p['fill'], 'fill_op': p['fill_op'], 'subpaths': new_subs})
    print(f'      dropped {dropped_subs_total} subpaths, {dropped_paths_total} whole paths')

    # 4. Sample text colors
    orig_img = np.asarray(Image.open(png_path).convert('RGB'))

    # Assemble cells
    cells = []
    cid = 100

    # 5a. Emit kept paths as stencil cells
    print('[4/5] emitting stencil cells...')
    for p in kept_paths:
        # flatten subpaths into one stencil per path (preserves evenodd winding)
        flat = [seg for sub in p['subpaths'] for seg in sub]
        bbox = path_bbox(flat)
        x0, y0, w, h = bbox
        if w < 0.1 or h < 0.1: continue
        stencil_xml = segments_to_stencil(flat, bbox, stencil_size=stencil_size)
        stencil_b64 = encode_stencil(stencil_xml)
        parts = [f'shape=stencil({stencil_b64})', f'fillColor={p["fill"]}',
                 'strokeColor=none', 'html=1']
        if p['fill_op'] < 1.0:
            parts.append(f'opacity={int(p["fill_op"]*100)}')
        style = ';'.join(parts) + ';'
        cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    # 5b. Emit drawio text cells on top
    print('[5/5] emitting text cells...')
    for r in ocr:
        text = (r.get('text') or '').strip()
        if not text: continue
        x0, y0, x1, y1 = r['x1'], r['y1'], r['x2'], r['y2']
        w = x1 - x0; h = y1 - y0
        if w < 2 or h < 2: continue
        color = _sample_text_color(orig_img, (x0, y0, x1, y1))
        fs = int(r.get('font_size') or max(6, int(h * 0.58)))
        fs = max(6, fs)
        style_parts = [
            'text', 'html=1', 'strokeColor=none', 'fillColor=none',
            'align=center', 'verticalAlign=middle',
            'whiteSpace=nowrap', 'rounded=0',
            f'fontFamily={font_family}',
            f'fontSize={fs}',
            f'fontColor={color}',
        ]
        font_style_bits = 0
        if r.get('bold'): font_style_bits |= 1
        if r.get('italic'): font_style_bits |= 2
        if font_style_bits:
            style_parts.append(f'fontStyle={font_style_bits}')
        # Pad the cell slightly to accommodate font rendering variance
        pad_x = max(2, int(fs * 0.3)); pad_y = 1
        gx = x0 - pad_x; gy = y0 - pad_y
        gw = w + 2 * pad_x; gh = h + 2 * pad_y
        val = (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                   .replace('"', '&quot;'))
        style = ';'.join(style_parts) + ';'
        cells.append(
            f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{gx}" y="{gy}" width="{gw}" height="{gh}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    body = '\n        '.join(cells)
    drawio = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-04-24T00:00:00.000Z" '
        'agent="svg_to_drawio_v4.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v4" name="traced">\n'
        f'    <mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">\n'
        '      <root>\n'
        '        <mxCell id="0"/>\n'
        '        <mxCell id="1" parent="0"/>\n'
        f'        {body}\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )
    Path(drawio_path).write_text(drawio)
    stats = {
        'total_paths': len(all_paths), 'total_subpaths': total_subs,
        'dropped_subpaths': dropped_subs_total, 'dropped_paths': dropped_paths_total,
        'kept_paths': len(kept_paths), 'text_cells': len(ocr),
        'total_cells': len(cells),
        'drawio_size_kb': os.path.getsize(drawio_path) // 1024,
    }
    print(f'      wrote {drawio_path}')
    print(f'      stats: {stats}')
    return stats


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('png')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--ocr', default=None, help='OCR cache JSON')
    ap.add_argument('--ocr-mode', default='claude', choices=['claude', 'easyocr'])
    ap.add_argument('--stencil-size', type=int, default=1000)
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert(args.svg, args.png, out, ocr_cache=args.ocr,
            ocr_mode=args.ocr_mode, stencil_size=args.stencil_size)
