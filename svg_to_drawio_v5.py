"""svg_to_drawio_v5.py — Path-cluster based text detection.

Instead of trusting OCR for text bounding boxes, we identify text clusters
directly from the SVG: dark-fill small paths that share a baseline form a
text line. Bbox + cap height + bold detection come from pixel-exact SVG
geometry. OCR is only used to label each cluster with its text content.

Pipeline:
 1. Parse SVG; collect "glyph candidates" = paths with dark fill + small bbox.
 2. Cluster candidates into text lines via y-baseline + x-proximity.
 3. For each cluster, measure:
      bbox = union of member path bboxes (pixel-precise)
      font_size = median glyph height (true cap height)
      bold = ratio of ink area / bbox area (threshold ~0.45)
 4. Crop the source PNG at each cluster bbox; run EasyOCR on the crop to
    get the text content (small focused images → higher recognition accuracy).
 5. Emit drawio:
      - Drop all clustered glyph paths
      - Drop canvas evenodd holes inside cluster bboxes (no cut-out artifact)
      - Emit remaining paths as stencil cells
      - Emit one native text cell per cluster with SVG-measured geometry
      - Unclustered candidates (punctuation/symbols OCR won't read) kept as
        stencil cells so visual fidelity is preserved

Usage:
    python svg_to_drawio_v5.py input.svg input.jpg [-o out.drawio]
        [--cluster-cache path.json]   # saved cluster analysis
        [--ocr-cache ocr.json]        # cluster_idx → text mapping
"""
from __future__ import annotations

import os
import re
import sys
import json
import base64
import zlib
import statistics
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_drawio_v2 import (
    parse_path, expand_path, path_bbox, segments_to_stencil, encode_stencil,
    _strip_ns,
)
from svg_to_drawio_v4 import split_subpaths, subpath_bbox


# ----------------------------------------------------------------------------
# Path parsing
# ----------------------------------------------------------------------------

def _parse_hex(fill: str):
    fill = fill.strip()
    if not fill.startswith('#'): return None
    if len(fill) == 4:
        return tuple(int(fill[i]*2, 16) for i in (1, 2, 3))
    if len(fill) == 7:
        return tuple(int(fill[i:i+2], 16) for i in (1, 3, 5))
    return None


def _is_dark(fill: str, thresh: int = 120) -> bool:
    rgb = _parse_hex(fill)
    if rgb is None: return False
    return sum(rgb) / 3 < thresh


def _path_ink_area(segments) -> float:
    """Approximate ink area by bounding polygon vertices (shoelace)."""
    # For stencil purposes this is not strictly needed; approximated by path
    # bbox area × typical 0.5 factor, or by polygon area if we have points.
    x0, y0, w, h = path_bbox(segments)
    return w * h


def parse_svg_paths(svg_path: str):
    """Return (width, height, paths_info).
    paths_info: list of dicts {'idx', 'fill', 'fill_op', 'expanded',
        'subpaths', 'bbox': (x0,y0,x1,y1), 'dark': bool}
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()
    w_attr = root.get('width', '1376'); h_attr = root.get('height', '768')
    def strip_unit(s):
        m = re.match(r'^\s*([0-9.]+)', s); return float(m.group(1)) if m else 0.0
    W = int(strip_unit(w_attr)); H = int(strip_unit(h_attr))
    if W == 0 or H == 0:
        vb = root.get('viewBox', '').split()
        if len(vb) == 4: W, H = int(float(vb[2])), int(float(vb[3]))
    paths_info = []
    idx = 0
    for elem in root.iter():
        if _strip_ns(elem.tag) != 'path': continue
        d = elem.get('d', '')
        if not d.strip(): continue
        fill = elem.get('fill', '#000000')
        if fill == 'none': continue
        try:
            fop = float(elem.get('fill-opacity', '1') or 1)
        except ValueError:
            fop = 1.0
        segs = parse_path(d)
        if not segs: continue
        expanded = expand_path(segs)
        subs = split_subpaths(expanded)
        x0, y0, w, h = path_bbox(expanded)
        paths_info.append({
            'idx': idx, 'fill': fill, 'fill_op': fop,
            'expanded': expanded, 'subpaths': subs,
            'bbox': (x0, y0, x0 + w, y0 + h),
            'dark': _is_dark(fill),
        })
        idx += 1
    return W, H, paths_info


# ----------------------------------------------------------------------------
# Text cluster detection via path clustering
# ----------------------------------------------------------------------------

def identify_glyph_candidates(paths_info):
    """Identify paths that are likely text glyphs: dark-fill, small bbox,
    non-linear shape. Returns list of (idx, bbox, path_info).
    """
    candidates = []
    for p in paths_info:
        if not p['dark']:
            continue
        x0, y0, x1, y1 = p['bbox']
        w = x1 - x0; h = y1 - y0
        # Typical glyph range (at 1376x768 image with ~10-20px text)
        if w < 1 or h < 4 or h > 40:
            continue
        # Reject long thin horizontal (rules, separators, arrow-shafts).
        # Real letters very rarely exceed w/h=2 — even 'm', 'w', 'M' stay
        # under 1.5. Arrow shafts are typically ≥3:1. Use 2.5 as the cut-off
        # to keep wide bold characters but discard arrows/dividers.
        if w > h * 2.5:
            continue
        # Reject square/tall (icons, big shapes): w or h > 80
        if w > 80 or h > 40:
            continue
        candidates.append(p)
    return candidates


def cluster_glyphs(candidates, W: int, H: int):
    """Cluster glyph candidates into text lines.

    Strategy:
      Phase A — horizontal clusters (default):
        - Sort by (y_center, x_left)
        - Group entries whose y-centers are within ~0.5 × glyph_height
        - Within each y-group, split on large x-gaps
      Phase B — vertical clusters (axis labels rotated 90°):
        - For glyphs unclaimed by Phase A
        - Sort by (x_center, y_top)
        - Group entries whose x-centers are within ~0.5 × glyph_width
        - Within each x-group, split on large y-gaps
    Returns list of clusters; vertical clusters have `'vertical': True`.
    """
    if not candidates:
        return []
    entries = []
    for p in candidates:
        x0, y0, x1, y1 = p['bbox']
        entries.append({
            'path': p,
            'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
            'yc': (y0 + y1) / 2, 'xc': (x0 + x1) / 2,
            'h': y1 - y0, 'w': x1 - x0,
        })
    med_h = statistics.median([e['h'] for e in entries])
    med_w = statistics.median([e['w'] for e in entries])
    y_tol = max(3, med_h * 0.45)
    x_tol = max(3, med_w * 0.45)

    # Phase A: horizontal lines
    entries.sort(key=lambda e: (e['yc'], e['x0']))
    lines = []
    for e in entries:
        assigned = False
        for L in lines:
            baseline = L['y_sum'] / L['count']
            if abs(e['yc'] - baseline) <= y_tol:
                L['members'].append(e)
                L['y_sum'] += e['yc']; L['count'] += 1
                assigned = True
                break
        if not assigned:
            lines.append({'members': [e], 'y_sum': e['yc'], 'count': 1})

    horiz_clusters = []
    used = set()
    for L in lines:
        members = sorted(L['members'], key=lambda e: e['x0'])
        if not members: continue
        line_med_w = statistics.median([m['w'] for m in members])
        x_gap_tol = max(8, line_med_w * 2.5)
        cur = [members[0]]
        prev_x1 = members[0]['x1']
        for m in members[1:]:
            if m['x0'] - prev_x1 > x_gap_tol:
                horiz_clusters.append(cur); cur = [m]
            else:
                cur.append(m)
            prev_x1 = max(prev_x1, m['x1'])
        horiz_clusters.append(cur)

    # A horizontal cluster needs ≥3 glyphs to be confidently horizontal text.
    # 1-2 glyph horizontal "clusters" may actually be parts of vertical text.
    confirmed_horiz = []
    suspect_glyphs = []  # 1-2 glyph horizontal clusters
    for cluster in horiz_clusters:
        if len(cluster) >= 3:
            confirmed_horiz.append(cluster)
            for m in cluster:
                used.add(id(m))
        else:
            suspect_glyphs.extend(cluster)

    # Phase B: vertical clusters from suspect glyphs
    suspect_glyphs.sort(key=lambda e: (e['xc'], e['y0']))
    vlines = []
    for e in suspect_glyphs:
        assigned = False
        for L in vlines:
            x_avg = L['x_sum'] / L['count']
            if abs(e['xc'] - x_avg) <= x_tol:
                L['members'].append(e)
                L['x_sum'] += e['xc']; L['count'] += 1
                assigned = True
                break
        if not assigned:
            vlines.append({'members': [e], 'x_sum': e['xc'], 'count': 1})

    vertical_clusters = []
    leftover_horiz = []
    for L in vlines:
        members = sorted(L['members'], key=lambda e: e['y0'])
        if len(members) < 3:
            # Can't confirm vertical; keep as their own (probably horiz singletons)
            for m in members:
                leftover_horiz.append([m])
            continue
        col_med_h = statistics.median([m['h'] for m in members])
        y_gap_tol = max(8, col_med_h * 2.5)
        cur = [members[0]]
        prev_y1 = members[0]['y1']
        for m in members[1:]:
            if m['y0'] - prev_y1 > y_gap_tol:
                if len(cur) >= 3:
                    vertical_clusters.append(cur)
                else:
                    leftover_horiz.extend([[g] for g in cur])
                cur = [m]
            else:
                cur.append(m)
            prev_y1 = max(prev_y1, m['y1'])
        if len(cur) >= 3:
            vertical_clusters.append(cur)
        else:
            leftover_horiz.extend([[g] for g in cur])

    clusters = confirmed_horiz + leftover_horiz + vertical_clusters
    # mark vertical clusters
    is_vertical_set = set(id(c) for c in vertical_clusters)

    # Build cluster metadata
    out = []
    for members in clusters:
        # Filter: single glyph with height > 20 is likely icon detail, not text
        if len(members) == 1 and members[0]['h'] > 20:
            continue
        # Filter: single glyph that's too small to be readable text
        if len(members) == 1 and (members[0]['h'] < 6 or members[0]['w'] < 4):
            continue
        is_vertical = id(members) in is_vertical_set
        xs_min = min(m['x0'] for m in members)
        ys_min = min(m['y0'] for m in members)
        xs_max = max(m['x1'] for m in members)
        ys_max = max(m['y1'] for m in members)
        if is_vertical:
            # For vertical text the "font size" comes from glyph WIDTH (rotated)
            widths = sorted([m['w'] for m in members])
            font_size = int(round(widths[len(widths)//2]))
        else:
            heights = sorted([m['h'] for m in members])
            font_size = int(round(heights[len(heights)//2]))
        member_area = sum(m['w'] * m['h'] for m in members)
        cluster_area = max(1, (xs_max - xs_min) * (ys_max - ys_min))
        ink_ratio = member_area / cluster_area
        is_bold = ink_ratio > 0.45
        out.append({
            'glyphs': members,
            'glyph_path_ids': set(m['path']['idx'] for m in members),
            'bbox': (xs_min, ys_min, xs_max, ys_max),
            'font_size': max(6, font_size),
            'bold': is_bold,
            'ink_ratio': round(ink_ratio, 3),
            'vertical': is_vertical,
        })
    return out


# ----------------------------------------------------------------------------
# OCR each cluster individually (cropped region)
# ----------------------------------------------------------------------------

def ocr_clusters(clusters, png_path: str,
                 cache_path: str | None = None,
                 pad: int = 8,
                 min_w: int = 10,
                 use_claude: bool = True,
                 full_image_ocr: list | None = None) -> None:
    """Run OCR on each cluster crop. Writes {'text', 'conf'} into cluster dict.

    Strategy:
     1. Run EasyOCR on the FULL image ONCE. This gives higher-quality text
        content than per-crop OCR (more context per word).
     2. For each SVG cluster, find the full-image OCR box that overlaps most.
        Copy that text to the cluster. If no overlap → fall back to per-crop.
     3. For clusters with NO match, try Claude as a last resort.

    `full_image_ocr` can be pre-computed and passed in.
    """
    if cache_path and Path(cache_path).exists():
        cached = json.loads(Path(cache_path).read_text())
        for c, r in zip(clusters, cached):
            c['text'] = r.get('text', '')
            c['ocr_conf'] = r.get('conf', 0.0)
            c['ocr_source'] = r.get('source', '')
        return
    import easyocr, cv2, numpy as np
    from PIL import Image
    import io, base64
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    img = np.asarray(Image.open(png_path).convert('RGB'))
    H, W = img.shape[:2]

    # Step 1: full-image OCR (single pass) if not provided
    if full_image_ocr is None:
        print('      running full-image OCR...')
        raw = reader.readtext(img, detail=1, paragraph=False)
        full_image_ocr = []
        for bbox, text, conf in raw:
            if not text.strip(): continue
            xs = [float(p[0]) for p in bbox]; ys = [float(p[1]) for p in bbox]
            full_image_ocr.append({
                'text': text, 'conf': float(conf),
                'x1': int(min(xs)), 'y1': int(min(ys)),
                'x2': int(max(xs)), 'y2': int(max(ys)),
            })
        print(f'      {len(full_image_ocr)} full-image boxes')

    client = None
    if use_claude:
        try:
            from svg_to_drawio.auth import make_anthropic_client
            client = make_anthropic_client()
            if client is None:
                raise RuntimeError('no ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN')
        except Exception as e:
            print(f'  (Claude disabled: {e})')

    REJECT_PATTERNS = [
        'no text', 'image shows', 'appears to', 'there is', "i don't see",
        'cannot identify', 'sorry', 'unclear', 'empty', 'blank',
        "i can't", 'unable to', 'this image', 'the image',
        'icon of', 'symbol of', 'picture of',
    ]

    def _claude_ocr_crop(pil_img) -> str | None:
        if client is None: return None
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        b64 = base64.standard_b64encode(buf.getvalue()).decode('ascii')
        try:
            msg = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=120,
                system=(
                    'You are an OCR engine. Your ONLY output is the literal text '
                    'characters visible in the image, VERBATIM. Rules:\n'
                    '- If the image contains NO text, output the single token: <NONE>\n'
                    '- Never describe the image.\n'
                    '- Never add commentary, quotes, markdown, or explanation.\n'
                    '- Preserve case, punctuation, symbols exactly as shown.\n'
                    '- For multi-line text inside the crop, join with a single space.\n'
                    '- Output nothing except the literal transcription or <NONE>.'
                ),
                messages=[{'role': 'user', 'content': [
                    {'type': 'image', 'source': {'type': 'base64',
                     'media_type': 'image/png', 'data': b64}},
                    {'type': 'text', 'text': 'Transcribe the text.'},
                ]}],
            )
            txt = msg.content[0].text.strip()
            # Filter garbage
            if not txt or txt == '<NONE>' or len(txt) > 200:
                return None
            low = txt.lower()
            if any(p in low for p in REJECT_PATTERNS):
                return None
            # Strip surrounding quotes
            if len(txt) >= 2 and txt[0] in '"\'' and txt[-1] == txt[0]:
                txt = txt[1:-1]
            return txt
        except Exception:
            return None

    def _overlap_ratio(a, b):
        ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
        ix0 = max(ax0,bx0); iy0 = max(ay0,by0)
        ix1 = min(ax1,bx1); iy1 = min(ay1,by1)
        if ix1<=ix0 or iy1<=iy0: return 0.0
        inter = (ix1-ix0)*(iy1-iy0)
        a_area = max(1,(ax1-ax0)*(ay1-ay0))
        return inter / a_area

    # Pre-pass: for each EasyOCR box, find the SINGLE cluster whose bbox best
    # overlaps it (mutually exclusive assignment). This prevents two SVG
    # clusters from claiming the same OCR text (which causes visible overlap).
    cluster_by_idx = {i: c for i, c in enumerate(clusters)}
    ocr_box_owner = {}  # ocr_box_id (idx in full_image_ocr) → cluster_idx
    if full_image_ocr:
        for ob_idx, ob in enumerate(full_image_ocr):
            obbox = (ob['x1'], ob['y1'], ob['x2'], ob['y2'])
            best_cluster_idx, best_score = None, 0.0
            for ci, c in cluster_by_idx.items():
                cb = c['bbox']
                clb = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
                # Score: overlap area / max(cluster area, ocr box area) — favors tight fit
                ix0 = max(clb[0], obbox[0]); iy0 = max(clb[1], obbox[1])
                ix1 = min(clb[2], obbox[2]); iy1 = min(clb[3], obbox[3])
                if ix1 <= ix0 or iy1 <= iy0: continue
                inter = (ix1-ix0)*(iy1-iy0)
                ca = max(1, (clb[2]-clb[0])*(clb[3]-clb[1]))
                oa = max(1, (obbox[2]-obbox[0])*(obbox[3]-obbox[1]))
                score = inter / max(ca, oa)  # tight fit = high
                if score > best_score:
                    best_score, best_cluster_idx = score, ci
            if best_cluster_idx is not None and best_score >= 0.3:
                ocr_box_owner[ob_idx] = best_cluster_idx

    # Reverse: cluster → owned ocr box
    cluster_owns = {}
    for ob_idx, ci in ocr_box_owner.items():
        cluster_owns[ci] = ob_idx

    def _rotate_crop_for_vertical(crop_bgr, vertical: bool):
        """If vertical text, rotate 90° clockwise so it reads horizontally for OCR."""
        if vertical:
            return cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)
        return crop_bgr

    results_out = []
    total = len(clusters)
    claude_calls = 0
    for idx, c in enumerate(clusters):
        x0, y0, x1, y1 = c['bbox']
        cluster_bb = (int(x0), int(y0), int(x1), int(y1))
        is_vertical = c.get('vertical', False)
        text = ''
        conf = 0.0
        source = ''

        # Step 2: only assign full-image OCR text if THIS cluster is the
        # designated owner of the OCR box (prevents dup text on adjacent clusters)
        if idx in cluster_owns and full_image_ocr:
            ob = full_image_ocr[cluster_owns[idx]]
            text = ob['text']; conf = ob['conf']; source = 'full_image'
        # Fallback for clusters without an OCR-owned box: only allow OCR boxes
        # that aren't already claimed by another cluster (prevents dup text)
        else:
            best_box, best_ov, best_ob_idx = None, 0.0, None
            for ob_idx2, ob in enumerate(full_image_ocr or []):
                if ob_idx2 in ocr_box_owner:
                    continue  # claimed by another cluster — skip
                obbox = (ob['x1'], ob['y1'], ob['x2'], ob['y2'])
                ov = _overlap_ratio(cluster_bb, obbox)
                if ov > best_ov:
                    best_ov, best_box, best_ob_idx = ov, ob, ob_idx2
            if best_box and best_ov >= 0.7:
                text = best_box['text']; conf = best_box['conf']; source = 'full_image_fallback'
                # Record claim so a later cluster can't grab the same OCR box
                ocr_box_owner[best_ob_idx] = idx

        # Step 3: crop-level OCR if no good match
        if not text:
            cx0 = max(0, int(x0) - pad); cy0 = max(0, int(y0) - pad)
            cx1 = min(W, int(x1) + pad); cy1 = min(H, int(y1) + pad)
            crop = img[cy0:cy1, cx0:cx1]
            if is_vertical:
                crop = _rotate_crop_for_vertical(crop, True)
            if crop.size and crop.shape[1] >= min_w:
                if crop.shape[0] < 24:
                    up = cv2.resize(crop, (crop.shape[1]*3, crop.shape[0]*3),
                                    interpolation=cv2.INTER_CUBIC)
                else:
                    up = crop
                raw = reader.readtext(up, detail=1, paragraph=False,
                                      decoder='beamsearch', beamWidth=8)
                if raw:
                    if len(raw) > 1:
                        parts = sorted(raw, key=lambda r: min(p[0] for p in r[0]))
                        text = ' '.join(p[1] for p in parts if float(p[2]) > 0.2)
                        conf = min(float(r[2]) for r in raw)
                    else:
                        text = raw[0][1]; conf = float(raw[0][2])
                    source = 'crop'
        # Step 4: Claude last resort
        if not text and client is not None:
            cx0 = max(0, int(x0) - pad); cy0 = max(0, int(y0) - pad)
            cx1 = min(W, int(x1) + pad); cy1 = min(H, int(y1) + pad)
            crop = img[cy0:cy1, cx0:cx1]
            if is_vertical:
                crop = _rotate_crop_for_vertical(crop, True)
            if crop.size and crop.shape[0] >= 6 and crop.shape[1] >= 10:
                pil = Image.fromarray(crop)
                if pil.height < 40:
                    pil = pil.resize((pil.width * 3, pil.height * 3), Image.LANCZOS)
                ctxt = _claude_ocr_crop(pil)
                claude_calls += 1
                if ctxt is not None and ctxt.strip():
                    text = ctxt.strip(); conf = 0.95; source = 'claude'

        c['text'] = text; c['ocr_conf'] = conf; c['ocr_source'] = source
        results_out.append({'text': text, 'conf': conf, 'source': source})
        if (idx + 1) % 40 == 0 or idx == total - 1:
            print(f'    OCR {idx+1}/{total}  (claude: {claude_calls})')
    if cache_path:
        Path(cache_path).write_text(json.dumps(results_out, indent=2, ensure_ascii=False))


# ----------------------------------------------------------------------------
# Text color sampling
# ----------------------------------------------------------------------------

def _sample_text_color(img, cluster_bbox, glyph_paths_fills):
    """For color, use the fill color of the glyph paths (they came from SVG)."""
    # majority fill color
    from collections import Counter
    fills = Counter(glyph_paths_fills)
    most = fills.most_common(1)[0][0]
    return most


# ----------------------------------------------------------------------------
# Main conversion
# ----------------------------------------------------------------------------

def convert(svg_path: str, png_path: str, drawio_path: str,
            cluster_cache: str | None = None,
            ocr_cache: str | None = None,
            stencil_size: int = 1000,
            font_family: str = 'Helvetica,Arial,sans-serif') -> dict:
    print('[1/6] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'      {W}x{H}, {len(paths)} paths')

    print('[2/6] identifying glyph candidates...')
    candidates = identify_glyph_candidates(paths)
    print(f'      {len(candidates)} glyph candidates')

    print('[3/6] clustering into text lines...')
    if cluster_cache and Path(cluster_cache).exists():
        saved = json.loads(Path(cluster_cache).read_text())
        clusters = saved  # glyphs field will be missing — reconstruct
        # Map glyph_path_ids back to sets
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
    else:
        clusters = cluster_glyphs(candidates, W, H)
        if cluster_cache:
            # Save without glyphs list (heavy)
            dump = [{
                'bbox': list(c['bbox']),
                'font_size': c['font_size'],
                'bold': c['bold'],
                'ink_ratio': c['ink_ratio'],
                'glyph_path_ids': sorted(c['glyph_path_ids']),
                'num_glyphs': len(c['glyphs']),
            } for c in clusters]
            Path(cluster_cache).write_text(json.dumps(dump, indent=2))
    print(f'      {len(clusters)} text clusters')

    print('[4/6] OCR each cluster crop...')
    # Optional pre-computed full-image OCR cache (looked up alongside cache files).
    full_ocr = None
    candidates = []
    if cluster_cache:
        cc = Path(cluster_cache); candidates.append(cc.parent / 'ocr.json')
    candidates.append(Path(svg_path).parent / 'ocr.json')
    for cand in candidates:
        if cand.exists():
            full_ocr = json.loads(cand.read_text())
            print(f'      loaded {len(full_ocr)} full-image OCR boxes from {cand}')
            break
    ocr_clusters(clusters, png_path, cache_path=ocr_cache, full_image_ocr=full_ocr)
    # Summary
    okc = sum(1 for c in clusters if c.get('text'))
    print(f'      {okc}/{len(clusters)} clusters got text from OCR')

    print('[5/6] assembling cells...')
    import numpy as np
    from PIL import Image
    orig_img = np.asarray(Image.open(png_path).convert('RGB'))

    # Consume glyph paths:
    #  - Cluster has text → drop glyphs (replaced by drawio text cell).
    #  - Cluster has NO text AND has many glyphs (≥4) → likely text that OCR
    #    failed; drop to avoid double-render with nearby-cluster text.
    #  - Cluster has NO text AND few glyphs (≤3) → probably icon details; keep.
    consumed_path_ids = set()
    for c in clusters:
        num_glyphs = len(c.get('glyph_path_ids', []))
        if c.get('text') or num_glyphs >= 4:
            consumed_path_ids.update(c['glyph_path_ids'])

    # Build OCR bboxes for canvas-hole dropping
    # (use cluster bbox, padded)
    text_bboxes = [c['bbox'] for c in clusters if c.get('text')]

    def _bbox_intersect_ratio_any(sbb, bb_list, thresh=0.5):
        sx0, sy0, sx1, sy1 = sbb
        sa = max(1, (sx1 - sx0) * (sy1 - sy0))
        for bx0, by0, bx1, by1 in bb_list:
            ix0 = max(sx0, bx0 - 3); iy0 = max(sy0, by0 - 2)
            ix1 = min(sx1, bx1 + 3); iy1 = min(sy1, by1 + 6)
            if ix1 <= ix0 or iy1 <= iy0: continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            if inter / sa >= thresh:
                return True
        return False

    cells = []
    cid = 100

    # Emit stencil cells for non-glyph paths
    for p in paths:
        if p['idx'] in consumed_path_ids:
            continue
        # Filter subpaths: drop canvas holes that are text-shaped
        new_subs = []
        for sub in p['subpaths']:
            x0, y0, w, h = subpath_bbox(sub)
            sbb = (x0, y0, x0 + w, y0 + h)
            if w < 100 and h < 36 and _bbox_intersect_ratio_any(sbb, text_bboxes, 0.45):
                # text-shaped hole in a light-colored background — drop
                if not p['dark']:
                    continue
                # else this is a dark glyph not matched to a cluster — drop too
                continue
            new_subs.append(sub)
        if not new_subs:
            continue
        flat = [seg for sub in new_subs for seg in sub]
        bbox = path_bbox(flat)
        x0, y0, w, h = bbox
        if w < 0.1 or h < 0.1: continue
        stencil_xml = segments_to_stencil(flat, bbox, stencil_size=stencil_size)
        stencil_b64 = encode_stencil(stencil_xml)
        style_parts = [f'shape=stencil({stencil_b64})',
                       f'fillColor={p["fill"]}',
                       'strokeColor=none', 'html=1']
        if p['fill_op'] < 1.0:
            style_parts.append(f'opacity={int(p["fill_op"]*100)}')
        style = ';'.join(style_parts) + ';'
        cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    # Build path lookup by idx to support cached clusters (no 'glyphs')
    path_by_idx = {p['idx']: p for p in paths}

    # Emit text cells for clusters with text
    for c in clusters:
        text = (c.get('text') or '').strip()
        if not text: continue
        x0, y0, x1, y1 = c['bbox']
        w = x1 - x0; h = y1 - y0
        if w < 2 or h < 2: continue
        # Dominant glyph fill color = text color
        if c.get('glyphs'):
            fills = [g['path']['fill'] for g in c['glyphs']]
        else:
            fills = [path_by_idx[pid]['fill'] for pid in c.get('glyph_path_ids', [])
                     if pid in path_by_idx]
        from collections import Counter
        color = Counter(fills).most_common(1)[0][0] if fills else '#333333'
        fs = c['font_size']
        style_parts = ['text', 'html=1', 'strokeColor=none', 'fillColor=none',
                       'align=left', 'verticalAlign=middle',
                       'whiteSpace=nowrap', 'rounded=0',
                       f'fontFamily={font_family}',
                       f'fontSize={fs}',
                       f'fontColor={color}']
        bits = 0
        if c.get('bold'): bits |= 1
        if bits: style_parts.append(f'fontStyle={bits}')
        # Small padding
        pad_x = max(1, int(fs * 0.1)); pad_y = 1
        gx = x0 - pad_x; gy = y0 - pad_y
        gw = w + 2 * pad_x; gh = h + 2 * pad_y
        val = (text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                   .replace('"', '&quot;'))
        style = ';'.join(style_parts) + ';'
        cells.append(
            f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    body = '\n        '.join(cells)
    drawio = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-04-24T00:00:00.000Z" '
        'agent="svg_to_drawio_v5.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v5" name="traced">\n'
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
    print(f'[6/6] wrote {drawio_path}  ({len(cells)} cells, {os.path.getsize(drawio_path)//1024}KB)')
    return {
        'paths': len(paths), 'candidates': len(candidates),
        'clusters': len(clusters),
        'clusters_with_text': okc,
        'consumed_paths': len(consumed_path_ids),
        'cells': len(cells),
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('png')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--cluster-cache', default=None)
    ap.add_argument('--ocr-cache', default=None)
    ap.add_argument('--stencil-size', type=int, default=1000)
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    stats = convert(args.svg, args.png, out,
                    cluster_cache=args.cluster_cache,
                    ocr_cache=args.ocr_cache,
                    stencil_size=args.stencil_size)
    print(json.dumps(stats, indent=2))
