"""Semantic recognition layer: turn raw stencil+text pairs into native
drawio shapes with editable text.

The flow is:
  1. For each text cluster, find the closest 'container' SVG path that
     wraps it (a near-rectangular shape whose bbox tightly contains the
     text bbox).
  2. Classify the container's silhouette: rect, rounded_rect, ellipse,
     diamond, cylinder, parallelogram, hexagon.
  3. Emit ONE native drawio mxCell with `value=<text>` and a drawio
     built-in shape style — so the user can edit the text in place,
     drag the box, change colors, etc., like a hand-built diagram.

The container path is then marked as 'consumed' so it doesn't also get
emitted as a separate stencil. Clusters with no matching container fall
back to the existing path-stencil + text-cell pair.
"""
from __future__ import annotations

import math


__all__ = [
    'find_container_for_cluster', 'classify_container_shape',
    'build_semantic_shapes', 'shape_style_for',
    'recognize_vibrant_containers',
]


def _ocr_crop_with_claude(png_path: str, bbox: tuple) -> str | None:
    """OCR a single PNG region with Claude haiku. Returns text or None."""
    try:
        import cv2, io, base64
        from PIL import Image
        from svg_to_drawio.auth import make_anthropic_client
    except ImportError:
        return None
    client = make_anthropic_client()
    if client is None:
        return None
    img = cv2.imread(png_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox)
    px = max(2, int((x1 - x0) * 0.03))
    py = max(2, int((y1 - y0) * 0.10))
    cx0 = max(0, x0 - px); cy0 = max(0, y0 - py)
    cx1 = min(W, x1 + px); cy1 = min(H, y1 + py)
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    if crop.shape[0] < 40:
        sf = 40 / crop.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1]*sf), int(crop.shape[0]*sf)),
                          interpolation=cv2.INTER_CUBIC)
    pil = Image.fromarray(crop[:, :, ::-1])  # BGR→RGB
    buf = io.BytesIO(); pil.save(buf, format='PNG')
    b64 = base64.standard_b64encode(buf.getvalue()).decode('ascii')
    try:
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=60,
            system=(
                'You are an OCR engine. Output ONLY the literal text in the '
                'image — VERBATIM. Preserve punctuation, case, line breaks '
                '(use real newline). If unreadable or no text, output the '
                'single token <NONE>. No quotes, no description.'
            ),
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': b64}},
                {'type': 'text', 'text': 'Transcribe the text.'},
            ]}],
        )
        text = msg.content[0].text.strip().strip('"\'')
        if not text or text == '<NONE>' or len(text) > 200:
            return None
        low = text.lower()
        for bad in ('no text', 'image shows', 'cannot identify', "i don't",
                    'unable to', 'this image'):
            if bad in low:
                return None
        return text
    except Exception:
        return None


def recognize_vibrant_containers(content_paths: list, panel_bgs: list,
                                  consumed: set, png_path: str,
                                  font_family: str = 'Helvetica',
                                  existing_shape_bboxes: list | None = None,
                                  full_image_ocr: list | None = None,
                                  existing_clusters: list | None = None):
    """Find big vibrant-fill rectangles (Stage 0/1/2/3 pills, Final 9B-V
    orange box, etc.) that have WHITE text inside — these don't reach the
    glyph clustering because identify_glyph_candidates filters by `dark`.
    Read their text directly from the PNG via Claude OCR and emit native
    shapes.

    Returns (extra_semantic_cells, extra_consumed).
    """
    def rgb_sum_chroma(h):
        s = h.lstrip('#')
        try:
            r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        except ValueError:
            return 0, 0
        return r + g + b, max(r, g, b) - min(r, g, b)

    existing_shape_bboxes = existing_shape_bboxes or []
    existing_clusters = existing_clusters or []
    pool = content_paths + panel_bgs
    candidates = []
    for p in pool:
        if p['idx'] in consumed:
            continue
        bb = p['bbox']
        w = bb[2] - bb[0]; h = bb[3] - bb[1]
        if w < 60 or h < 18 or w > 480 or h > 100:
            continue
        rs, ch = rgb_sum_chroma(p.get('fill', '#000'))
        # Vibrant container: medium-bright, non-white, has chroma
        if not (200 <= rs <= 650 and ch >= 40):
            continue
        # Skip if already overlapping an existing native shape
        skip = False
        for ebb in existing_shape_bboxes:
            ix0 = max(bb[0], ebb[0]); iy0 = max(bb[1], ebb[1])
            ix1 = min(bb[2], ebb[2]); iy1 = min(bb[3], ebb[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                if inter / max(1, w * h) > 0.6:
                    skip = True; break
        if skip:
            continue
        # Skip if this is actually a multi-label PANEL — i.e. there are
        # already 2+ existing text clusters fully inside this bbox. A real
        # white-text pill has either 0 dark-glyph clusters inside (because
        # its text is white and gets filtered) or 1 (a partial match).
        inside_count = 0
        for c in existing_clusters:
            tb = c.get('bbox')
            if not tb or not (c.get('text') or '').strip():
                continue
            if (bb[0] <= tb[0] and bb[1] <= tb[1]
                    and bb[2] >= tb[2] and bb[3] >= tb[3]):
                inside_count += 1
                if inside_count >= 2:
                    break
        if inside_count >= 2:
            continue
        candidates.append(p)

    # Dedupe by bbox: prefer largest unique-position container
    candidates.sort(key=lambda p: -(p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))
    seen: list[tuple] = []
    picked = []
    for p in candidates:
        bb = p['bbox']
        is_dup = False
        for s in seen:
            ix0 = max(bb[0], s[0]); iy0 = max(bb[1], s[1])
            ix1 = min(bb[2], s[2]); iy1 = min(bb[3], s[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1-ix0)*(iy1-iy0)
                if inter / max(1, (bb[2]-bb[0])*(bb[3]-bb[1])) > 0.5:
                    is_dup = True; break
        if not is_dup:
            seen.append(bb); picked.append(p)

    def _find_text_in_bbox(bb):
        """Concatenate full-image OCR boxes that lie inside bb."""
        if not full_image_ocr:
            return None
        x0, y0, x1, y1 = bb
        lines = []
        for o in full_image_ocr:
            ox0, oy0 = o.get('x1', 0), o.get('y1', 0)
            ox1, oy1 = o.get('x2', 0), o.get('y2', 0)
            ocx, ocy = (ox0+ox1)/2, (oy0+oy1)/2
            if x0 - 2 <= ocx <= x1 + 2 and y0 - 2 <= ocy <= y1 + 2:
                if (o.get('text') or '').strip():
                    lines.append((oy0, o['text'].strip()))
        if not lines:
            return None
        lines.sort()
        return '\n'.join(t for _, t in lines)

    # Lazy EasyOCR — used when full_image_ocr wasn't provided. We OCR each
    # vibrant crop directly so white-on-color labels get captured.
    _easyocr_reader = [None]

    def _easyocr_crop(bb):
        try:
            import easyocr, cv2, numpy as np
        except ImportError:
            return None
        if _easyocr_reader[0] is None:
            try:
                _easyocr_reader[0] = easyocr.Reader(['en'], gpu=True, verbose=False)
            except Exception:
                _easyocr_reader[0] = easyocr.Reader(['en'], gpu=False, verbose=False)
        reader = _easyocr_reader[0]
        img = cv2.imread(png_path)
        if img is None:
            return None
        H, W = img.shape[:2]
        x0, y0, x1, y1 = (int(v) for v in bb)
        px = max(2, int((x1-x0)*0.02)); py = max(2, int((y1-y0)*0.10))
        cx0 = max(0, x0 - px); cy0 = max(0, y0 - py)
        cx1 = min(W, x1 + px); cy1 = min(H, y1 + py)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            return None
        if crop.shape[0] < 30:
            sf = 30 / crop.shape[0]
            crop = cv2.resize(crop, (int(crop.shape[1]*sf), int(crop.shape[0]*sf)),
                              interpolation=cv2.INTER_CUBIC)
        try:
            raw = reader.readtext(crop, detail=1, paragraph=False,
                                   decoder='beamsearch', beamWidth=8)
        except Exception:
            return None
        if not raw:
            return None
        # Group by y-row, then join by x
        raw_sorted = sorted(raw, key=lambda r: (min(p[1] for p in r[0]),
                                                 min(p[0] for p in r[0])))
        lines = []
        last_y = None
        line_buf = []
        for box, text, conf in raw_sorted:
            if float(conf) < 0.3 or not text.strip():
                continue
            ys = [p[1] for p in box]
            top = min(ys); bottom = max(ys); cy = (top+bottom)/2
            if last_y is not None and abs(cy - last_y) > (bottom-top) * 0.7:
                lines.append(' '.join(line_buf)); line_buf = []
            line_buf.append(text.strip())
            last_y = cy
        if line_buf:
            lines.append(' '.join(line_buf))
        out = '\n'.join(lines).strip()
        return out or None

    extra_cells = []
    extra_consumed: set = set()
    accepted_bboxes: list = []
    for p in picked:
        # Prefer full-image OCR (passed in or already cached) — that pass
        # already detected white-on-color text. Fall back to per-crop
        # EasyOCR, then per-crop Claude OCR.
        text = _find_text_in_bbox(p['bbox'])
        if not text:
            text = _easyocr_crop(p['bbox'])
        if not text:
            text = _ocr_crop_with_claude(png_path, p['bbox'])
        if not text:
            continue
        # Drop OCR fragments: text starting with a lowercase letter is
        # usually a partial bleed from a neighbor label. Pills/buttons
        # always start with an uppercase letter (or digit). Also drop
        # noise-like results (<3 alnum chars).
        first = text.lstrip().lstrip('<>').lstrip()[:1]
        if first and first.isalpha() and first.islower():
            continue
        alnum = sum(1 for ch in text if ch.isalnum())
        if alnum < 3:
            continue
        # Drop if this bbox heavily overlaps a vibrant container we
        # already accepted (avoid 'Stage 0' pill duplicates).
        bb = p['bbox']
        a = max(1, (bb[2]-bb[0])*(bb[3]-bb[1]))
        is_dup = False
        for abb in accepted_bboxes:
            ix0 = max(bb[0], abb[0]); iy0 = max(bb[1], abb[1])
            ix1 = min(bb[2], abb[2]); iy1 = min(bb[3], abb[3])
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1-ix0)*(iy1-iy0)
                if inter / a > 0.4:
                    is_dup = True; break
        if is_dup:
            continue
        accepted_bboxes.append(bb)
        shape = classify_container_shape(p)
        bb = p['bbox']
        extra_cells.append({
            'cluster': None,
            'container_path': p,
            'bbox': (bb[0], bb[1], bb[2], bb[3]),
            'shape': shape,
            'fill': p.get('fill', '#ffffff'),
            'text': text,
            'font_size': max(10, min(14, int((bb[3]-bb[1]) * 0.4))),
            'bold': True,
            'font_family': font_family,
        })
        extra_consumed.add(p['idx'])
    return extra_cells, extra_consumed


# Map shape labels to drawio style fragments. The full style is built by
# appending fillColor/strokeColor/fontFamily/fontSize/fontStyle.
SHAPE_STYLES = {
    'rect':           'rounded=0;whiteSpace=wrap;html=1;',
    'rounded_rect':   'rounded=1;whiteSpace=wrap;html=1;arcSize=12;',
    'ellipse':        'ellipse;whiteSpace=wrap;html=1;',
    'diamond':        'rhombus;whiteSpace=wrap;html=1;',
    'cylinder':       'shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;',
    'cloud':          'ellipse;shape=cloud;whiteSpace=wrap;html=1;',
    'hexagon':        'shape=hexagon;whiteSpace=wrap;html=1;',
    'parallelogram':  'shape=parallelogram;whiteSpace=wrap;html=1;',
}


def shape_style_for(shape: str) -> str | None:
    return SHAPE_STYLES.get(shape)


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def _outer_subpath(p: dict):
    """Largest subpath as a polyline endpoint list."""
    from svg_to_drawio.parse import split_subpaths
    subs = split_subpaths(p['expanded'])
    if not subs:
        return []
    best, best_a = [], 0.0
    for s in subs:
        pts = []
        for seg in s:
            cmd = seg[0]
            if cmd in ('M', 'L'):
                pts.append((seg[1], seg[2]))
            elif cmd == 'C':
                pts.append((seg[5], seg[6]))
            elif cmd == 'Q':
                pts.append((seg[3], seg[4]))
        if len(pts) >= 3:
            a = _polygon_area(pts)
            if a > best_a:
                best, best_a = pts, a
    return best


def _bbox_contains(outer, inner, pad: float = 4.0) -> bool:
    """outer bbox contains inner bbox (with padding)."""
    return (outer[0] - pad <= inner[0]
            and outer[1] - pad <= inner[1]
            and outer[2] + pad >= inner[2]
            and outer[3] + pad >= inner[3])


def _rgb_sum(hex_color: str) -> int:
    try:
        s = hex_color.lstrip('#')
        return int(s[0:2], 16) + int(s[2:4], 16) + int(s[4:6], 16)
    except Exception:
        return 0


def find_container_for_cluster(cluster: dict, candidate_paths: list,
                               max_w_ratio: float = 4.0,
                               max_h_ratio: float = 5.0,
                               max_w_abs: float = 420,
                               max_h_abs: float = 220) -> dict | None:
    """Find the SVG path whose bbox tightly contains the text cluster.

    Heuristics:
      - Path bbox contains text cluster bbox (4 px padding allowance).
      - Path is 1.3-`max_w_ratio`× wider AND 1.3-`max_h_ratio`× taller
        than the text. Loose-enough to allow a 2-line label inside a
        single-line-text-bbox cluster, tight enough to reject the full
        panel.
      - Absolute size cap (`max_w_abs`/`max_h_abs`): real container
        boxes in flowcharts are rarely wider than 350 or taller than 100;
        anything beyond that is a panel/section, not a label container.
      - Path fill is pale (RGB sum ≥ 600) — container backgrounds are
        light tints, not solid dark colors.
      - Among candidates, pick the SMALLEST (tightest) area.
    """
    tb = cluster['bbox']
    tw = tb[2] - tb[0]; th = tb[3] - tb[1]
    if tw < 5 or th < 5:
        return None

    candidates = []
    for p in candidate_paths:
        pb = p['bbox']
        pw = pb[2] - pb[0]; ph = pb[3] - pb[1]
        if pw < tw * 1.3 or ph < th * 1.3:
            continue
        if pw > tw * max_w_ratio and pw > max_w_abs * 0.6:
            continue
        if ph > th * max_h_ratio and ph > max_h_abs * 0.6:
            continue
        if pw > max_w_abs or ph > max_h_abs:
            continue
        if not _bbox_contains(pb, tb, pad=4.0):
            continue
        # Accept any non-dark fill — pale (white-ish) AND vibrant (orange,
        # green, purple stage pills). RGB sum < 200 = near-black; those are
        # arrows/lines, not container backgrounds. Anything else is OK.
        fill_sum = _rgb_sum(p.get('fill', '#000'))
        if fill_sum < 200:
            continue
        candidates.append((pw * ph, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def classify_container_shape(path: dict) -> str:
    """Classify a container's silhouette into a drawio shape type.

    Signals:
      - outer-subpath area / bbox area ratio:
          ~1.0  → rect (or rounded_rect)
          ~0.78 → ellipse (circle/oval)
          ~0.5  → diamond (rhombus)
          ~0.85 with rounded corners → rounded_rect
      - Number of vertices in approximated polygon (4 = rect/diamond,
        many = ellipse/cloud)
      - Aspect ratio: very tall and narrow may be a cylinder

    Falls back to 'rect' on ambiguous shapes.
    """
    bb = path['bbox']
    bw = bb[2] - bb[0]; bh = bb[3] - bb[1]
    if bw <= 0 or bh <= 0:
        return 'rect'
    bbox_area = bw * bh

    outer = _outer_subpath(path)
    if not outer:
        return 'rect'
    area = _polygon_area(outer)
    ratio = area / bbox_area

    # Diamond / rhombus
    if 0.40 <= ratio <= 0.60:
        return 'diamond'
    # Ellipse / circle
    if 0.70 <= ratio <= 0.82:
        return 'ellipse'
    # Rectangle territory: distinguish sharp corners from rounded corners
    if ratio > 0.85:
        # Detect rounded corners by looking at how many points cluster
        # near the bbox corners. A sharp rect has exactly 4 corner points;
        # a rounded rect has more (curve segments approximated to several
        # points around each corner).
        corners = [(bb[0], bb[1]), (bb[2], bb[1]),
                   (bb[2], bb[3]), (bb[0], bb[3])]
        near_corner = 0
        threshold = max(3.0, min(bw, bh) * 0.08)
        for px, py in outer:
            for cx, cy in corners:
                if abs(px - cx) < threshold and abs(py - cy) < threshold:
                    near_corner += 1
                    break
        # Heuristic: sharp rect has ~4 distinct corner points; rounded rect
        # has many (one per Bezier control point per corner). Threshold 6+
        # near corner = rounded.
        if near_corner >= 6:
            return 'rounded_rect'
        # Many vertices but not at bbox corners → likely curved/cloud
        if len(outer) > 16:
            return 'rounded_rect'
        return 'rect'
    # Lower ratio fallthrough — odd shape, default to rect
    return 'rect'


def build_semantic_shapes(clusters: list, content_paths: list,
                          consumed: set,
                          font_family: str = 'Helvetica') -> tuple[list, set]:
    """For each text cluster, try to find a container and emit a native
    shape. Returns (semantic_cells_data, extra_consumed_path_idxs).

    `semantic_cells_data` is a list of dicts the caller can turn into
    drawio mxCell XML. We DON'T emit XML here — the caller knows the
    cell-id range and can serialize.
    """
    candidate_paths = [p for p in content_paths
                       if p['idx'] not in consumed
                       and (p['bbox'][2] - p['bbox'][0]) >= 30
                       and (p['bbox'][3] - p['bbox'][1]) >= 18]

    # First pass: for each cluster, find its best container (or None).
    matches: list[tuple[dict, dict]] = []
    for c in clusters:
        if c.get('vertical'):
            continue
        text = (c.get('text') or '').strip()
        if not text:
            continue
        path = find_container_for_cluster(c, candidate_paths)
        if path is None:
            continue
        matches.append((c, path))

    # Second pass: when multiple clusters claim the same container, keep
    # only the one with the LONGEST text (most likely the proper label;
    # the others are OCR fragments inside the same box).
    by_container: dict = {}
    for c, p in matches:
        prev = by_container.get(p['idx'])
        if prev is None:
            by_container[p['idx']] = (c, p)
        else:
            prev_text = (prev[0].get('text') or '').strip()
            cur_text = (c.get('text') or '').strip()
            if len(cur_text) > len(prev_text):
                by_container[p['idx']] = (c, p)

    extra_consumed: set = set()
    semantic_cells = []
    absorbed_clusters: set = set()
    for _, (c, path) in by_container.items():
        shape = classify_container_shape(path)
        bb = path['bbox']
        # Absorb OTHER text clusters that fall fully inside this container
        # — their text becomes additional lines in the native shape and
        # they get suppressed from the regular text-cell emit pass.
        text_lines = [(c['bbox'][1], (c.get('text') or '').strip(), c)]
        for other in clusters:
            if other is c or id(other) in absorbed_clusters:
                continue
            if other.get('vertical'):
                continue
            ot = (other.get('text') or '').strip()
            if not ot:
                continue
            ob = other['bbox']
            if (bb[0] - 4 <= ob[0] and bb[1] - 4 <= ob[1]
                    and bb[2] + 4 >= ob[2] and bb[3] + 4 >= ob[3]):
                text_lines.append((ob[1], ot, other))
        # Sort by y so stacked rows render in visual order
        text_lines.sort(key=lambda t: t[0])
        for _y, _t, ocluster in text_lines:
            if ocluster is not c:
                absorbed_clusters.add(id(ocluster))
        merged_text = '\n'.join(line for _y, line, _o in text_lines)
        semantic_cells.append({
            'cluster': c,
            'container_path': path,
            'bbox': (bb[0], bb[1], bb[2], bb[3]),
            'shape': shape,
            'fill': path.get('fill', '#ffffff'),
            'text': merged_text,
            'font_size': c.get('font_size', 10),
            'bold': c.get('bold', False),
            'font_family': font_family,
        })
        extra_consumed.add(path['idx'])
    # Mark absorbed sub-clusters so the caller can skip their text emit.
    return semantic_cells, extra_consumed, absorbed_clusters
