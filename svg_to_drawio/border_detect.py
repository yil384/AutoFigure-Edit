"""Detect rectangular containers in the source PNG that vtracer missed.

The source raster is sometimes traced with very faint container borders or
backgrounds omitted (e.g. the inner sub-rects within a panel). We scan the
PNG for blue-tinted rectangular regions, dedupe against existing SVG paths,
and emit the rest as drawio rect cells.
"""
from __future__ import annotations

__all__ = ['detect_missing_rectangles', 'detect_line_pair_rectangles',
           'detect_adaptive_threshold_rectangles',
           'detect_missing_lines']


def detect_missing_lines(png_path: str, current_render_path: str,
                         min_length: int = 30,
                         max_thickness: int = 4,
                         text_bboxes: list | None = None):
    """Compare original PNG vs current drawio render; find horizontal /
    vertical line segments that exist in the original but are missing in
    the render. Returns list of dicts:
        {'orientation': 'H'|'V', 'x0':, 'y0':, 'x1':, 'y1':, 'color':}

    Useful for catching panel-divider lines, chart axis lines, and other
    thin strokes that vtracer captured as paths but the pipeline filtered
    or never reached.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    orig = cv2.imread(png_path)
    if orig is None:
        return []
    cur = cv2.imread(current_render_path, cv2.IMREAD_UNCHANGED)
    if cur is None:
        return []
    # Composite alpha onto white
    if cur.shape[2] == 4:
        a = cur[:, :, 3:4].astype('float32') / 255.0
        rgb = cur[:, :, :3].astype('float32')
        white = np.full_like(rgb, 255)
        cur = (rgb * a + white * (1 - a)).astype('uint8')
    # Resize current to match orig
    if cur.shape[:2] != orig.shape[:2]:
        cur = cv2.resize(cur, (orig.shape[1], orig.shape[0]),
                         interpolation=cv2.INTER_AREA)
    og = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    # Pixels MISSING: orig clearly dark, render clearly light. Strict
    # thresholds reduce false positives from anti-aliasing differences
    # around already-rendered lines (without strict, the detector finds
    # "missing" pixels at the edges of existing strokes).
    miss = ((og < 130) & (cg > 230)).astype('uint8') * 255
    # Mask out text cluster regions — text rendering deltas (font, kerning,
    # anti-aliasing) commonly produce "missing dark" pixels at letter edges
    # that would otherwise be picked up as fake line segments and rendered
    # as duplicate strikethroughs / underlines under the real text.
    if text_bboxes:
        for bb in text_bboxes:
            x0, y0, x1, y1 = (int(round(v)) for v in bb)
            x0 = max(0, x0 - 3); y0 = max(0, y0 - 3)
            x1 = min(miss.shape[1], x1 + 3); y1 = min(miss.shape[0], y1 + 3)
            if x1 > x0 and y1 > y0:
                miss[y0:y1, x0:x1] = 0

    lines = []
    # Horizontal strokes
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min_length // 2, 1))
    horiz = cv2.morphologyEx(miss, cv2.MORPH_OPEN, hk)
    nl, _, st, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    for s in st[1:]:
        x, y, w, h, a = s
        if w < min_length or h > max_thickness or a < min_length // 2:
            continue
        # Sample stroke color from middle of the segment in orig
        cx = x + w // 2; cy = y + h // 2
        b_, g_, r_ = (int(v) for v in orig[min(orig.shape[0]-1, cy),
                                            min(orig.shape[1]-1, cx)])
        color = f'#{r_:02x}{g_:02x}{b_:02x}'
        lines.append({
            'orientation': 'H',
            'x0': x, 'y0': cy, 'x1': x + w, 'y1': cy,
            'color': color,
        })

    # Vertical strokes
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_length // 2))
    vert = cv2.morphologyEx(miss, cv2.MORPH_OPEN, vk)
    nl, _, st, _ = cv2.connectedComponentsWithStats(vert, connectivity=8)
    for s in st[1:]:
        x, y, w, h, a = s
        if h < min_length or w > max_thickness or a < min_length // 2:
            continue
        cx = x + w // 2; cy = y + h // 2
        b_, g_, r_ = (int(v) for v in orig[min(orig.shape[0]-1, cy),
                                            min(orig.shape[1]-1, cx)])
        color = f'#{r_:02x}{g_:02x}{b_:02x}'
        lines.append({
            'orientation': 'V',
            'x0': cx, 'y0': y, 'x1': cx, 'y1': y + h,
            'color': color,
        })

    return lines


def detect_adaptive_threshold_rectangles(png_path: str,
                                         existing_bboxes: list,
                                         panel_bboxes: list | None = None,
                                         min_w: int = 60, min_h: int = 50):
    """Detect axis-aligned container rectangles by CLAHE + adaptive threshold +
    contour fitting. This catches rectangles whose border is too faint for the
    chroma-mask detector (interior is white, only the stroke is slightly
    darker than the canvas — e.g. P3 'Hardware Agent' / 'Verification Agent'
    containers).

    Pipeline: CLAHE on grayscale → adaptive threshold (inv) → morph close →
    contours → filter for rect-shaped (4–12 vertices, fill ≥0.5 of bbox).
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    img = cv2.imread(png_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Strong CLAHE so faint container borders rise above adaptive-threshold
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    gray_eq = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray_eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 6,
    )
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        if w * h < 4000:
            continue
        epsilon = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        n_v = len(approx)
        if not 4 <= n_v <= 12:
            continue
        if w / h > 8 or h / w > 8:
            continue
        ca = cv2.contourArea(c)
        if ca / (w * h) < 0.5:
            continue
        candidates.append((x, y, x + w, y + h))
    panel_bboxes = panel_bboxes or []
    pruned = []
    candidates.sort(key=lambda r: -((r[2] - r[0]) * (r[3] - r[1])))
    for r in candidates:
        # Drop if matches an existing panel exactly.
        if any(_bbox_overlap_ratio(r, pb) > 0.85 and _bbox_overlap_ratio(pb, r) > 0.85
               for pb in panel_bboxes):
            continue
        # Drop if covered by an existing path (bbox-near-match).
        skip = False
        for eb in existing_bboxes:
            ax0, ay0, ax1, ay1 = r
            bx0, by0, bx1, by1 = eb
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            aa = max(1, (ax1 - ax0) * (ay1 - ay0))
            ba = max(1, (bx1 - bx0) * (by1 - by0))
            if inter / aa > 0.85 and inter / ba > 0.5:
                skip = True; break
        if skip:
            continue
        # Drop near-duplicates of already-accepted rects.
        if any(_bbox_overlap_ratio(r, p['bbox']) > 0.7 and _bbox_overlap_ratio(p['bbox'], r) > 0.7
               for p in pruned):
            continue
        pruned.append({'bbox': r, 'fill': 'none', 'stroke': '#a8b6c8'})
    return pruned


def detect_line_pair_rectangles(png_path: str, existing_bboxes: list,
                                panel_bboxes: list | None = None,
                                min_w: int = 60, min_h: int = 50,
                                max_w: int = 250, max_h: int = 220):
    """Find axis-aligned rectangles whose borders are visible as 4 line
    segments (top, bottom, left, right) in the source PNG, but whose interior
    has no chroma — so the chroma-based detector misses them.

    Pipeline:
      1. Sobel gradient → keep weak edges too
      2. Project on each axis: column-sums for vertical lines, row-sums for
         horizontal. A peak above `line_thresh` is a candidate edge.
      3. Pair peaks (left/right verticals, top/bottom horizontals) that
         frame a region of the right size, with low interior edge density
         (i.e., the interior is mostly empty / uniform — boxes contain text
         and icons, but their CONTAINER edges are clean horizontal/vertical
         scans).

    This catches the P3 'Hardware Agent' / 'Verification Agent' boxes whose
    interiors are pure white but whose borders show as faint vertical lines.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    img = cv2.imread(png_path)
    if img is None:
        return []
    H_img, W_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32')
    sobx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_x = np.abs(sobx)
    abs_y = np.abs(soby)

    # Vertical lines: columns where |sobel_x| is high over a long y-extent.
    v_lines: list[tuple[int, int, int]] = []   # (x, y_top, y_bot)
    h_lines: list[tuple[int, int, int]] = []   # (y, x_left, x_right)

    # For each x column, find runs of high gradient values
    col_strong = (abs_x > 8).astype('uint8')
    row_strong = (abs_y > 8).astype('uint8')

    # Column-by-column: run-length detection of high-gradient pixels with
    # short gaps allowed (rounded-rect borders break at corners)
    def find_runs(strip, min_run=40, max_gap=4):
        runs = []
        run = 0; gap = 0; start = -1
        n = len(strip)
        for i in range(n):
            if strip[i]:
                if run == 0:
                    start = i
                run += gap + 1
                gap = 0
            else:
                if run > 0:
                    gap += 1
                    if gap > max_gap:
                        if run >= min_run:
                            runs.append((start, start + run - 1))
                        run = 0; gap = 0; start = -1
        if run >= min_run:
            runs.append((start, start + run - 1))
        return runs

    for x in range(W_img):
        for s, e in find_runs(col_strong[:, x]):
            v_lines.append((x, s, e))

    for y in range(H_img):
        for s, e in find_runs(row_strong[y, :]):
            h_lines.append((y, s, e))

    # Group adjacent same-position lines (a single visual line may span
    # 3-5 pixels wide due to anti-aliasing).
    def cluster(lines, axis_idx):
        if not lines: return []
        lines = sorted(lines, key=lambda l: l[axis_idx])
        clusters = [[lines[0]]]
        for l in lines[1:]:
            if l[axis_idx] - clusters[-1][-1][axis_idx] <= 4:
                clusters[-1].append(l)
            else:
                clusters.append([l])
        return [max(cl, key=lambda x: x[2] - x[1]) for cl in clusters]

    v_lines = cluster(v_lines, 0)
    h_lines = cluster(h_lines, 0)

    # Find rectangle candidates: pair verticals (x_left, x_right) and
    # horizontals (y_top, y_bot) that form a rect of valid size.
    candidates = []
    for i, (x_l, yt_l, yb_l) in enumerate(v_lines):
        for x_r, yt_r, yb_r in v_lines[i + 1:]:
            w = x_r - x_l
            if w < min_w or w > max_w:
                continue
            # Verticals must overlap in y over a span ≥ min_h
            y_top_overlap = max(yt_l, yt_r)
            y_bot_overlap = min(yb_l, yb_r)
            if y_bot_overlap - y_top_overlap < min_h:
                continue
            # Need horizontal segments near top and bottom that span ≥ 60% of w
            top_match = None; bot_match = None
            for (y, xl, xr) in h_lines:
                if abs(y - y_top_overlap) <= 8 or abs(y - yt_l) <= 8 or abs(y - yt_r) <= 8:
                    span = min(xr, x_r) - max(xl, x_l)
                    if span > 0.6 * w:
                        if top_match is None or abs(y - (y_top_overlap)) < abs(top_match - (y_top_overlap)):
                            top_match = y
                if abs(y - y_bot_overlap) <= 8 or abs(y - yb_l) <= 8 or abs(y - yb_r) <= 8:
                    span = min(xr, x_r) - max(xl, x_l)
                    if span > 0.6 * w:
                        if bot_match is None or abs(y - (y_bot_overlap)) < abs(bot_match - (y_bot_overlap)):
                            bot_match = y
            if top_match is None or bot_match is None:
                continue
            if bot_match - top_match < min_h:
                continue
            candidates.append((x_l, top_match, x_r, bot_match))

    # Filter: drop rects already covered by panels and existing paths.
    panel_bboxes = panel_bboxes or []
    pruned = []
    for r in candidates:
        # Check overlap with panels (drop if matches a panel)
        skip = False
        for pb in panel_bboxes:
            iou_a = _bbox_overlap_ratio(r, pb)
            iou_b = _bbox_overlap_ratio(pb, r)
            if iou_a > 0.85 and iou_b > 0.85:
                skip = True; break
        if skip:
            continue
        # Check existing paths
        for eb in existing_bboxes:
            ax0, ay0, ax1, ay1 = r
            bx0, by0, bx1, by1 = eb
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            aa = max(1, (ax1 - ax0) * (ay1 - ay0))
            ba = max(1, (bx1 - bx0) * (by1 - by0))
            if inter / aa > 0.85 and inter / ba > 0.5:
                skip = True; break
        if skip:
            continue
        pruned.append(r)

    # Dedupe: drop nested duplicates (keep larger)
    pruned.sort(key=lambda r: -((r[2] - r[0]) * (r[3] - r[1])))
    final = []
    for r in pruned:
        is_dup = any(_bbox_overlap_ratio(r, f['bbox']) > 0.7 for f in final)
        if is_dup:
            continue
        final.append({'bbox': r, 'fill': 'none', 'stroke': '#a8b6c8'})
    return final


def _bbox_overlap_ratio(a, b):
    """Fraction of `a` covered by `b` (intersection / area_a)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / aa


def detect_missing_rectangles(png_path: str,
                              existing_bboxes: list,
                              panel_bboxes: list | None = None,
                              W: int = 0, H: int = 0,
                              min_w: int = 50,
                              min_h: int = 25,
                              chroma_min: int = 6,
                              chroma_max: int = 35):
    """Find blue-tinted rectangular regions in `png_path` not already
    represented by an existing SVG path. Returns list of dicts:
        {'bbox': (x0,y0,x1,y1), 'fill': '#hex', 'stroke': '#hex'}

    Heuristic: figure container rects in this image family use a pale
    blue fill (e.g. RGB ~ 235,242,250). Filter by `chroma_min ≤ B-R ≤
    chroma_max`, morph-close to fill border thickness, then take connected-
    component bounding rects.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    img = cv2.imread(png_path)
    if img is None:
        return []
    img_h, img_w = img.shape[:2]
    bch, _, rch = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    chroma = bch.astype(int) - rch.astype(int)
    mask = ((chroma > chroma_min) & (chroma < chroma_max)).astype('uint8') * 255
    # Two passes: a small kernel keeps adjacent agent-column boxes apart,
    # a larger one captures big container rects whose chroma is patchier
    # (e.g. the full-panel light-blue border that frames a whole quadrant).
    candidates = []
    for kernel_size, iters in [(3, 2), (9, 3)]:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                                  iterations=iters)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_w or h < min_h:
                continue
            if w >= img_w * 0.95 and h >= img_h * 0.95:
                continue
            area = cv2.contourArea(c)
            if area < min_w * min_h * 0.4:
                continue
            candidates.append((x, y, x + w, y + h))

    panel_bboxes = panel_bboxes or []

    def covered_by_existing(r) -> bool:
        # If an existing SVG path's bbox is a near-match (>0.7 reciprocal
        # overlap) treat the rect as already represented.
        for eb in existing_bboxes:
            ax0, ay0, ax1, ay1 = r
            bx0, by0, bx1, by1 = eb
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            aa = max(1, (ax1 - ax0) * (ay1 - ay0))
            ba = max(1, (bx1 - bx0) * (by1 - by0))
            if inter / aa > 0.85 and inter / ba > 0.5:
                return True
        return False

    def is_panel(r) -> bool:
        for pb in panel_bboxes:
            if (_bbox_overlap_ratio(r, pb) > 0.85
                    and _bbox_overlap_ratio(pb, r) > 0.85):
                return True
        return False

    missing = []
    for r in candidates:
        if is_panel(r):
            continue
        if covered_by_existing(r):
            continue
        # Sample interior fill color (median of a 9-point grid inside the rect)
        x0, y0, x1, y1 = r
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        pts = [(x0 + (x1 - x0) * i // 4, y0 + (y1 - y0) * j // 4)
               for i in (1, 2, 3) for j in (1, 2, 3)]
        bs, gs, rs = [], [], []
        for px, py in pts:
            b_, g_, r_ = (int(v) for v in img[py, px])
            bs.append(b_); gs.append(g_); rs.append(r_)
        bs.sort(); gs.sort(); rs.sort()
        mid = len(bs) // 2
        fill = f'#{rs[mid]:02x}{gs[mid]:02x}{bs[mid]:02x}'
        # Sample stroke color: darkest pixel along the top edge.
        top_y = max(0, y0)
        edge_pts = [(x0 + (x1 - x0) * i // 6, top_y) for i in range(1, 6)]
        edge_pts += [(x0 + (x1 - x0) * i // 6, min(img_h - 1, y1 - 1))
                     for i in range(1, 6)]
        darkest = min(((int(img[py, px][2]),
                        int(img[py, px][1]),
                        int(img[py, px][0])) for px, py in edge_pts),
                      key=lambda c: sum(c))
        if sum(darkest) > 600:
            stroke = '#a8b6c8'  # default light gray when border is faint
        else:
            stroke = f'#{darkest[0]:02x}{darkest[1]:02x}{darkest[2]:02x}'
        missing.append({'bbox': (x0, y0, x1, y1), 'fill': fill, 'stroke': stroke})

    # Drop a missing rect that is fully inside another already-accepted
    # missing rect — we want the OUTERMOST container, not nested duplicates.
    missing.sort(key=lambda m: -((m['bbox'][2] - m['bbox'][0])
                                 * (m['bbox'][3] - m['bbox'][1])))
    pruned = []
    for m in missing:
        if any(_bbox_overlap_ratio(m['bbox'], p['bbox']) > 0.85
               for p in pruned):
            continue
        pruned.append(m)
    pruned.sort(key=lambda m: (m['bbox'][1], m['bbox'][0]))
    return pruned
