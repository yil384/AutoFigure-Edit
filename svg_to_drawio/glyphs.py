"""Glyph candidate identification, clustering, and outlier removal.

Outlier removal: drops paths whose w/h is >2.5x the cluster median (catches
panel border or icon outlines accidentally swept into a text cluster).
"""
from svg_to_drawio_v5 import identify_glyph_candidates as _v5_glyph_candidates, cluster_glyphs

__all__ = [
    'identify_glyph_candidates', 'cluster_glyphs', 'remove_glyph_outliers',
    'reabsorb_vertical_glyphs', 'evict_non_glyph_paths',
    'split_spatially_joined_clusters', 'is_rectangle_silhouette',
]


def _polygon_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace area, absolute value."""
    n = len(pts)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def _outer_subpath_area(p: dict) -> float:
    """Return the area of the LARGEST subpath only (outer silhouette).

    This measures how rectangle-shaped the path is, regardless of what
    'holes' the evenodd path data declares. Chip-body / icon-block paths
    have outer ≈ bbox area even when their inner subpaths cut out details
    (eyes, screen, brain) — those holes don't make the silhouette less
    rectangular, they just punch out lighter regions that get redrawn on
    top by other paths. For glyph detection we want to reject these
    rectangle-silhouette paths, not measure net painted area.
    """
    from svg_to_drawio.parse import split_subpaths
    subs = split_subpaths(p['expanded'])
    if not subs:
        return 0.0
    best = 0.0
    for s in subs:
        pts = []
        for seg in s:
            if seg[0] in ('M', 'L'):
                pts.append((seg[1], seg[2]))
            elif seg[0] == 'C':
                pts.append((seg[5], seg[6]))
            elif seg[0] == 'Q':
                pts.append((seg[3], seg[4]))
        if len(pts) >= 3:
            a = _polygon_area(pts)
            if a > best:
                best = a
    return best


def identify_glyph_candidates(paths_info):
    return _v5_glyph_candidates(paths_info)


def is_rectangle_silhouette(p: dict, threshold: float = 0.70) -> bool:
    """True when path's outer-subpath area ≥ threshold × bbox area, i.e.
    the path's silhouette is essentially a rectangle (or filled circle).
    Used to keep chip bodies, header strips, ZX-graph colored circles,
    and other icon-block paths from being consumed as if they were text
    glyphs. Threshold 0.70 catches both rectangles (~1.0) and circles
    (π/4 ≈ 0.785).
    """
    bb = p['bbox']
    bw = bb[2] - bb[0]; bh = bb[3] - bb[1]
    if bw <= 0 or bh <= 0 or bw < 20 or bh < 20:
        return False
    outer = _outer_subpath_area(p)
    return outer / (bw * bh) > threshold


def split_spatially_joined_clusters(clusters: list, path_by_idx: dict,
                                    min_gap_ratio: float = 1.5):
    """Split horizontal clusters whose glyphs span two (or more) labels
    separated by a horizontal gap. Uses glyph spatial positions, not OCR text.

    Why: when two labels share a y-baseline with a horizontal gap (e.g. P3's
    'Bloch-' and 'Interfero-' as the top row of two separate stacked labels),
    cluster_glyphs greedily merges them on baseline match. The combined OCR
    text shows the gap as a space ('Bloch- Interfero-'), but downstream
    rendering and any cross-line merges treat them as one label — wrong.

    A horizontal gap > glyph_height * `min_gap_ratio` between adjacent glyphs
    is the split signal. The cluster's existing OCR text is sliced by space:
    one word per gap-separated group. A sub-cluster whose word count does not
    match groups gets `text = ''` (caller can re-OCR if desired).

    Returns (new_clusters, n_split). Sub-clusters inherit font_size, bold,
    vertical from the parent.
    """
    new_clusters = []
    n = 0
    for c in clusters:
        if c.get('vertical'):
            new_clusters.append(c); continue
        text = (c.get('text') or '').strip()
        if not text:
            new_clusters.append(c); continue
        # Split is appropriate only when the cluster's glyphs span MORE labels
        # than its OCR text suggests. Two heuristics qualify a cluster:
        #   A. Text contains multiple words that EACH end with '-' — that's
        #      two adjacent stacked-label tops, e.g. 'Bloch- Interfero-'.
        #   B. Glyph count is substantially greater than letter count
        #      (>1.4×) — the cluster absorbed glyphs from a sibling label
        #      whose text the OCR didn't capture, e.g. i=103 'Messiah' with
        #      12 glyphs (covering 'Messiah'+'meters', text='Messiah').
        # Otherwise normal multi-word labels like 'Filter Funnel' would split.
        words = text.split()
        hyphen_word_count = sum(1 for w in words if w.endswith('-'))
        n_glyphs = len(c.get('glyph_path_ids', []))
        n_letters = sum(1 for ch in text if ch.isalnum())
        is_multi_hyphen = hyphen_word_count >= 2
        is_glyph_excess = (n_letters > 0 and n_glyphs / n_letters > 1.4 and n_glyphs >= 8)
        if not (is_multi_hyphen or is_glyph_excess):
            new_clusters.append(c); continue
        gids = list(c.get('glyph_path_ids', []))
        if len(gids) < 4:
            new_clusters.append(c); continue
        # Collect glyph bboxes; skip cluster if any glyph isn't in path_by_idx
        glyphs = []
        for gid in gids:
            p = path_by_idx.get(gid)
            if not p:
                glyphs = None; break
            bb = p['bbox']
            glyphs.append((bb[0], bb[1], bb[2], bb[3], gid))
        if not glyphs:
            new_clusters.append(c); continue
        glyphs.sort(key=lambda g: g[0])
        heights = sorted(g[3] - g[1] for g in glyphs)
        med_h = heights[len(heights) // 2] if heights else 1.0
        widths = sorted(g[2] - g[0] for g in glyphs)
        med_w = widths[len(widths) // 2] if widths else 1.0
        # Compute every adjacent gap. Clamp negatives.
        per_gap = []
        for i in range(len(glyphs) - 1):
            gap = max(0.0, glyphs[i + 1][0] - glyphs[i][2])
            per_gap.append(gap)
        if not per_gap:
            new_clusters.append(c); continue
        # Word boundary signal: only split when a gap is BOTH
        #   - absolutely large: > 3 px AND > 60% of median glyph width
        #     (a word-space is roughly half-to-full character width)
        #   - relatively an outlier: > 2.5x the median letter gap, so we
        #     don't split inside a single word that has slightly varying
        #     letter spacing
        sorted_gaps = sorted(per_gap)
        median_letter_gap = sorted_gaps[len(sorted_gaps) // 2]
        abs_threshold = max(3.0, med_w * 0.6)
        rel_threshold = max(2.5, median_letter_gap * 2.5)
        gap_threshold = max(abs_threshold, rel_threshold)
        # Build groups
        groups = [[glyphs[0]]]
        for i in range(1, len(glyphs)):
            if per_gap[i - 1] > gap_threshold:
                groups.append([glyphs[i]])
            else:
                groups[-1].append(glyphs[i])
        if len(groups) < 2:
            new_clusters.append(c); continue
        # Drop tiny groups (likely punctuation noise) — require ≥2 glyphs.
        big_groups = [g for g in groups if len(g) >= 2]
        if len(big_groups) < 2:
            new_clusters.append(c); continue
        # OK: split. Distribute the OCR text by space if the word count matches
        # the number of big groups; otherwise leave sub-text empty.
        text = (c.get('text') or '').strip()
        words = text.split() if text else []
        # Skip split when the parent has more text words than glyph groups
        # (OCR captured more structure than spatial gaps reveal).
        if words and len(words) > len(big_groups):
            new_clusters.append(c); continue
        same_count = (len(words) == len(big_groups))
        # Heuristic: when word count is one less than group count, the parent
        # OCR missed exactly one word — the textless group is the rightmost
        # one (continuation cell, e.g. 'Messiah' + 'meters' where OCR only
        # got 'Messiah'). For larger gaps (words count differs by ≥2),
        # alignment between OCR text and groups is ambiguous; keep the
        # parent intact rather than guess wrong (which previously assigned
        # 'Surgery' / 'Scores' to the leftmost two of FOUR glyph groups,
        # leaving the rightmost two to be re-OCR'd as garbage).
        if not same_count and len(words) != len(big_groups) - 1:
            new_clusters.append(c); continue
        for gi, group in enumerate(big_groups):
            xs = [g[0] for g in group] + [g[2] for g in group]
            ys = [g[1] for g in group] + [g[3] for g in group]
            sub_bbox = [min(xs), min(ys), max(xs), max(ys)]
            if same_count:
                sub_text = words[gi]
                sub_source = (c.get('ocr_source', '') + '+split')
            elif gi < len(words):
                # words count = groups count - 1: assign words to the
                # leftmost N groups; the trailing group needs OCR.
                sub_text = words[gi]
                sub_source = (c.get('ocr_source', '') + '+split')
            else:
                sub_text = ''
                sub_source = 'split_pending'
            sub = {
                'bbox': sub_bbox,
                'font_size': c.get('font_size', 10),
                'bold': c.get('bold', False),
                'ink_ratio': c.get('ink_ratio', 0.5),
                'glyph_path_ids': set(g[4] for g in group),
                'num_glyphs': len(group),
                'vertical': False,
                'text': sub_text,
                'ocr_conf': c.get('ocr_conf', 0.0) if sub_text else 0.0,
                'ocr_source': sub_source,
            }
            new_clusters.append(sub)
        n += 1
    return new_clusters, n


def evict_non_glyph_paths(clusters, path_by_idx, max_aspect: float = 2.5):
    """Remove paths from clusters that are clearly not letters (e.g. arrow
    shafts wider than 2.5x their height). Without this, an arrow body whose
    bbox happened to align with a text-row baseline gets swept into a text
    cluster, which then `_consume_glyph_paths` claims, hiding the arrow.

    Returns (clusters, n_evicted).
    """
    n = 0
    for c in clusters:
        gids = list(c.get('glyph_path_ids', []))
        if not gids:
            continue
        kept = []
        for gid in gids:
            p = path_by_idx.get(gid)
            if not p:
                kept.append(gid)
                continue
            bb = p['bbox']
            w = bb[2] - bb[0]; h = bb[3] - bb[1]
            if h < 0.5:
                kept.append(gid); continue
            if w / h > max_aspect or h / max(w, 0.5) > max_aspect * 2:
                n += 1
                continue  # too wide / too tall to be a glyph
            kept.append(gid)
        if len(kept) != len(gids):
            was_set = isinstance(c.get('glyph_path_ids'), set)
            c['glyph_path_ids'] = set(kept) if was_set else kept
            c['num_glyphs'] = len(kept)
            if kept:
                xs0, ys0, xs1, ys1 = [], [], [], []
                for gid in kept:
                    p = path_by_idx.get(gid)
                    if not p:
                        continue
                    xs0.append(p['bbox'][0]); ys0.append(p['bbox'][1])
                    xs1.append(p['bbox'][2]); ys1.append(p['bbox'][3])
                if xs0:
                    c['bbox'] = [min(xs0), min(ys0), max(xs1), max(ys1)]
    return clusters, n


def reabsorb_vertical_glyphs(clusters, path_by_idx):
    """Reclaim glyphs from spurious horizontal clusters that actually belong
    to a vertical neighbor.

    cluster_glyphs Phase A is greedy on the y-baseline: a glyph from a vertical
    label can land in a horizontal cluster purely because some other ink shares
    its y-row (e.g. a chart bar tip). When dedup later drops that horizontal
    cluster as noise, the vertical glyph dies with it. This pass pulls those
    glyphs into the vertical cluster they geometrically belong to so the
    vertical text bbox grows to cover the full label.

    Heuristic: for each horizontal cluster whose glyphs all sit inside the
    x-column of some vertical cluster (within ±x_tol) and within y-distance of
    that vertical column, transfer those glyphs.

    Returns (clusters, n_moved).
    """
    moved = 0
    verticals = [c for c in clusters if c.get('vertical')]
    if not verticals:
        return clusters, 0

    # Only steal from clusters whose OCR text is short/noisy. Two real
    # horizontal labels never need stealing from each other; the danger case is
    # 'horizontal' clusters that are really fragments of a vertical label
    # mixed in with chart-ink debris on the same y-row. EasyOCR usually returns
    # 1-2 char gibberish on those.
    def _looks_like_noise(t: str) -> bool:
        if not t:
            return True
        if len(t) <= 2:
            return True
        return not any(ch.isalpha() for ch in t)

    for vc in verticals:
        vbb = vc['bbox']
        vx0, vy0, vx1, vy1 = vbb
        vxc = (vx0 + vx1) / 2
        vw = vx1 - vx0
        x_tol = max(4, vw)
        y_pad = max(20, (vy1 - vy0) * 0.6)
        y_min = vy0 - y_pad
        y_max = vy1 + y_pad
        for hc in clusters:
            if hc is vc or hc.get('vertical'):
                continue
            if not _looks_like_noise((hc.get('text') or '').strip()):
                continue  # legitimate horizontal label, leave alone
            hgids = list(hc.get('glyph_path_ids', []))
            if not hgids:
                continue
            in_glyph_ids = []
            for gid in hgids:
                p = path_by_idx.get(gid)
                if not p:
                    continue
                bb = p['bbox']
                gxc = (bb[0] + bb[2]) / 2
                gyc = (bb[1] + bb[3]) / 2
                if abs(gxc - vxc) <= x_tol and y_min <= gyc <= y_max:
                    in_glyph_ids.append(gid)
            if not in_glyph_ids:
                continue
            # Transfer matched glyphs only (leave the rest in hc)
            was_set = isinstance(vc.get('glyph_path_ids'), set)
            vgids = set(vc.get('glyph_path_ids', set())) if was_set else list(vc.get('glyph_path_ids', []))
            for gid in in_glyph_ids:
                if was_set:
                    vgids.add(gid)
                elif gid not in vgids:
                    vgids.append(gid)
            vc['glyph_path_ids'] = vgids
            vc['num_glyphs'] = len(vgids)
            # Drop transferred glyphs from hc
            taken = set(in_glyph_ids)
            hgids_kept = [gid for gid in hgids if gid not in taken]
            if was_set:
                hc['glyph_path_ids'] = set(hgids_kept)
            else:
                hc['glyph_path_ids'] = hgids_kept
            hc['num_glyphs'] = len(hgids_kept)
            moved += len(in_glyph_ids)
            # Recompute vertical cluster bbox
            xs0, ys0, xs1, ys1 = [], [], [], []
            for gid in vgids:
                p = path_by_idx.get(gid)
                if not p:
                    continue
                xs0.append(p['bbox'][0]); ys0.append(p['bbox'][1])
                xs1.append(p['bbox'][2]); ys1.append(p['bbox'][3])
            if xs0:
                vc['bbox'] = [min(xs0), min(ys0), max(xs1), max(ys1)]
    return clusters, moved


def remove_glyph_outliers(clusters, path_by_idx):
    """Remove glyphs from clusters whose width/height exceeds 2.5x the cluster's
    median glyph dimension. Recomputes cluster bbox from remaining glyphs.
    Returns (clusters, n_outliers_returned).

    Use case: when cluster_glyphs sweeps in a panel border (e.g. 24x29 gray
    rounded rect) along with the actual letters of "Key:" (6x10 each), the
    border gets dropped here so it can render as a separate shape.
    """
    n_outliers = 0
    for c in clusters:
        gids = list(c.get('glyph_path_ids', []))
        if len(gids) < 3:
            continue
        widths, heights = [], []
        for gid in gids:
            p = path_by_idx.get(gid)
            if not p:
                continue
            bb = p['bbox']
            widths.append(bb[2] - bb[0])
            heights.append(bb[3] - bb[1])
        if len(widths) < 3:
            continue
        ws, hs = sorted(widths), sorted(heights)
        mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
        if mw < 1 or mh < 1:
            continue
        kept = []
        for gid in gids:
            p = path_by_idx.get(gid)
            if not p:
                kept.append(gid)
                continue
            bb = p['bbox']
            if (bb[2] - bb[0]) > mw * 2.5 or (bb[3] - bb[1]) > mh * 2.5:
                n_outliers += 1
                continue
            kept.append(gid)
        was_set = isinstance(c.get('glyph_path_ids'), set)
        c['glyph_path_ids'] = set(kept) if was_set else kept
        c['num_glyphs'] = len(kept)
        if kept:
            xs0, ys0, xs1, ys1 = [], [], [], []
            for gid in kept:
                p = path_by_idx.get(gid)
                if not p:
                    continue
                xs0.append(p['bbox'][0])
                ys0.append(p['bbox'][1])
                xs1.append(p['bbox'][2])
                ys1.append(p['bbox'][3])
            if xs0:
                c['bbox'] = [min(xs0), min(ys0), max(xs1), max(ys1)]
    return clusters, n_outliers
