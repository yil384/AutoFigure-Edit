"""svg_to_drawio_v12.py — v11 + transparent background + text dedup + stronger quantize.

Fixes per user feedback:
 1. White full-canvas background path → DROPPED (drawio default shows as transparent)
 2. Overlapping text cells → de-duplicated (merge boxes with same content and high overlap)
 3. Icons — more aggressive color quantization (threshold 30 vs 20)
"""
from __future__ import annotations

import os, re, sys, json
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_drawio_v2 import (
    parse_path, expand_path, path_bbox, segments_to_stencil, encode_stencil,
    _strip_ns,
)
from svg_to_drawio_v4 import split_subpaths, subpath_bbox
from svg_to_drawio_v5 import (
    _parse_hex, _is_dark,
    parse_svg_paths, identify_glyph_candidates,
    cluster_glyphs, ocr_clusters,
)
from svg_to_drawio_v6 import bbox_overlap_or_near, bbox_union, cluster_icons
from svg_to_drawio_v7 import segments_to_svg_d, svg_to_image_cell_style
from svg_to_drawio_v9 import quantized_icon_svg
from svg_to_drawio_v11 import absorb_singletons


def is_full_canvas_background(p, W, H) -> bool:
    """True if this path is a full-page white/near-white rectangle (background)."""
    if p['fill'] != '#ffffff' and p['fill'].upper() not in ('#FFFFFF', '#FFF'):
        return False
    x0, y0, x1, y1 = p['bbox']
    return (x0 < 5 and y0 < 5 and x1 > W - 5 and y1 > H - 5)


def dedupe_text_clusters(clusters):
    """Drop clusters whose bbox heavily overlaps another (likely sub-cluster
    of the same text region):
      - Adjacent clusters where one's bbox is mostly inside the other → drop smaller
      - Same text + any overlap → drop the smaller
      - One text contained in the other AND overlapping → drop the contained one
    """
    def _overlap(a, b):
        ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
        ix0=max(ax0,bx0); iy0=max(ay0,by0)
        ix1=min(ax1,bx1); iy1=min(ay1,by1)
        if ix1<=ix0 or iy1<=iy0: return 0
        return (ix1-ix0)*(iy1-iy0)

    # Pre-pass: drop clusters that are sub-fragments of larger neighbors.
    # If cluster a has text that is a substring of cluster b's text, AND they
    # are on the same row, AND a is positioned at the start of where b begins
    # (or vice versa), → a is a leftover OCR fragment, drop it.
    drop_idx = set()
    for i, a in enumerate(clusters):
        if i in drop_idx: continue
        a_text = (a.get('text') or '').strip()
        a_bb = a['bbox']; aa = max(1,(a_bb[2]-a_bb[0])*(a_bb[3]-a_bb[1]))
        a_glyphs = len(a.get('glyph_path_ids', []))
        a_yc = (a_bb[1] + a_bb[3]) / 2
        a_h = a_bb[3] - a_bb[1]
        for j, b in enumerate(clusters):
            if i == j or j in drop_idx: continue
            b_text = (b.get('text') or '').strip()
            b_bb = b['bbox']; ba = max(1,(b_bb[2]-b_bb[0])*(b_bb[3]-b_bb[1]))
            b_glyphs = len(b.get('glyph_path_ids', []))
            b_yc = (b_bb[1] + b_bb[3]) / 2
            ov = _overlap(a_bb, b_bb)

            # Condition 1: bbox heavily contained
            if ov / aa > 0.6 and aa < ba * 0.6 and a_glyphs < b_glyphs:
                drop_idx.add(i); break

            # Condition 2: substring text + same row + adjacent or overlapping
            same_row = abs(a_yc - b_yc) < max(4, a_h * 0.5)
            if not (a_text and b_text and same_row): continue
            # Normalize dashes (— ‐ – etc) to '-' so OCR variants of the
            # same hyphenated label compare as equal substrings.
            def _norm_dashes(s):
                return s.replace('—', '-').replace('–', '-').replace('‐', '-')
            ta = _norm_dashes(a_text.lower())
            tb = _norm_dashes(b_text.lower())
            # Adjacency in x: gap proportional to font size (some glyphs split
            # by larger inter-letter spacing).
            x_gap = max(b_bb[0] - a_bb[2], a_bb[0] - b_bb[2])
            font_h = max(a_h, 1)
            adjacent = x_gap < max(8, font_h * 1.5)
            # Identical text: same row + adjacent → could be a duplicate fragment
            # from one OCR box assigned to two glyph clusters (mutex failure),
            # OR two genuinely distinct labels with the same text. Discriminate
            # by glyph count: if combined glyphs > text_length, both clusters
            # individually have enough glyphs to be full renderings → distinct.
            # If combined glyphs ≤ text_length, they're fragments of one render.
            if ta == tb and adjacent:
                combined = a_glyphs + b_glyphs
                text_len = len(ta.replace(' ', ''))
                if combined <= max(3, int(text_len * 1.2)):
                    ux0 = min(a_bb[0], b_bb[0]); uy0 = min(a_bb[1], b_bb[1])
                    ux1 = max(a_bb[2], b_bb[2]); uy1 = max(a_bb[3], b_bb[3])
                    b['bbox'] = [ux0, uy0, ux1, uy1]
                    b_paths = list(b.get('glyph_path_ids', []))
                    a_paths = list(a.get('glyph_path_ids', []))
                    b['glyph_path_ids'] = b_paths + [p for p in a_paths if p not in b_paths]
                    b['num_glyphs'] = len(b['glyph_path_ids'])
                    drop_idx.add(i); break
            # Substring: short text fully contained in long text (skip identical;
            # the identical-text branch above already handled them with merge logic).
            # Drop the shorter when it's contained AND adjacent — even when
            # glyph counts tie (1==1 fragments like '4-' next to '4-6' on a
            # timeline both have 1 glyph each, neither is "bigger" so the
            # < check would pass nothing).
            if (ta != tb and ta in tb and adjacent and a_glyphs <= b_glyphs):
                drop_idx.add(i); break
            if (ta != tb and tb in ta and adjacent and b_glyphs <= a_glyphs):
                drop_idx.add(j); break
            # Word match: a is single word that's also in b (common for fragmented OCR).
            # Skip identical case (already handled by merge logic above).
            a_words = set(ta.split())
            b_words = set(tb.split())
            if ta != tb and a_words and a_words.issubset(b_words) and adjacent and a_glyphs < b_glyphs:
                drop_idx.add(i); break
            if ta != tb and b_words and b_words.issubset(a_words) and adjacent and b_glyphs < a_glyphs:
                drop_idx.add(j); break
    # Pre-pass 2: drop tiny horizontal clusters in chart-y axis label columns.
    # These are usually chart graphic artifacts (tick marks, decorations) that
    # EasyOCR mistakenly read as text (e.g. "2:", "83", "p"). A horizontal
    # cluster is considered noise if (a) it has ≤4 glyphs AND (b) its content
    # is mostly non-alphabetic gibberish AND (c) there is a vertical cluster
    # nearby (same column, within 80px y-distance).
    def _looks_like_noise(t):
        # Very short or non-alphabetic content tends to be chart artifact
        if len(t) <= 2: return True
        # contains only digits/punct/symbols
        alpha = sum(1 for ch in t if ch.isalpha())
        return alpha == 0
    for i, a in enumerate(clusters):
        if i in drop_idx: continue
        if a.get('vertical'): continue
        a_text = (a.get('text') or '').strip()
        if not a_text: continue
        a_glyphs = len(a.get('glyph_path_ids', []))
        if a_glyphs > 4: continue  # only tiny suspect clusters
        if not _looks_like_noise(a_text): continue
        a_bb = a['bbox']
        a_xc = (a_bb[0] + a_bb[2]) / 2
        for j, b in enumerate(clusters):
            if i == j or j in drop_idx: continue
            if not b.get('vertical'): continue
            b_text = (b.get('text') or '').strip()
            if not b_text: continue
            b_bb = b['bbox']
            # Same column? vertical cluster x range overlaps horizontal cluster x center
            if not (b_bb[0] - 5 <= a_xc <= b_bb[2] + 5): continue
            # Y distance within 80px
            y_dist = max(b_bb[1] - a_bb[3], a_bb[1] - b_bb[3], 0)
            if y_dist > 80: continue
            drop_idx.add(i); break
    pre_clusters = [c for i, c in enumerate(clusters) if i not in drop_idx]

    keep = []
    for c in pre_clusters:
        t = (c.get('text') or '').strip().lower()
        if not t:
            keep.append(c); continue
        cbb = c['bbox']
        ca = max(1, (cbb[2]-cbb[0])*(cbb[3]-cbb[1]))
        duplicate = False
        for k in keep:
            kt = (k.get('text') or '').strip().lower()
            if not kt: continue
            kbb = k['bbox']
            ov = _overlap(cbb, kbb)
            ka = max(1, (kbb[2]-kbb[0])*(kbb[3]-kbb[1]))
            # Duplicate if:
            #  exact same text AND any overlap (>5% of smaller)
            #  one contains the other AND >70% overlap
            min_area = min(ca, ka)
            if (t == kt and ov / min_area > 0.05) or (ov / ca > 0.7 and (t in kt or kt in t)):
                duplicate = True; break
        if not duplicate:
            keep.append(c)
    return keep


def filter_overlapping_text_cells(clusters):
    """After dedup, also adjust text cell widths so they don't overlap visually."""
    # Sort by y then x for row grouping
    sorted_c = sorted([c for c in clusters if (c.get('text') or '').strip()],
                      key=lambda c: (c['bbox'][1], c['bbox'][0]))
    # For each text cell, if its RIGHT edge exceeds the LEFT edge of the next cluster
    # on the same row (same y within 5px), shrink the width
    for i, c in enumerate(sorted_c):
        _, y0, x1, y1 = c['bbox']
        cy = (c['bbox'][1] + y1) / 2
        for j, k in enumerate(sorted_c):
            if i == j: continue
            ky = (k['bbox'][1] + k['bbox'][3]) / 2
            if abs(cy - ky) > 6: continue  # different row
            if k['bbox'][0] > c['bbox'][0] and k['bbox'][0] < x1:
                # k starts inside c's right edge — trim c
                c['bbox'] = (c['bbox'][0], c['bbox'][1], max(c['bbox'][0] + 1, k['bbox'][0] - 1), c['bbox'][3])
    return clusters


def convert(svg_path: str, png_path: str, drawio_path: str,
            cluster_cache: str | None = None,
            ocr_cache: str | None = None,
            stencil_size: int = 1000,
            font_family: str = 'DejaVu Sans',
            max_icon_dim: float = 120,
            quantize_threshold: float = 30.0,
            transparent_bg: bool = True) -> dict:
    from collections import Counter
    from PIL import Image
    import numpy as np

    print('[1/9] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'      {W}x{H}, {len(paths)} paths')

    # Drop canvas background
    if transparent_bg:
        bg_paths = [p for p in paths if is_full_canvas_background(p, W, H)]
        paths = [p for p in paths if not is_full_canvas_background(p, W, H)]
        print(f'[2/9] dropped {len(bg_paths)} canvas background paths')

    print('[3/9] text clustering...')
    if cluster_cache and Path(cluster_cache).exists():
        saved = json.loads(Path(cluster_cache).read_text())
        clusters = saved
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
    else:
        candidates = identify_glyph_candidates(paths)
        clusters = cluster_glyphs(candidates, W, H)
    print(f'      {len(clusters)} clusters before dedup')

    # Dedup overlapping text clusters
    # Need OCR first to have text
    print('[4/9] OCR...')
    full_ocr = None
    candidates = []
    if cluster_cache:
        cc = Path(cluster_cache); candidates.append(cc.parent / 'ocr.json')
    candidates.append(Path(svg_path).parent / 'ocr.json')
    for cand in candidates:
        if cand.exists():
            full_ocr = json.loads(cand.read_text()); break
    ocr_clusters(clusters, png_path, cache_path=ocr_cache, full_image_ocr=full_ocr)
    ok = sum(1 for c in clusters if c.get('text'))
    print(f'      {ok}/{len(clusters)} with text')

    # Dedup
    before = len(clusters)
    clusters = dedupe_text_clusters(clusters)
    print(f'[5/9] deduped: {before} → {len(clusters)} clusters (dropped {before - len(clusters)} overlapping)')
    clusters = filter_overlapping_text_cells(clusters)

    consumed = set()
    for c in clusters:
        ng = len(c.get('glyph_path_ids', []))
        if c.get('text') or ng >= 4:
            consumed.update(c['glyph_path_ids'])

    text_bboxes = [c['bbox'] for c in clusters if c.get('text')]
    def _in_text(sbb, thresh=0.45):
        sx0,sy0,sx1,sy1 = sbb; sa = max(1,(sx1-sx0)*(sy1-sy0))
        for bx0,by0,bx1,by1 in text_bboxes:
            ix0=max(sx0, bx0-3); iy0=max(sy0, by0-2)
            ix1=min(sx1, bx1+3); iy1=min(sy1, by1+6)
            if ix1<=ix0 or iy1<=iy0: continue
            if (ix1-ix0)*(iy1-iy0)/sa >= thresh: return True
        return False

    remaining = []
    for p in paths:
        if p['idx'] in consumed: continue
        kept = []
        for sub in p['subpaths']:
            x0, y0, w, h = subpath_bbox(sub)
            if w<100 and h<36 and _in_text((x0,y0,x0+w,y0+h)): continue
            kept.append(sub)
        if not kept: continue
        flat = [s for sub in kept for s in sub]
        x0, y0, w, h = path_bbox(flat)
        if w<0.1 or h<0.1: continue
        remaining.append({
            'idx': p['idx'], 'fill': p['fill'], 'fill_op': p['fill_op'],
            'expanded': flat, 'bbox': (x0, y0, x0+w, y0+h), 'dark': p['dark'],
        })

    print(f'[6/9] clustering {len(remaining)} remaining into icons...')
    icon_groups = cluster_icons(remaining, W, H,
                                max_icon_dim=max_icon_dim,
                                min_paths_per_icon=2, eps=2.0)
    n_icon_before = sum(1 for g in icon_groups if g.get('is_icon'))

    path_by_idx = {p['idx']: p for p in remaining}
    icon_groups, absorbed = absorb_singletons(icon_groups, path_by_idx, max_absorb_dim=40)
    n_icon_after = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'      {n_icon_after} icons ({absorbed} singletons absorbed), '
          f'{len(icon_groups)-n_icon_after} remaining singletons')

    print('[7/9] emitting cells...')
    cells = []
    cid = 100
    icon_count = 0; singleton_count = 0

    # Invisible bounding rect to preserve canvas dimensions when bg is transparent
    if transparent_bg:
        cells.append(
            f'<mxCell id="{cid}" value="" '
            f'style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="0" y="0" width="{W}" height="{H}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    for g in icon_groups:
        path_ids = sorted(g['path_ids'])
        if g.get('is_icon') and len(path_ids) >= 2:
            gp = [path_by_idx[pid] for pid in path_ids]
            bbox = g['bbox']
            x0,y0,x1,y1 = bbox; w=x1-x0; h=y1-y0
            svg, nc = quantized_icon_svg(gp, bbox, threshold=quantize_threshold)
            style = svg_to_image_cell_style(svg)
            cells.append(
                f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
                f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                f'</mxCell>'
            )
            icon_count += 1
            cid += 1
        else:
            for pid in path_ids:
                p = path_by_idx[pid]
                x0,y0,x1,y1 = p['bbox']; w=x1-x0; h=y1-y0
                # Skip large white rects here too (belt-and-suspenders)
                if transparent_bg and p['fill'].lower() in ('#ffffff','#fff') and w * h > 400_000:
                    continue
                stencil_xml = segments_to_stencil(p['expanded'], (x0,y0,w,h), stencil_size=stencil_size)
                stencil_b64 = encode_stencil(stencil_xml)
                style_parts = [f'shape=stencil({stencil_b64})',
                               f'fillColor={p["fill"]}',
                               'strokeColor=none', 'html=1']
                if p['fill_op']<1.0:
                    style_parts.append(f'opacity={int(p["fill_op"]*100)}')
                style = ';'.join(style_parts) + ';'
                cells.append(
                    f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
                    f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                    f'</mxCell>'
                )
                singleton_count += 1
                cid += 1

    all_paths_by_idx = {p['idx']: p for p in paths}
    text_count = 0
    for c in clusters:
        text = (c.get('text') or '').strip()
        if not text: continue
        x0,y0,x1,y1 = c['bbox']; w=x1-x0; h=y1-y0
        if w<2 or h<2: continue
        if c.get('glyphs'):
            fills = [gg['path']['fill'] for gg in c['glyphs']]
        else:
            fills = [all_paths_by_idx[pid]['fill']
                     for pid in c.get('glyph_path_ids', []) if pid in all_paths_by_idx]
        color = Counter(fills).most_common(1)[0][0] if fills else '#333333'
        fs = c['font_size']
        style_parts = ['text','html=1','strokeColor=none','fillColor=none',
                       'align=left','verticalAlign=middle',
                       'whiteSpace=nowrap','rounded=0',
                       f'fontFamily={font_family}', f'fontSize={fs}',
                       f'fontColor={color}']
        bits = 0
        if c.get('bold'): bits |= 1
        if bits: style_parts.append(f'fontStyle={bits}')
        pad_x = max(1, int(fs*0.1)); pad_y = 1
        gx = x0-pad_x; gy = y0-pad_y; gw = w+2*pad_x; gh = h+2*pad_y
        val = (text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;'))
        style = ';'.join(style_parts) + ';'
        cells.append(
            f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" as="geometry"/>'
            f'</mxCell>'
        )
        text_count += 1
        cid += 1

    body = '\n        '.join(cells)
    # Use background="none" on mxGraphModel for transparent export
    drawio = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-04-24T00:00:00.000Z" '
        'agent="svg_to_drawio_v12.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v12" name="traced">\n'
        f'    <mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{W}" pageHeight="{H}" math="0" shadow="0" background="none">\n'
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
    print(f'[8/9] wrote {drawio_path}  ({len(cells)} cells, {os.path.getsize(drawio_path)//1024}KB)')
    print(f'[9/9] stats: {icon_count} icons, {singleton_count} singletons, {text_count} text')
    return {'icons': icon_count, 'singletons': singleton_count,
            'text': text_count, 'total_cells': len(cells),
            'dedup_dropped': before - len(clusters),
            'absorbed': absorbed}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('png')
    ap.add_argument('-o','--output', default=None)
    ap.add_argument('--cluster-cache', default=None)
    ap.add_argument('--ocr-cache', default=None)
    ap.add_argument('--font-family', default='DejaVu Sans')
    ap.add_argument('--max-icon-dim', type=float, default=120)
    ap.add_argument('--quantize', type=float, default=30.0)
    ap.add_argument('--keep-bg', action='store_true')
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert(args.svg, args.png, out,
            cluster_cache=args.cluster_cache,
            ocr_cache=args.ocr_cache,
            font_family=args.font_family,
            max_icon_dim=args.max_icon_dim,
            quantize_threshold=args.quantize,
            transparent_bg=not args.keep_bg)
