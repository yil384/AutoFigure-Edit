"""svg_to_drawio_v14.py — v12/v13 + correct z-order + solid panel backgrounds.

Fixes per user:
 - Panel background paths (light-color large rects) had evenodd holes carved at
   icon positions. Result: moving an icon revealed empty holes in the bg.
 - Z-order in v13 emitted panel bg rects AFTER icons → bg drawn ON TOP of icons.

v14:
 - Detect "panel background" paths: large + light fill.
 - Drop ALL inner subpaths from panel bg paths (no holes, solid colored block).
 - Emit panel backgrounds FIRST (z=0), then small singletons, then icons, then text.
"""
from __future__ import annotations

import os, re, sys, json
from pathlib import Path

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
from svg_to_drawio_v12 import (
    is_full_canvas_background, dedupe_text_clusters, filter_overlapping_text_cells,
)


def is_panel_background(p, W, H) -> bool:
    """True if this path is a panel-shaped background (large rounded/plain rect).

    Two cases:
      A. Large light-colored panel: ≥ 80×80, avg RGB > 200 (typical pastel bg)
      B. Smaller mid-tone rounded rect: ≥ 80×30, avg RGB > 130, AND it's actually
         a rounded rectangle outline (heuristic: ratio area / bbox > 0.7)
    Both excluded from icon clustering so they render as bottom-layer stencils.
    """
    rgb = _parse_hex(p['fill'])
    if rgb is None: return False
    avg = sum(rgb) / 3
    x0, y0, x1, y1 = p['bbox']
    w = x1 - x0; h = y1 - y0
    if w > W * 0.95 and h > H * 0.95: return False
    # Case A
    if avg > 200 and w >= 80 and h >= 80:
        return True
    # Case B: pill-shaped or rounded rect of medium tone
    if avg > 130 and w >= 80 and h >= 25 and (w * h) >= 2000:
        # Heuristic: dense polygon area suggests rect-like shape
        # Use simple shoelace on first subpath (outer)
        first_sub = None
        for seg in p['expanded']:
            if seg[0] == 'M':
                first_sub = []
            if first_sub is not None:
                if seg[0] in ('M', 'L'):
                    first_sub.append((seg[1], seg[2]))
                elif seg[0] == 'C': first_sub.append((seg[5], seg[6]))
                elif seg[0] == 'Q': first_sub.append((seg[3], seg[4]))
                elif seg[0] == 'Z':
                    break
        if first_sub and len(first_sub) >= 3:
            n = len(first_sub)
            area = 0.0
            for i in range(n):
                x_i, y_i = first_sub[i]
                x_j, y_j = first_sub[(i + 1) % n]
                area += x_i * y_j - x_j * y_i
            area = abs(area) / 2
            bbox_area = w * h
            if area / max(1, bbox_area) > 0.7:
                return True
    return False


def keep_outer_subpath_only(p):
    """Return new path with only the largest subpath (outer boundary), drops
    all inner subpaths (holes).
    """
    subs = split_subpaths(p['expanded'])
    if not subs:
        return p
    # Find outer = largest area subpath
    best = max(subs, key=lambda s: max(1, (lambda b: (b[2]*b[3]))(path_bbox(s))))
    new_p = dict(p)
    new_p['expanded'] = best
    x0, y0, w, h = path_bbox(best)
    new_p['bbox'] = (x0, y0, x0 + w, y0 + h)
    return new_p


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

    print('[1/10] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'       {W}x{H}, {len(paths)} paths')

    if transparent_bg:
        before = len(paths)
        paths = [p for p in paths if not is_full_canvas_background(p, W, H)]
        print(f'[2/10] dropped {before - len(paths)} canvas bg paths')

    # Identify panel backgrounds (large light rects with holes)
    panel_bgs = [p for p in paths if is_panel_background(p, W, H)]
    panel_bg_idxs = set(p['idx'] for p in panel_bgs)
    print(f'[3/10] found {len(panel_bgs)} panel background paths')

    # Drop holes (inner subpaths) from each panel bg → solid blocks
    panel_bgs_solid = [keep_outer_subpath_only(p) for p in panel_bgs]

    # Remaining content paths (everything except canvas bg AND panel bg)
    content_paths = [p for p in paths if p['idx'] not in panel_bg_idxs]

    print('[4/10] text clustering...')
    if cluster_cache and Path(cluster_cache).exists():
        saved = json.loads(Path(cluster_cache).read_text())
        clusters = saved
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
    else:
        candidates = identify_glyph_candidates(content_paths)
        clusters = cluster_glyphs(candidates, W, H)
        if cluster_cache:
            dump = [{'bbox': list(c['bbox']), 'font_size': c['font_size'],
                     'bold': c['bold'], 'ink_ratio': c['ink_ratio'],
                     'glyph_path_ids': sorted(c['glyph_path_ids']),
                     'num_glyphs': len(c['glyphs']),
                     'vertical': c.get('vertical', False)}
                    for c in clusters]
            Path(cluster_cache).write_text(json.dumps(dump, indent=2))
    nv = sum(1 for c in clusters if c.get('vertical'))
    print(f'       {len(clusters)} clusters ({nv} vertical)')

    print('[5/10] OCR...')
    full_ocr = None
    # Look for a pre-computed full-image OCR cache next to the cluster cache
    # (or alongside the input SVG). Fallback: ocr_clusters() will run EasyOCR
    # on the full image if not found.
    full_ocr_candidates = []
    if cluster_cache:
        cc = Path(cluster_cache)
        full_ocr_candidates.append(cc.parent / f'{cc.stem.replace(".clusters","")}.full_ocr.json')
        full_ocr_candidates.append(cc.parent / 'ocr.json')
    full_ocr_candidates.append(Path(svg_path).parent / 'ocr.json')
    for cand in full_ocr_candidates:
        if cand.exists():
            full_ocr = json.loads(cand.read_text())
            print(f'       loaded {len(full_ocr)} full-image OCR boxes from {cand}')
            break
    ocr_clusters(clusters, png_path, cache_path=ocr_cache, full_image_ocr=full_ocr)
    ok = sum(1 for c in clusters if c.get('text'))
    print(f'       {ok}/{len(clusters)} with text')

    # Capture which paths belong to ALL clusters (incl. deduped) BEFORE dropping
    # so deduped clusters' glyph paths don't reappear as raw path stencils.
    consumed = set()
    for c in clusters:
        ng = len(c.get('glyph_path_ids', []))
        if c.get('text') or ng >= 4:
            consumed.update(c['glyph_path_ids'])

    before = len(clusters)
    clusters = dedupe_text_clusters(clusters)
    print(f'[6/10] deduped: {before} → {len(clusters)} clusters')
    clusters = filter_overlapping_text_cells(clusters)

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
    for p in content_paths:
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

    print(f'[7/10] clustering {len(remaining)} into icons...')
    icon_groups = cluster_icons(remaining, W, H,
                                max_icon_dim=max_icon_dim,
                                min_paths_per_icon=2, eps=2.0)
    path_by_idx = {p['idx']: p for p in remaining}
    icon_groups, absorbed = absorb_singletons(icon_groups, path_by_idx, max_absorb_dim=40)
    n_icons = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'       {n_icons} icons, {len(icon_groups)-n_icons} singletons')

    print('[8/10] emitting cells in correct z-order...')
    cells = []
    cid = 100

    # Layer 0: invisible locked canvas rect (preserves page dimensions, can't
    # be selected/moved/edited/deleted by user)
    if transparent_bg:
        cells.append(
            f'<mxCell id="{cid}" value="" '
            f'style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;'
            f'locked=1;movable=0;editable=0;deletable=0;resizable=0;rotatable=0;'
            f'connectable=0;selectable=0;" '
            f'vertex="1" parent="1">'
            f'<mxGeometry x="0" y="0" width="{W}" height="{H}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    # Layer 1: solid panel backgrounds (deepest z)
    panel_count = 0
    for p in panel_bgs_solid:
        x0, y0, x1, y1 = p['bbox']
        w = x1-x0; h = y1-y0
        if w < 0.1 or h < 0.1: continue
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
        panel_count += 1
        cid += 1

    # Layer 2: large stencil singletons (lines, arrows, structural shapes inside panels)
    # then icons, then small singletons
    # For correct z, emit icons LAST among singletons+icons so they overlay panels and lines
    sing_count = 0; icon_count = 0
    for g in icon_groups:
        if not (g.get('is_icon') and len(g['path_ids']) >= 2):
            for pid in sorted(g['path_ids']):
                p = path_by_idx[pid]
                x0,y0,x1,y1 = p['bbox']; w=x1-x0; h=y1-y0
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
                sing_count += 1
                cid += 1

    # Then icons (atomic SVG image cells) on top
    for g in icon_groups:
        if g.get('is_icon') and len(g['path_ids']) >= 2:
            gp = [path_by_idx[pid] for pid in sorted(g['path_ids'])]
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

    # Layer 3: text cells (top)
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
        is_vertical = c.get('vertical', False)
        style_parts = ['text','html=1','strokeColor=none','fillColor=none',
                       'align=left','verticalAlign=middle',
                       'whiteSpace=nowrap','rounded=0',
                       f'fontFamily={font_family}', f'fontSize={fs}',
                       f'fontColor={color}']
        bits = 0
        if c.get('bold'): bits |= 1
        if bits: style_parts.append(f'fontStyle={bits}')
        if is_vertical:
            # drawio rotation=-90 (counterclockwise), text reads bottom-to-top
            style_parts.append('rotation=-90')
            # For vertical, swap geometry: drawio rotates around center, so use
            # cluster bbox as-is but font size = column width
            pad = max(1, int(fs * 0.1))
            gx = x0 - pad; gy = y0 - pad
            gw = w + 2*pad; gh = h + 2*pad
        else:
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
    drawio = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-04-25T00:00:00.000Z" '
        'agent="svg_to_drawio_v14.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v14" name="traced">\n'
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
    print(f'[9/10] wrote {drawio_path}  ({len(cells)} cells, {os.path.getsize(drawio_path)//1024}KB)')
    print(f'[10/10] z-order: {panel_count} panel-bgs (bottom) → {sing_count} singletons → '
          f'{icon_count} icons → {text_count} text (top)')
    return {'panels': panel_count, 'singletons': sing_count,
            'icons': icon_count, 'text': text_count, 'total': len(cells)}


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
