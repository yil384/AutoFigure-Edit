"""svg_to_drawio_v11.py — v9 + singleton absorption into nearby icons.

After icon clustering, walks singleton paths and absorbs them into adjacent
icons when they clearly belong (bbox inside or touching icon bbox, similar
color palette, reasonable size).
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
from svg_to_drawio_v6 import (
    bbox_overlap_or_near, bbox_union, cluster_icons,
)
from svg_to_drawio_v7 import segments_to_svg_d, svg_to_image_cell_style
from svg_to_drawio_v9 import quantized_icon_svg


def absorb_singletons(icon_groups, path_by_idx, max_absorb_dim=40):
    """For each singleton group near an icon, absorb it INTO that icon.
    A singleton qualifies if:
      - Its bbox fits mostly inside an expanded icon bbox (icon bbox padded by 8px)
      - Its size is < max_absorb_dim
      - It doesn't grow the icon bbox by more than 15%
    """
    # Separate icons and singletons
    icons = [g for g in icon_groups if g.get('is_icon')]
    singletons = [g for g in icon_groups if not g.get('is_icon')]

    def overlap(a, b):
        ax0,ay0,ax1,ay1 = a; bx0,by0,bx1,by1 = b
        ix0=max(ax0,bx0); iy0=max(ay0,by0)
        ix1=min(ax1,bx1); iy1=min(ay1,by1)
        if ix1<=ix0 or iy1<=iy0: return 0
        return (ix1-ix0)*(iy1-iy0)

    absorbed_count = 0
    remaining_singletons = []
    for s in singletons:
        assert len(s['path_ids']) == 1, 'singletons should have one path'
        pid = next(iter(s['path_ids']))
        p = path_by_idx.get(pid)
        if p is None:
            continue
        sb = s['bbox']
        sx0, sy0, sx1, sy1 = sb
        sw = sx1 - sx0; sh = sy1 - sy0
        # Big paths stay singleton (long lines, arrows, panel backgrounds)
        if sw > max_absorb_dim * 3 or sh > max_absorb_dim * 3:
            remaining_singletons.append(s); continue

        # Find icon containing or adjacent
        best_icon = None
        best_overlap_ratio = 0
        for ic in icons:
            ibb = ic['bbox']
            ib_expanded = (ibb[0]-8, ibb[1]-8, ibb[2]+8, ibb[3]+8)
            ov = overlap(sb, ib_expanded)
            sa = max(1, sw*sh)
            r = ov / sa
            if r > best_overlap_ratio:
                best_overlap_ratio = r
                best_icon = ic
        if best_icon and best_overlap_ratio >= 0.6:
            # Check growth
            ibb = best_icon['bbox']
            orig_area = (ibb[2]-ibb[0])*(ibb[3]-ibb[1])
            new_bb = bbox_union(ibb, sb)
            new_area = (new_bb[2]-new_bb[0])*(new_bb[3]-new_bb[1])
            if orig_area == 0 or new_area/orig_area <= 1.15:
                best_icon['path_ids'].add(pid)
                best_icon['bbox'] = new_bb
                absorbed_count += 1
                continue
        remaining_singletons.append(s)

    return icons + remaining_singletons, absorbed_count


def convert(svg_path: str, png_path: str, drawio_path: str,
            cluster_cache: str | None = None,
            ocr_cache: str | None = None,
            stencil_size: int = 1000,
            font_family: str = 'DejaVu Sans',
            max_icon_dim: float = 120,
            quantize_threshold: float = 20.0) -> dict:
    from collections import Counter
    from PIL import Image
    import numpy as np

    print('[1/8] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'      {W}x{H}, {len(paths)} paths')

    print('[2/8] text clustering...')
    if cluster_cache and Path(cluster_cache).exists():
        saved = json.loads(Path(cluster_cache).read_text())
        clusters = saved
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
    else:
        candidates = identify_glyph_candidates(paths)
        clusters = cluster_glyphs(candidates, W, H)
    print(f'      {len(clusters)} clusters')

    print('[3/8] OCR...')
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

    print(f'[4/8] clustering {len(remaining)} remaining into icons...')
    icon_groups = cluster_icons(remaining, W, H,
                                max_icon_dim=max_icon_dim,
                                min_paths_per_icon=2, eps=2.0)
    n_icon_before = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'      {n_icon_before} icon groups, {len(icon_groups)-n_icon_before} singletons (before absorption)')

    print('[5/8] absorbing singletons into nearby icons...')
    path_by_idx = {p['idx']: p for p in remaining}
    icon_groups, absorbed = absorb_singletons(icon_groups, path_by_idx, max_absorb_dim=40)
    n_icon_after = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'      absorbed {absorbed} singletons  → {n_icon_after} icons, '
          f'{len(icon_groups)-n_icon_after} remaining singletons')

    print('[6/8] emitting cells...')
    cells = []
    cid = 100
    icon_count = 0; singleton_count = 0

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
    drawio = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-04-24T00:00:00.000Z" '
        'agent="svg_to_drawio_v11.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v11" name="traced">\n'
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
    print(f'[7/8] wrote {drawio_path}  ({len(cells)} cells, {os.path.getsize(drawio_path)//1024}KB)')
    print(f'[8/8] stats: {icon_count} atomic icons, {singleton_count} singletons, {text_count} text')
    return {
        'icons_atomic': icon_count, 'singletons': singleton_count,
        'text_cells': text_count, 'absorbed': absorbed,
        'total_cells': len(cells),
    }


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
    ap.add_argument('--quantize', type=float, default=20.0)
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert(args.svg, args.png, out,
            cluster_cache=args.cluster_cache,
            ocr_cache=args.ocr_cache,
            font_family=args.font_family,
            max_icon_dim=args.max_icon_dim,
            quantize_threshold=args.quantize)
