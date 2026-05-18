"""svg_to_drawio_v7.py — v6 + icon merge into atomic SVG image cells.

Each icon group becomes ONE drawio mxCell with a self-contained SVG data URI.
This makes each icon physically indivisible in drawio: user clicks → one cell
selected; drag → whole icon moves; impossible to "split" it into sub-paths.

Non-icon singleton paths remain as individual stencil cells (arrows, connectors,
single background rects — these SHOULD be individually editable).

Text cells remain as drawio native text (unchanged from v5/v6).
"""
from __future__ import annotations

import os, re, sys, json, base64, zlib
from pathlib import Path
from xml.etree import ElementTree as ET
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


# ----------------------------------------------------------------------------
# Serialize segments back to SVG path d string
# ----------------------------------------------------------------------------

def segments_to_svg_d(segments) -> str:
    """Convert [(cmd, *args), ...] back to an SVG path d attribute string."""
    parts = []
    for seg in segments:
        cmd = seg[0]
        if cmd == 'M':
            parts.append(f'M{seg[1]:.3f} {seg[2]:.3f}')
        elif cmd == 'L':
            parts.append(f'L{seg[1]:.3f} {seg[2]:.3f}')
        elif cmd == 'C':
            parts.append(f'C{seg[1]:.3f} {seg[2]:.3f} {seg[3]:.3f} {seg[4]:.3f} {seg[5]:.3f} {seg[6]:.3f}')
        elif cmd == 'Q':
            parts.append(f'Q{seg[1]:.3f} {seg[2]:.3f} {seg[3]:.3f} {seg[4]:.3f}')
        elif cmd == 'Z':
            parts.append('Z')
    return ' '.join(parts)


def icon_to_svg(icon_paths, bbox):
    """Build a self-contained SVG for one icon group.
    Coordinates normalized so bbox origin becomes (0,0).
    Returns SVG string.
    """
    x0, y0, x1, y1 = bbox
    W = x1 - x0; H = y1 - y0
    paths_svg = []
    for p in icon_paths:
        # Shift each segment's coords by (-x0, -y0)
        shifted = []
        for seg in p['expanded']:
            cmd = seg[0]
            if cmd == 'M': shifted.append(('M', seg[1] - x0, seg[2] - y0))
            elif cmd == 'L': shifted.append(('L', seg[1] - x0, seg[2] - y0))
            elif cmd == 'C':
                shifted.append(('C',
                    seg[1] - x0, seg[2] - y0,
                    seg[3] - x0, seg[4] - y0,
                    seg[5] - x0, seg[6] - y0))
            elif cmd == 'Q':
                shifted.append(('Q',
                    seg[1] - x0, seg[2] - y0,
                    seg[3] - x0, seg[4] - y0))
            elif cmd == 'Z': shifted.append(('Z',))
        d = segments_to_svg_d(shifted)
        fo = p.get('fill_op', 1.0)
        fa = f' fill-opacity="{fo}"' if fo < 1.0 else ''
        paths_svg.append(
            f'<path d="{d}" fill="{p["fill"]}" fill-rule="evenodd"{fa}/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{W:.2f}" height="{H:.2f}" '
        f'viewBox="0 0 {W:.2f} {H:.2f}">'
        + ''.join(paths_svg)
        + '</svg>'
    )
    return svg


def svg_to_image_cell_style(svg: str) -> str:
    """Build drawio cell style with embedded SVG data URI (no base64 — cleaner
    and editable). drawio splits style on ';', so we must escape the data URI
    accordingly. drawio supports data URIs like `data:image/svg+xml,<url-encoded>`
    without ';base64,'.
    """
    # URL-encode the SVG; use quote to keep it safe across style parsing
    encoded = quote(svg, safe='')
    # Build style
    # shape=image;aspect=fixed;imageAspect=0;image=data:image/svg+xml,...
    return (f'shape=image;html=1;imageAspect=0;'
            f'image=data:image/svg+xml,{encoded};')


# ----------------------------------------------------------------------------
# Main conversion (v7)
# ----------------------------------------------------------------------------

def convert(svg_path: str, png_path: str, drawio_path: str,
            cluster_cache: str | None = None,
            ocr_cache: str | None = None,
            stencil_size: int = 1000,
            font_family: str = 'DejaVu Sans',
            max_icon_dim: float = 120) -> dict:
    import numpy as np
    from PIL import Image
    from collections import Counter

    print('[1/7] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'      {W}x{H}, {len(paths)} paths')

    print('[2/7] text clustering...')
    candidates = identify_glyph_candidates(paths)
    if cluster_cache and Path(cluster_cache).exists():
        saved = json.loads(Path(cluster_cache).read_text())
        clusters = saved
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
    else:
        clusters = cluster_glyphs(candidates, W, H)
        if cluster_cache:
            dump = [{'bbox': list(c['bbox']), 'font_size': c['font_size'],
                     'bold': c['bold'], 'ink_ratio': c['ink_ratio'],
                     'glyph_path_ids': sorted(c['glyph_path_ids']),
                     'num_glyphs': len(c['glyphs'])} for c in clusters]
            Path(cluster_cache).write_text(json.dumps(dump, indent=2))
    print(f'      {len(clusters)} clusters')

    print('[3/7] OCR...')
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

    # Consumed by text
    consumed = set()
    for c in clusters:
        ng = len(c.get('glyph_path_ids', []))
        if c.get('text') or ng >= 4:
            consumed.update(c['glyph_path_ids'])
    print(f'[4/7] consumed {len(consumed)} text-glyph paths')

    # Filter remaining paths: drop canvas holes at text positions
    text_bboxes = [c['bbox'] for c in clusters if c.get('text')]
    def _in_text(sbb, thresh=0.45):
        sx0, sy0, sx1, sy1 = sbb
        sa = max(1, (sx1-sx0)*(sy1-sy0))
        for bx0,by0,bx1,by1 in text_bboxes:
            ix0 = max(sx0, bx0-3); iy0 = max(sy0, by0-2)
            ix1 = min(sx1, bx1+3); iy1 = min(sy1, by1+6)
            if ix1<=ix0 or iy1<=iy0: continue
            if (ix1-ix0)*(iy1-iy0)/sa >= thresh: return True
        return False

    remaining = []
    for p in paths:
        if p['idx'] in consumed: continue
        kept_subs = []
        for sub in p['subpaths']:
            x0, y0, w, h = subpath_bbox(sub)
            if w<100 and h<36 and _in_text((x0,y0,x0+w,y0+h)):
                continue
            kept_subs.append(sub)
        if not kept_subs: continue
        flat = [s for sub in kept_subs for s in sub]
        x0, y0, w, h = path_bbox(flat)
        if w<0.1 or h<0.1: continue
        remaining.append({
            'idx': p['idx'], 'fill': p['fill'], 'fill_op': p['fill_op'],
            'expanded': flat, 'bbox': (x0, y0, x0+w, y0+h),
            'dark': p['dark'],
        })

    print(f'[5/7] clustering {len(remaining)} remaining paths into icons...')
    icon_groups = cluster_icons(remaining, W, H,
                                max_icon_dim=max_icon_dim,
                                min_paths_per_icon=2, eps=2.0)
    n_icon = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'      {n_icon} icon groups, {len(icon_groups)-n_icon} singletons')

    print('[6/7] emitting cells...')
    path_by_idx = {p['idx']: p for p in remaining}
    cells = []
    cid = 100

    # Each icon group → ONE image cell with embedded SVG
    icon_cells_count = 0
    singleton_cells_count = 0
    for g in icon_groups:
        path_ids = sorted(g['path_ids'])
        if g.get('is_icon') and len(path_ids) >= 2:
            gp = [path_by_idx[pid] for pid in path_ids]
            bbox = g['bbox']
            svg_icon = icon_to_svg(gp, bbox)
            style = svg_to_image_cell_style(svg_icon)
            x0, y0, x1, y1 = bbox
            w = x1-x0; h = y1-y0
            cells.append(
                f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
                f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                f'</mxCell>'
            )
            icon_cells_count += 1
            cid += 1
        else:
            # Singleton — emit as stencil
            for pid in path_ids:
                p = path_by_idx[pid]
                x0, y0, x1, y1 = p['bbox']
                w = x1-x0; h = y1-y0
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
                singleton_cells_count += 1
                cid += 1

    # Text cells
    all_paths_by_idx = {p['idx']: p for p in paths}
    for c in clusters:
        text = (c.get('text') or '').strip()
        if not text: continue
        x0,y0,x1,y1 = c['bbox']
        w = x1-x0; h = y1-y0
        if w<2 or h<2: continue
        if c.get('glyphs'):
            fills = [gg['path']['fill'] for gg in c['glyphs']]
        else:
            fills = [all_paths_by_idx[pid]['fill']
                     for pid in c.get('glyph_path_ids', [])
                     if pid in all_paths_by_idx]
        color = Counter(fills).most_common(1)[0][0] if fills else '#333333'
        fs = c['font_size']
        style_parts = ['text','html=1','strokeColor=none','fillColor=none',
                       'align=left','verticalAlign=middle',
                       'whiteSpace=nowrap','rounded=0',
                       f'fontFamily={font_family}',
                       f'fontSize={fs}',
                       f'fontColor={color}']
        bits = 0
        if c.get('bold'): bits |= 1
        if bits: style_parts.append(f'fontStyle={bits}')
        pad_x = max(1, int(fs*0.1)); pad_y = 1
        gx = x0 - pad_x; gy = y0 - pad_y
        gw = w + 2*pad_x; gh = h + 2*pad_y
        val = (text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                   .replace('"','&quot;'))
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
        'agent="svg_to_drawio_v7.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v7" name="traced">\n'
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
    sz = os.path.getsize(drawio_path)
    print(f'[7/7] wrote {drawio_path}  ({len(cells)} cells, {sz//1024}KB)')
    return {
        'paths': len(paths),
        'icons_atomic': icon_cells_count,
        'singleton_stencils': singleton_cells_count,
        'text_cells': ok,
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
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert(args.svg, args.png, out,
            cluster_cache=args.cluster_cache,
            ocr_cache=args.ocr_cache,
            font_family=args.font_family,
            max_icon_dim=args.max_icon_dim)
