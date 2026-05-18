"""svg_to_drawio_v9.py — v7 base + icon color quantization to crispify blurry icons.

For each atomic icon SVG (emitted in v7), cluster the paths by fill color:
  - Paths with similar colors (ΔE76 distance < threshold) are merged into a
    single color group
  - Each group's paths are re-emitted with the GROUP's averaged color
Result: fewer distinct colors per icon → crisper visual, less blur.

Also supports per-icon LLM decisions:
  - "keep" → render original icon as-is
  - "emoji" → replace with Noto Color Emoji text cell (only if user confirms)
  - "drop" → remove cell entirely
  - "simplify" → color-quantize (default)
"""
from __future__ import annotations

import os, re, sys, json, base64, zlib
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


# ----------------------------------------------------------------------------
# Color quantization on a path group
# ----------------------------------------------------------------------------

def color_distance(c1, c2):
    """ΔE76 simple approximation via sqrt sum of squared differences in RGB."""
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5


def quantize_colors(paths, threshold: float = 18.0, min_cluster_size: int = 1):
    """Greedy cluster of path fill colors.
    Returns list of {'fill': mean_hex, 'paths': [...] } clusters.
    """
    # Parse fills to RGB
    pts = []
    for p in paths:
        rgb = _parse_hex(p['fill'])
        if rgb is None: continue
        pts.append({'p': p, 'rgb': rgb, 'area': (p['bbox'][2]-p['bbox'][0])*(p['bbox'][3]-p['bbox'][1])})
    # Sort by area desc (larger paths seed clusters)
    pts.sort(key=lambda t: -t['area'])

    clusters = []  # each: {'rgb_sum': [r,g,b], 'count':N, 'weight':area, 'paths':[]}
    for entry in pts:
        assigned = False
        for cl in clusters:
            mean = [cl['rgb_sum'][i] / max(1, cl['weight']) for i in range(3)]
            if color_distance(entry['rgb'], mean) <= threshold:
                w = entry['area']
                for i in range(3):
                    cl['rgb_sum'][i] += entry['rgb'][i] * w
                cl['weight'] += w
                cl['paths'].append(entry['p'])
                cl['count'] += 1
                assigned = True
                break
        if not assigned:
            w = max(1, entry['area'])
            clusters.append({
                'rgb_sum': [entry['rgb'][i] * w for i in range(3)],
                'weight': w, 'count': 1, 'paths': [entry['p']],
            })
    # Compute final mean colors
    out = []
    for cl in clusters:
        mean = [int(round(cl['rgb_sum'][i] / max(1, cl['weight']))) for i in range(3)]
        r, g, b = [max(0, min(255, v)) for v in mean]
        out.append({
            'fill': f'#{r:02X}{g:02X}{b:02X}',
            'paths': cl['paths'],
            'count': cl['count'],
        })
    return out


def quantized_icon_svg(icon_paths, bbox, threshold=18.0):
    """Build a color-quantized SVG for an icon group."""
    x0, y0, x1, y1 = bbox
    W = x1 - x0; H = y1 - y0
    clusters = quantize_colors(icon_paths, threshold=threshold)
    paths_svg = []
    for cl in clusters:
        # Concatenate d strings from all paths in this color cluster
        ds = []
        for p in cl['paths']:
            shifted = []
            for seg in p['expanded']:
                cmd = seg[0]
                if cmd == 'M': shifted.append(('M', seg[1]-x0, seg[2]-y0))
                elif cmd == 'L': shifted.append(('L', seg[1]-x0, seg[2]-y0))
                elif cmd == 'C':
                    shifted.append(('C', seg[1]-x0, seg[2]-y0,
                                    seg[3]-x0, seg[4]-y0,
                                    seg[5]-x0, seg[6]-y0))
                elif cmd == 'Q':
                    shifted.append(('Q', seg[1]-x0, seg[2]-y0,
                                    seg[3]-x0, seg[4]-y0))
                elif cmd == 'Z': shifted.append(('Z',))
            ds.append(segments_to_svg_d(shifted))
        merged_d = ' '.join(ds)
        paths_svg.append(f'<path d="{merged_d}" fill="{cl["fill"]}" fill-rule="evenodd"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" height="{H:.2f}" '
           f'viewBox="0 0 {W:.2f} {H:.2f}">' + ''.join(paths_svg) + '</svg>')
    return svg, len(clusters)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def convert(svg_path: str, png_path: str, drawio_path: str,
            cluster_cache: str | None = None,
            ocr_cache: str | None = None,
            icon_plan: str | None = None,
            stencil_size: int = 1000,
            font_family: str = 'DejaVu Sans',
            max_icon_dim: float = 120,
            quantize_threshold: float = 18.0) -> dict:
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
        if cluster_cache:
            dump = [{'bbox': list(c['bbox']), 'font_size': c['font_size'],
                     'bold': c['bold'], 'ink_ratio': c['ink_ratio'],
                     'glyph_path_ids': sorted(c['glyph_path_ids']),
                     'num_glyphs': len(c['glyphs'])} for c in clusters]
            Path(cluster_cache).write_text(json.dumps(dump, indent=2))
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
    n_icon = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'      {n_icon} icon groups, {len(icon_groups)-n_icon} singletons')

    # Load icon plan (LLM decisions) if provided
    plan_by_bbox = {}  # keyed by approximate bbox tuple (rounded to int)
    if icon_plan and Path(icon_plan).exists():
        plan_data = json.loads(Path(icon_plan).read_text())
        for p in plan_data:
            key = (round(p['x']), round(p['y']), round(p['w']), round(p['h']))
            plan_by_bbox[key] = p
        print(f'      loaded plan for {len(plan_by_bbox)} icons')

    print('[5/8] emitting cells with quantization + plan...')
    path_by_idx = {p['idx']: p for p in remaining}
    cells = []
    cid = 100
    counts = Counter()

    for g in icon_groups:
        path_ids = sorted(g['path_ids'])
        if g.get('is_icon') and len(path_ids) >= 2:
            gp = [path_by_idx[pid] for pid in path_ids]
            bbox = g['bbox']
            x0, y0, x1, y1 = bbox
            w = x1-x0; h = y1-y0
            # Check plan
            key = (round(x0), round(y0), round(w), round(h))
            decision = plan_by_bbox.get(key, {}).get('recommend', 'simplify')

            if decision == 'drop':
                counts['dropped'] += 1
                continue
            if decision == 'emoji':
                emoji = plan_by_bbox[key].get('emoji', '')
                if emoji:
                    fs = max(16, min(int(min(w, h) * 0.85), 80))
                    val = emoji.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')
                    style = ('text;html=1;strokeColor=none;fillColor=none;align=center;'
                             f'verticalAlign=middle;whiteSpace=nowrap;rounded=0;'
                             f'fontFamily=Noto Color Emoji;fontSize={fs};fontColor=#000000;')
                    cells.append(
                        f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">'
                        f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                        f'</mxCell>'
                    )
                    counts['emoji'] += 1
                    cid += 1
                    continue
                # else fall through to simplify

            # Default: quantize and emit as SVG image cell
            svg, nc = quantized_icon_svg(gp, bbox, threshold=quantize_threshold)
            style = svg_to_image_cell_style(svg)
            cells.append(
                f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
                f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                f'</mxCell>'
            )
            counts[f'simplify_{nc}'] += 1
            counts['simplify_total'] += 1
            cid += 1
        else:
            # Singleton → stencil
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
                counts['singleton'] += 1
                cid += 1

    # Text cells
    all_paths_by_idx = {p['idx']: p for p in paths}
    for c in clusters:
        text = (c.get('text') or '').strip()
        if not text: continue
        x0,y0,x1,y1 = c['bbox']; w = x1-x0; h = y1-y0
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
                       f'fontFamily={font_family}', f'fontSize={fs}',
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
        'agent="svg_to_drawio_v9.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v9" name="traced">\n'
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
    print(f'[8/8] wrote {drawio_path}  ({len(cells)} cells, {os.path.getsize(drawio_path)//1024}KB)')
    print(f'      counts: {dict(counts)}')
    return dict(counts)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('png')
    ap.add_argument('-o','--output', default=None)
    ap.add_argument('--cluster-cache', default=None)
    ap.add_argument('--ocr-cache', default=None)
    ap.add_argument('--icon-plan', default=None)
    ap.add_argument('--font-family', default='DejaVu Sans')
    ap.add_argument('--max-icon-dim', type=float, default=120)
    ap.add_argument('--quantize', type=float, default=18.0,
                    help='color distance threshold (RGB ΔE76); higher=fewer colors, more simplify')
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert(args.svg, args.png, out,
            cluster_cache=args.cluster_cache,
            ocr_cache=args.ocr_cache,
            icon_plan=args.icon_plan,
            font_family=args.font_family,
            max_icon_dim=args.max_icon_dim,
            quantize_threshold=args.quantize)
