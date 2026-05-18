"""svg_to_drawio_v6.py — v5 + icon grouping.

Non-text paths that visually form a single icon (robot, chip, funnel, etc.)
are grouped into a drawio group cell so they move together.

Icon detection via spatial clustering with DBSCAN-style nearest-neighbor
merging:
 - Start with all non-text, non-background paths
 - Greedy merge: if two paths' bboxes overlap or touch (within ε px), merge
 - Filter: only cluster if it has >= 2 colored paths AND total bbox <= 200×200
 - Large background rects (≥ 200px either dim OR filling most of a panel)
   are kept as standalone cells.

Usage:
    python svg_to_drawio_v6.py input.svg input.jpg [-o output.drawio]
      [--cluster-cache cluster.json --ocr-cache ocr.json]
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
from svg_to_drawio_v4 import split_subpaths, subpath_bbox
from svg_to_drawio_v5 import (
    _parse_hex, _is_dark,
    parse_svg_paths, identify_glyph_candidates,
    cluster_glyphs, ocr_clusters,
)


# ----------------------------------------------------------------------------
# Icon detection — spatial clustering of non-text paths
# ----------------------------------------------------------------------------

def bbox_overlap_or_near(a, b, eps: float = 4.0) -> bool:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    return (ax0 <= bx1 + eps and bx0 <= ax1 + eps and
            ay0 <= by1 + eps and by0 <= ay1 + eps)


def bbox_union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def cluster_icons(paths, W, H,
                  max_icon_dim: float = 120,
                  min_paths_per_icon: int = 2,
                  eps: float = 2.0) -> list:
    """Group non-background paths into icon clusters via bbox adjacency.

    Uses strict per-merge validation: each proposed union must yield a final
    cluster bbox within max_icon_dim×max_icon_dim. Repeatedly iterates until
    stable to avoid transitive over-merging.
    """
    # Build list of candidate paths (exclude very large backgrounds)
    cands = []
    for p in paths:
        x0, y0, x1, y1 = p['bbox']
        w = x1 - x0; h = y1 - y0
        if w >= max_icon_dim or h >= max_icon_dim:
            continue
        if w < 1 or h < 1:
            continue
        cands.append(p)

    # Per-cluster bbox tracking (not just per-path)
    cluster_bbox = {p['idx']: p['bbox'] for p in cands}
    parent = {p['idx']: p['idx'] for p in cands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def try_union(a, b) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb: return False
        ba = cluster_bbox[ra]; bb = cluster_bbox[rb]
        ub = bbox_union(ba, bb)
        if ub[2] - ub[0] > max_icon_dim or ub[3] - ub[1] > max_icon_dim:
            return False
        parent[ra] = rb
        cluster_bbox[rb] = ub
        cluster_bbox[ra] = ub
        return True

    # Iterate to fixpoint: check adjacency between cluster bboxes
    from collections import defaultdict
    changed = True
    iteration = 0
    while changed and iteration < 8:
        changed = False
        iteration += 1
        # Group by root → check neighbor roots
        GRID = 32
        grid = defaultdict(set)
        for pid in list(cluster_bbox.keys()):
            root = find(pid)
            bb = cluster_bbox[root]
            gx0 = int(bb[0]) // GRID; gx1 = int(bb[2]) // GRID
            gy0 = int(bb[1]) // GRID; gy1 = int(bb[3]) // GRID
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    grid[(gx, gy)].add(root)
        for p in cands:
            root = find(p['idx'])
            bb = cluster_bbox[root]
            gx0 = int(bb[0]) // GRID - 1; gx1 = int(bb[2]) // GRID + 1
            gy0 = int(bb[1]) // GRID - 1; gy1 = int(bb[3]) // GRID + 1
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    for other_root in list(grid.get((gx, gy), [])):
                        if other_root == root: continue
                        ob = cluster_bbox[other_root]
                        if bbox_overlap_or_near(bb, ob, eps=eps):
                            if try_union(root, other_root):
                                changed = True
                                root = find(p['idx'])
                                bb = cluster_bbox[root]

    # Collect groups
    groups = {}
    for p in cands:
        root = find(p['idx'])
        groups.setdefault(root, []).append(p)

    # Also: paths that were EXCLUDED from cands (bg rects) as singleton groups
    excluded = [p for p in paths if all(p['idx'] != c['idx'] for c in cands)]

    # Build output
    icons = []
    for group_paths in groups.values():
        if len(group_paths) < min_paths_per_icon:
            # Emit as singleton (keep layout)
            for p in group_paths:
                icons.append({
                    'path_ids': {p['idx']},
                    'bbox': p['bbox'],
                    'is_icon': False,
                })
            continue
        xs0 = min(p['bbox'][0] for p in group_paths)
        ys0 = min(p['bbox'][1] for p in group_paths)
        xs1 = max(p['bbox'][2] for p in group_paths)
        ys1 = max(p['bbox'][3] for p in group_paths)
        icons.append({
            'path_ids': {p['idx'] for p in group_paths},
            'bbox': (xs0, ys0, xs1, ys1),
            'is_icon': True,
            'num_paths': len(group_paths),
        })
    # Background rects as singletons
    for p in excluded:
        icons.append({'path_ids': {p['idx']}, 'bbox': p['bbox'], 'is_icon': False})
    return icons


# ----------------------------------------------------------------------------
# Text color / glyph utilities (same as v5)
# ----------------------------------------------------------------------------

def convert(svg_path: str, png_path: str, drawio_path: str,
            cluster_cache: str | None = None,
            ocr_cache: str | None = None,
            stencil_size: int = 1000,
            font_family: str = 'DejaVu Sans') -> dict:
    import numpy as np
    from PIL import Image

    print('[1/7] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'      {W}x{H}, {len(paths)} paths')

    print('[2/7] identifying glyph candidates + clustering...')
    candidates = identify_glyph_candidates(paths)
    if cluster_cache and Path(cluster_cache).exists():
        saved = json.loads(Path(cluster_cache).read_text())
        clusters = saved
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
    else:
        clusters = cluster_glyphs(candidates, W, H)
        if cluster_cache:
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

    # Consumed path ids (glyph paths we replace with text cells)
    consumed = set()
    for c in clusters:
        ng = len(c.get('glyph_path_ids', []))
        if c.get('text') or ng >= 4:
            consumed.update(c['glyph_path_ids'])
    print(f'[4/7] consumed {len(consumed)} glyph paths')

    # Non-text paths remaining → icon clustering
    print('[5/7] clustering icons...')
    remaining = [p for p in paths if p['idx'] not in consumed]
    # Compute text bboxes for canvas-hole dropping
    text_bboxes = [c['bbox'] for c in clusters if c.get('text')]

    def _in_text(sbb, thresh=0.45):
        sx0,sy0,sx1,sy1 = sbb
        sa = max(1,(sx1-sx0)*(sy1-sy0))
        for bx0,by0,bx1,by1 in text_bboxes:
            ix0 = max(sx0, bx0 - 3); iy0 = max(sy0, by0 - 2)
            ix1 = min(sx1, bx1 + 3); iy1 = min(sy1, by1 + 6)
            if ix1<=ix0 or iy1<=iy0: continue
            if (ix1-ix0)*(iy1-iy0) / sa >= thresh:
                return True
        return False

    # Drop canvas holes at text positions from each remaining path's subpaths
    remaining_processed = []
    for p in remaining:
        kept = []
        for sub in p['subpaths']:
            x0, y0, w, h = subpath_bbox(sub)
            if w < 100 and h < 36 and _in_text((x0,y0,x0+w,y0+h)):
                continue
            kept.append(sub)
        if not kept:
            continue
        flat = [seg for sub in kept for seg in sub]
        x0, y0, w, h = path_bbox(flat)
        if w < 0.1 or h < 0.1:
            continue
        remaining_processed.append({
            'idx': p['idx'], 'fill': p['fill'], 'fill_op': p['fill_op'],
            'expanded': flat, 'bbox': (x0, y0, x0 + w, y0 + h),
            'dark': p['dark'],
        })

    # Spatial icon clustering on processed paths
    icon_groups = cluster_icons(remaining_processed, W, H,
                                max_icon_dim=200, min_paths_per_icon=2)
    n_icon_groups = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'      {n_icon_groups} icon groups, {len(icon_groups)-n_icon_groups} singletons')

    print('[6/7] emitting cells...')
    orig_img = np.asarray(Image.open(png_path).convert('RGB'))
    cells = []
    cid = 100
    path_by_idx = {p['idx']: p for p in remaining_processed}

    # Emit each icon group as a drawio group cell containing child path cells
    next_group_id = 1000
    for group in icon_groups:
        path_ids = group['path_ids']
        if group.get('is_icon') and len(path_ids) >= 2:
            gx0, gy0, gx1, gy1 = group['bbox']
            gw = gx1 - gx0; gh = gy1 - gy0
            # Drawio group cell: style="group;html=1;..."
            group_id = f'g{next_group_id}'; next_group_id += 1
            cells.append(
                f'<mxCell id="{group_id}" value="" '
                f'style="group;html=1;" vertex="1" connectable="0" parent="1">'
                f'<mxGeometry x="{gx0:.2f}" y="{gy0:.2f}" width="{gw:.2f}" height="{gh:.2f}" as="geometry"/>'
                f'</mxCell>'
            )
            # Child cells positioned RELATIVE to the group (x-gx0, y-gy0)
            for pid in sorted(path_ids):
                p = path_by_idx[pid]
                x0, y0, x1, y1 = p['bbox']
                w = x1 - x0; h = y1 - y0
                stencil_xml = segments_to_stencil(p['expanded'], (x0, y0, w, h), stencil_size=stencil_size)
                stencil_b64 = encode_stencil(stencil_xml)
                style_parts = [f'shape=stencil({stencil_b64})',
                               f'fillColor={p["fill"]}',
                               'strokeColor=none', 'html=1']
                if p['fill_op'] < 1.0:
                    style_parts.append(f'opacity={int(p["fill_op"]*100)}')
                style = ';'.join(style_parts) + ';'
                # Relative coords inside group
                rx = x0 - gx0; ry = y0 - gy0
                cells.append(
                    f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="{group_id}">'
                    f'<mxGeometry x="{rx:.2f}" y="{ry:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
                    f'</mxCell>'
                )
                cid += 1
        else:
            # Singleton — emit directly with parent=1
            for pid in sorted(path_ids):
                p = path_by_idx[pid]
                x0, y0, x1, y1 = p['bbox']
                w = x1 - x0; h = y1 - y0
                stencil_xml = segments_to_stencil(p['expanded'], (x0, y0, w, h), stencil_size=stencil_size)
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

    # Emit text cells (top layer)
    from collections import Counter
    for c in clusters:
        text = (c.get('text') or '').strip()
        if not text: continue
        x0, y0, x1, y1 = c['bbox']
        w = x1 - x0; h = y1 - y0
        if w < 2 or h < 2: continue
        # Determine color from glyph paths (use original path's fill)
        if c.get('glyphs'):
            fills = [g['path']['fill'] for g in c['glyphs']]
        else:
            # cached cluster: resolve path_by_idx but include consumed glyph paths too
            all_paths_by_idx = {p['idx']: p for p in paths}
            fills = [all_paths_by_idx[pid]['fill']
                     for pid in c.get('glyph_path_ids', [])
                     if pid in all_paths_by_idx]
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
        'agent="svg_to_drawio_v6.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_v6" name="traced">\n'
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
    print(f'[7/7] wrote {drawio_path}  ({len(cells)} cells, {os.path.getsize(drawio_path)//1024}KB)')
    return {'paths': len(paths), 'clusters': len(clusters),
            'icon_groups': n_icon_groups, 'cells': len(cells)}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('svg')
    ap.add_argument('png')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--cluster-cache', default=None)
    ap.add_argument('--ocr-cache', default=None)
    ap.add_argument('--font-family', default='DejaVu Sans')
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert(args.svg, args.png, out,
            cluster_cache=args.cluster_cache,
            ocr_cache=args.ocr_cache,
            font_family=args.font_family)
