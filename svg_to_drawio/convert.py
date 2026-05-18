"""Main pipeline: PNG + SVG → drawio XML.

Stages (numbered the way they print so logs match):
  1. parse SVG paths
  2. drop full-canvas background
  3. detect panel backgrounds, drop their inner subpaths
  4. cluster glyphs, remove outliers
  5. OCR + multi-label split
  6. dedup text clusters, consume sub-glyph fragments
  7. detect line/arrowhead pairs → native edges
  8. cluster remaining paths into icons, drop text-overlapping icons
  9. emit cells in z-order
  10. write .drawio file
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from svg_to_drawio.parse import (
    parse_svg_paths, subpath_bbox, path_bbox,
)
from svg_to_drawio.panels import (
    is_full_canvas_background, is_panel_background, keep_outer_subpath_only,
)
from svg_to_drawio.glyphs import (
    identify_glyph_candidates, cluster_glyphs, remove_glyph_outliers,
    reabsorb_vertical_glyphs, evict_non_glyph_paths,
    split_spatially_joined_clusters,
)
from svg_to_drawio.ocr import (
    ocr_clusters, split_multilabel_clusters, clean_suspect_vertical_ocr,
    reject_hallucinated_text, recover_vertical_with_crop,
    recover_missing_chars, recover_inter_cluster_chars,
    merge_hyphen_continuation, cross_reference_typos,
    recover_split_pending, merge_stacked_text_lines,
)
from svg_to_drawio.dedup import dedupe_text_clusters, filter_overlapping_text_cells
from svg_to_drawio.edges import is_line_shaped, is_arrowhead_shaped, attach_arrowhead
from svg_to_drawio.icons import cluster_icons, absorb_singletons
from svg_to_drawio.emit import (
    emit_canvas, emit_panel_bg, emit_singleton_solid, emit_icon, emit_text,
    emit_edge_cell, wrap_drawio_xml, fill_color_for_cluster, emit_stencil,
    emit_container_rect, emit_simple_line, emit_native_shape,
)
from svg_to_drawio.semantic import build_semantic_shapes, recognize_vibrant_containers
from svg_to_drawio.border_detect import (
    detect_missing_rectangles, detect_adaptive_threshold_rectangles,
    detect_missing_lines,
)

__all__ = ['convert_pair']


def _load_clusters(cluster_cache: str | None, content_paths: list, W: int, H: int):
    if cluster_cache and Path(cluster_cache).exists():
        clusters = json.loads(Path(cluster_cache).read_text())
        for c in clusters:
            c['glyph_path_ids'] = set(c.get('glyph_path_ids', []))
        return clusters
    candidates = identify_glyph_candidates(content_paths)
    return cluster_glyphs(candidates, W, H)


def _bbox_overlaps_any(sbb, bbs, thresh: float = 0.5, pad: int = 4) -> bool:
    sx0, sy0, sx1, sy1 = sbb
    sa = max(1, (sx1 - sx0) * (sy1 - sy0))
    for bx0, by0, bx1, by1 in bbs:
        ix0 = max(sx0, bx0 - pad); iy0 = max(sy0, by0 - pad)
        ix1 = min(sx1, bx1 + pad); iy1 = min(sy1, by1 + pad)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if (ix1 - ix0) * (iy1 - iy0) / sa >= thresh:
            return True
    return False


def _consume_glyph_paths(clusters: list, content_paths: list, full_ocr: list | None) -> set:
    """Mark which content paths are claimed by text clusters (so they don't
    get double-rendered as singleton stencils).

    Exception: paths whose outer subpath nearly fills their bbox (chip
    bodies, header strips, icon blocks) are kept even if a cluster groups
    them — those are container/icon shapes, not letters. Without this, the
    RL-based Exploitation chip body, Hardware Agent header strip, etc.
    silently disappear.
    """
    from svg_to_drawio.glyphs import is_rectangle_silhouette

    path_by_idx = {p['idx']: p for p in content_paths}

    def _glyph_ids_minus_rects(c):
        ids = list(c.get('glyph_path_ids', []))
        return [pid for pid in ids
                if pid not in path_by_idx
                or not is_rectangle_silhouette(path_by_idx[pid])]

    consumed: set = set()
    for c in clusters:
        ng = len(c.get('glyph_path_ids', []))
        if c.get('text') or ng >= 4:
            consumed.update(_glyph_ids_minus_rects(c))

    text_bbs = [c['bbox'] for c in clusters
                if c.get('text') and len(c.get('glyph_path_ids', [])) >= 1]
    for c in clusters:
        if c.get('text') or len(c.get('glyph_path_ids', [])) >= 4:
            continue
        if _bbox_overlaps_any(c['bbox'], text_bbs, thresh=0.3, pad=8):
            consumed.update(_glyph_ids_minus_rects(c))

    if full_ocr:
        used_ocr_texts = {c.get('text', '') for c in clusters if c.get('text')}
        ocr_bbs = [(o['x1'], o['y1'], o['x2'], o['y2'])
                   for o in full_ocr if o.get('text', '') in used_ocr_texts]
        for c in clusters:
            if c.get('text') or len(c.get('glyph_path_ids', [])) >= 4:
                continue
            if _bbox_overlaps_any(c['bbox'], ocr_bbs, thresh=0.3, pad=4):
                consumed.update(_glyph_ids_minus_rects(c))

    # Also consume small dark paths fully inside any used-text OCR box
    # (sub-glyph fragments that escaped clustering).
    if full_ocr:
        used_ocr_texts = {c.get('text', '') for c in clusters if c.get('text')}
        ocr_bbs_pad = [(o['x1'] - 2, o['y1'] - 2, o['x2'] + 2, o['y2'] + 2)
                       for o in full_ocr if o.get('text', '') in used_ocr_texts]
        for p in content_paths:
            if p['idx'] in consumed:
                continue
            if not p.get('dark'):
                continue
            x0, y0, x1, y1 = p['bbox']
            if (x1 - x0) >= 60 or (y1 - y0) >= 40:
                continue
            for bx0, by0, bx1, by1 in ocr_bbs_pad:
                if x0 >= bx0 and y0 >= by0 and x1 <= bx1 and y1 <= by1:
                    consumed.add(p['idx'])
                    break
    return consumed


def _filter_subpaths_inside_text(content_paths: list, consumed: set,
                                 text_bboxes: list) -> list:
    """For non-consumed paths, drop subpaths that fall inside text cluster
    bboxes — keeps a path if at least one subpath survives.

    Exception: thin-outline paths (panel borders encoded as outer+holes) keep
    all their subpaths so the holes survive for stencil rendering. Without
    this, dropping the 'header inner' hole of a panel border converts the
    border into a near-solid fill that blankets the header.
    """
    from svg_to_drawio.emit import is_thin_outline

    def _in_text(sbb, thresh: float = 0.45) -> bool:
        sx0, sy0, sx1, sy1 = sbb
        sa = max(1, (sx1 - sx0) * (sy1 - sy0))
        for bx0, by0, bx1, by1 in text_bboxes:
            ix0 = max(sx0, bx0 - 3); iy0 = max(sy0, by0 - 2)
            ix1 = min(sx1, bx1 + 3); iy1 = min(sy1, by1 + 6)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if (ix1 - ix0) * (iy1 - iy0) / sa >= thresh:
                return True
        return False

    out = []
    for p in content_paths:
        if p['idx'] in consumed:
            continue
        # Probe shape on the original path before filtering subpaths
        is_outline = is_thin_outline({
            'bbox': (p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3]),
            'expanded': p['expanded'],
        })
        if is_outline:
            out.append({
                'idx': p['idx'], 'fill': p['fill'], 'fill_op': p['fill_op'],
                'expanded': p['expanded'], 'bbox': p['bbox'], 'dark': p['dark'],
                '_thin_outline': True,
            })
            continue
        kept = []
        # Only drop subpaths that are clearly LETTER-sized; larger subpaths
        # (e.g. agent header strips at 74×35 with light blue fill) are
        # containers we want to render — even if their bbox happens to
        # cover a text cluster. Same exemption for shape-silhouette paths
        # (chip bodies, ZX-graph colored circles): they're icon shapes,
        # not letter strokes.
        from svg_to_drawio.glyphs import is_rectangle_silhouette
        fill_hex = p.get('fill', '').lower().lstrip('#')
        try:
            r_ = int(fill_hex[0:2], 16); g_ = int(fill_hex[2:4], 16); b_ = int(fill_hex[4:6], 16)
            chroma = max(r_, g_, b_) - min(r_, g_, b_)
        except Exception:
            chroma = 0
        is_pale_container = '#' + fill_hex in (
            '#d9e5f2', '#ecf2f8', '#dbe1e7', '#dce2e8', '#e9eef5',
            '#e9f0f8', '#eaeff8', '#ebf0f8', '#e5eaf1',
        )
        is_silhouette = is_rectangle_silhouette({
            'bbox': p['bbox'], 'expanded': p['expanded'],
        }, threshold=0.7)
        # Colorful fill (chroma ≥ 30) is never a letter — letters are
        # rendered in dark gray/black; colored fills indicate icon parts
        # (red/green ZX nodes, orange brick blocks, etc.).
        is_colorful = chroma >= 30
        exempt = is_pale_container or is_silhouette or is_colorful
        for sub in p['subpaths']:
            x0, y0, w, h = subpath_bbox(sub)
            if (not exempt and w < 30 and h < 24
                    and _in_text((x0, y0, x0 + w, y0 + h))):
                continue
            kept.append(sub)
        if not kept:
            continue
        flat = [s for sub in kept for s in sub]
        x0, y0, w, h = path_bbox(flat)
        if w < 0.1 or h < 0.1:
            continue
        out.append({
            'idx': p['idx'], 'fill': p['fill'], 'fill_op': p['fill_op'],
            'expanded': flat, 'bbox': (x0, y0, x0 + w, y0 + h), 'dark': p['dark'],
        })
    return out


def _split_lines_arrows_other(remaining: list):
    line_paths, arrowhead_paths, other = [], [], []
    for p in remaining:
        if is_line_shaped(p, max_thickness=4.0, min_aspect=6.0):
            line_paths.append(p)
        elif is_arrowhead_shaped(p, max_dim=12.0):
            arrowhead_paths.append(p)
        else:
            other.append(p)
    return line_paths, arrowhead_paths, other


def _pair_arrows_with_lines(line_paths, arrowhead_paths):
    """Returns ([(line, start_arrow, end_arrow), ...], [unused_arrows])."""
    used_arrows = set()
    line_with_arrows = []
    for lp in line_paths:
        free = [a for a in arrowhead_paths if id(a) not in used_arrows]
        sa, ea = attach_arrowhead(lp, free, max_dist=8.0)
        if sa is not None:
            used_arrows.add(id(sa))
        if ea is not None:
            used_arrows.add(id(ea))
        line_with_arrows.append((lp, sa, ea))
    unattached = [a for a in arrowhead_paths if id(a) not in used_arrows]
    return line_with_arrows, unattached


def _drop_text_overlapping_icons(icon_groups: list, clusters: list,
                                  path_by_idx_local: dict | None = None) -> list:
    text_bboxes_only = [c['bbox'] for c in clusters if c.get('text')]
    path_by_idx_local = path_by_idx_local or {}
    def _overlap(g):
        if not g.get('is_icon'):
            return False
        gx0, gy0, gx1, gy1 = g['bbox']
        ga = max(1, (gx1 - gx0) * (gy1 - gy0))
        for tx0, ty0, tx1, ty1 in text_bboxes_only:
            ix0 = max(gx0, tx0 - 3); iy0 = max(gy0, ty0 - 3)
            ix1 = min(gx1, tx1 + 3); iy1 = min(gy1, ty1 + 3)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            if (ix1 - ix0) * (iy1 - iy0) / ga >= 0.5:
                return True
        return False

    def _is_letter_only_icon(g, path_by_idx_local):
        """Detect icons that wrap a single letter glyph plus tiny color-
        quantization fragments. Big-bold titles like 'CodeV-R1 Training
        Pipeline' produce these — half the letters cluster, the rest end
        up as icon stragglers and render as embedded SVG image-cells of
        individual letters next to the text. The icon's largest dark path
        is letter-shaped; smaller paths around it are vtracer interior
        cuts that don't change the verdict.
        """
        if not g.get('is_icon'):
            return False
        gx0, gy0, gx1, gy1 = g['bbox']
        gw = gx1 - gx0; gh = gy1 - gy0
        if gw > 32 or gh > 32:
            return False
        # Find the largest dark path in the group; if it's letter-shaped,
        # the whole icon is just rendering that letter.
        best = None; best_area = 0
        for pid in g['path_ids']:
            p = path_by_idx_local.get(pid)
            if p is None:
                continue
            if not p.get('dark'):
                continue
            bb = p['bbox']
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if area > best_area:
                best_area = area; best = p
        if best is None:
            return False
        bb = best['bbox']
        w = bb[2] - bb[0]; h = bb[3] - bb[1]
        if h > 32 or w > 30 or h < 8 or w < 4:
            return False
        return True

    return [g for g in icon_groups
            if not _overlap(g)
            and not _is_letter_only_icon(g, path_by_idx_local)]


def _load_full_ocr(svg_path: str, cluster_cache: str | None):
    candidates = []
    if cluster_cache:
        candidates.append(Path(cluster_cache).parent / 'ocr.json')
    candidates.append(Path(svg_path).parent / 'ocr.json')
    for cand in candidates:
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def convert_pair(svg_path: str, png_path: str, drawio_path: str,
                 cluster_cache: str | None = None,
                 ocr_cache: str | None = None,
                 stencil_size: int = 1000,
                 font_family: str = 'DejaVu Sans',
                 max_icon_dim: float = 120,
                 quantize_threshold: float = 30.0,
                 transparent_bg: bool = True) -> dict:
    print('[1/11] parsing SVG...')
    W, H, paths = parse_svg_paths(svg_path)
    print(f'        {W}x{H}, {len(paths)} paths')

    if transparent_bg:
        before = len(paths)
        paths = [p for p in paths if not is_full_canvas_background(p, W, H)]
        print(f'[2/11] dropped {before - len(paths)} canvas bg paths')

    panel_bgs = [p for p in paths if is_panel_background(p, W, H)]
    panel_bg_idxs = set(p['idx'] for p in panel_bgs)
    panel_bgs_solid = [keep_outer_subpath_only(p) for p in panel_bgs]
    print(f'[3/11] found {len(panel_bgs)} panel bg paths')

    content_paths = [p for p in paths if p['idx'] not in panel_bg_idxs]
    path_by_idx_all = {p['idx']: p for p in content_paths}

    print('[4/11] text clustering...')
    clusters = _load_clusters(cluster_cache, content_paths, W, H)
    clusters, n_outliers = remove_glyph_outliers(clusters, path_by_idx_all)
    clusters, n_evicted = evict_non_glyph_paths(clusters, path_by_idx_all)
    nv = sum(1 for c in clusters if c.get('vertical'))
    msg = f'        {len(clusters)} clusters ({nv} vertical)'
    if n_outliers:
        msg += f', -{n_outliers} outliers'
    if n_evicted:
        msg += f', -{n_evicted} non-glyph (arrow shafts etc.)'
    print(msg)

    print('[5/11] OCR...')
    full_ocr = _load_full_ocr(svg_path, cluster_cache)
    ocr_clusters(clusters, png_path, cache_path=ocr_cache, full_image_ocr=full_ocr)
    clusters, n_split = split_multilabel_clusters(clusters, full_ocr, path_by_idx_all)
    if n_split:
        print(f'        +split {n_split} multi-label clusters')
    # After OCR we can use cluster.text to identify noise-like horizontal
    # 'clusters' that actually hold glyphs from a nearby vertical label, and
    # transfer those glyphs into the vertical cluster so its bbox grows enough
    # for the rotated text to render at full length.
    clusters, n_reabsorbed = reabsorb_vertical_glyphs(clusters, path_by_idx_all)
    if n_reabsorbed:
        print(f'        reabsorbed {n_reabsorbed} glyphs into vertical clusters')
    n_cleared = clean_suspect_vertical_ocr(clusters)
    if n_cleared:
        print(f'        cleared {n_cleared} suspect vertical OCR misreads')
        # Try to recover the cleared vertical labels with a rotated crop OCR
        n_recovered = recover_vertical_with_crop(clusters, png_path)
        if n_recovered:
            print(f'        recovered {n_recovered} vertical labels via rotated-crop OCR')
    n_halluc = reject_hallucinated_text(clusters)
    if n_halluc:
        print(f'        cleared {n_halluc} hallucinated OCR (glyph-text length mismatch)')
    # If a cluster has more glyphs than its OCR text plausibly contains, the
    # OCR likely missed connector characters (e.g. '&'). Send the cluster's
    # crop to Claude for a corrected reading.
    n_missing = recover_missing_chars(clusters, png_path)
    if n_missing:
        print(f'        recovered {n_missing} clusters with missing chars (e.g. "&")')
    n_pairs = recover_inter_cluster_chars(clusters, png_path)
    if n_pairs:
        print(f'        merged {n_pairs} sibling cluster pairs (filled in connector chars)')
    # Split horizontal clusters whose glyphs span multiple labels separated by
    # a horizontal gap (e.g. P3's 'Bloch-' and 'Interfero-' top-of-label rows).
    clusters, n_spatial = split_spatially_joined_clusters(clusters, path_by_idx_all)
    if n_spatial:
        print(f'        split {n_spatial} spatially joined clusters')
        n_recovered = recover_split_pending(clusters, png_path)
        if n_recovered:
            print(f'        recovered {n_recovered} sub-cluster texts via Claude OCR')
    n_hyphen = merge_hyphen_continuation(clusters, png_path)
    if n_hyphen:
        print(f'        merged {n_hyphen} hyphen-continued multi-line labels')
    n_stacked = merge_stacked_text_lines(clusters)
    if n_stacked:
        print(f'        merged {n_stacked} stacked multi-line labels')

    consumed = _consume_glyph_paths(clusters, content_paths, full_ocr)

    before = len(clusters)
    clusters = dedupe_text_clusters(clusters)
    print(f'[6/11] deduped: {before} → {len(clusters)} clusters')
    clusters = filter_overlapping_text_cells(clusters)

    # Semantic recognition: for each text cluster, try to find a wrapping
    # container path and emit them together as a SINGLE native drawio
    # shape (rounded_rect / ellipse / cylinder / etc.) with editable text.
    # This replaces "stencil + floating text label" with a real shape — the
    # way a human builds the diagram in drawio.
    # Search BOTH content paths and panel backgrounds: many small "label
    # boxes" get classified as panels by is_panel_background but are
    # actually editable containers from the user's perspective.
    semantic_pool = content_paths + panel_bgs
    semantic_cells, semantic_consumed, absorbed_cluster_ids = build_semantic_shapes(
        clusters, semantic_pool, consumed, font_family=font_family)
    if semantic_cells:
        print(f'        recognized {len(semantic_cells)} native shapes '
              f'(text+container fused, {len(absorbed_cluster_ids)} sub-labels absorbed)')
    consumed = consumed | semantic_consumed
    semantic_cluster_ids = {id(s['cluster']) for s in semantic_cells}
    semantic_cluster_ids |= absorbed_cluster_ids
    # Vibrant-container recognition: catches the boxes with white text on
    # colored fill (Stage 0/1/2/3 pills, AT-GRPO orange box, Final 9B-V)
    # that never produced a dark-glyph cluster.
    vib_cells, vib_consumed = recognize_vibrant_containers(
        content_paths, panel_bgs, consumed, png_path,
        font_family=font_family,
        existing_shape_bboxes=[sc['bbox'] for sc in semantic_cells],
        full_image_ocr=full_ocr,
        existing_clusters=clusters,
    )
    if vib_cells:
        print(f'        recognized {len(vib_cells)} vibrant containers '
              f'(white-on-color, OCRd from PNG)')
    semantic_cells.extend(vib_cells)
    consumed = consumed | vib_consumed
    # Pre-allocate stable string IDs for native shapes so edges emitted
    # earlier in the file can reference them via source/target.
    shape_index = []
    for i, sc in enumerate(semantic_cells):
        sc['_id'] = f'shape_{i}'
        shape_index.append({'id': sc['_id'], 'bbox': sc['bbox']})

    # Consume paths whose bbox is FULLY INSIDE any native shape's bbox.
    # These are the container's inner fill / decoration paths — the native
    # shape now redraws this region with editable fill+text, so the inner
    # stencils would otherwise overdraw and leave stale outlines when the
    # user drags the shape. Skip line-shaped paths so that connectors
    # crossing into the shape (arrowheads near the boundary etc.) survive.
    inner_consumed = 0
    for p in content_paths:
        if p['idx'] in consumed:
            continue
        pb = p['bbox']
        pw = pb[2] - pb[0]; ph = pb[3] - pb[1]
        # Treat connectors / lines / large icons as boundary-spanning
        # (don't consume them just because they happen to fit in a shape).
        if pw > 200 or ph > 200:
            continue
        for sc in semantic_cells:
            sb = sc['bbox']
            pad = 2.0
            if (sb[0] - pad <= pb[0] and sb[1] - pad <= pb[1]
                    and sb[2] + pad >= pb[2] and sb[3] + pad >= pb[3]):
                consumed.add(p['idx'])
                inner_consumed += 1
                break
    if inner_consumed:
        print(f'        consumed {inner_consumed} inner stencils replaced by native shapes')
    # Drop any panel_bg path that became a native shape — otherwise it
    # would also render as a stencil at the bottom of z-order, double-
    # drawing the same rectangle.
    panel_bgs_solid = [p for p in panel_bgs_solid if p['idx'] not in semantic_consumed]

    text_bboxes = [c['bbox'] for c in clusters if c.get('text')]
    remaining = _filter_subpaths_inside_text(content_paths, consumed, text_bboxes)

    print('[7/11] detecting edges (lines + arrowheads)...')
    line_paths, arrowhead_paths, other_remaining = _split_lines_arrows_other(remaining)
    print(f'        {len(line_paths)} lines, {len(arrowhead_paths)} arrowhead candidates')
    line_with_arrows, unattached_arrows = _pair_arrows_with_lines(line_paths, arrowhead_paths)

    print(f'[8/11] clustering {len(other_remaining)} into icons...')
    icon_groups = cluster_icons(other_remaining, W, H,
                                max_icon_dim=max_icon_dim,
                                min_paths_per_icon=2, eps=2.0)
    path_by_idx = {p['idx']: p for p in other_remaining}
    icon_groups, _ = absorb_singletons(icon_groups, path_by_idx, max_absorb_dim=40)
    before_drop = sum(1 for g in icon_groups if g.get('is_icon'))
    icon_groups = _drop_text_overlapping_icons(icon_groups, clusters,
                                                 path_by_idx_local=path_by_idx)
    n_icons = sum(1 for g in icon_groups if g.get('is_icon'))
    print(f'        {n_icons} icons (dropped {before_drop - n_icons} text-overlapping icons)')

    print('[9/11] emitting cells in z-order...')
    cells = []
    cid = 100

    if transparent_bg:
        cells.append(emit_canvas(cid, W, H))
        cid += 1

    panel_count = 0
    for p in panel_bgs_solid:
        xml = emit_panel_bg(cid, p, stencil_size=stencil_size)
        if xml:
            cells.append(xml); cid += 1; panel_count += 1

    # Re-detect blue-tinted container rectangles in the source PNG that the
    # vtracer SVG didn't capture (P2 panel outline, P1 'Filter Funnel' inner
    # rect, P3 'Decomposition Agent' inner rect, etc.). Emit them just above
    # the panel background so they sit under singletons / icons / text.
    existing_bboxes = [tuple(p['bbox']) for p in paths]
    panel_bboxes_for_detect = [tuple(p['bbox']) for p in panel_bgs]
    missing_rects = detect_missing_rectangles(
        png_path, existing_bboxes, panel_bboxes_for_detect, W, H)
    # Second pass: catch faint-border rects (white interior, gray stroke) that
    # the chroma-only first pass misses (e.g. P3 Hardware Agent box).
    extra_rects = detect_adaptive_threshold_rectangles(
        png_path, existing_bboxes + [tuple(m['bbox']) for m in missing_rects],
        panel_bboxes_for_detect)
    missing_rects.extend(extra_rects)
    container_count = 0
    for m in missing_rects:
        # Use the sampled fill if it's clearly a pale background (sum of RGB
        # ≥ ~660), otherwise emit stroke-only so we don't blanket existing
        # icons/text with a misread dark color.
        fill_hex = m.get('fill', '#eaf2f7')
        try:
            r_ = int(fill_hex[1:3], 16); g_ = int(fill_hex[3:5], 16); b_ = int(fill_hex[5:7], 16)
            is_pale = (r_ + g_ + b_) >= 660
        except Exception:
            is_pale = False
        fill = fill_hex if is_pale else 'none'
        xml = emit_container_rect(cid, m['bbox'],
                                  fill=fill,
                                  stroke=m.get('stroke', '#a8b6c8'))
        if xml:
            cells.append(xml); cid += 1; container_count += 1
    if container_count:
        print(f'        added {container_count} missing container rects from PNG')

    edge_count = 0
    bound_count = 0
    for lp, sa, ea in line_with_arrows:
        xml = emit_edge_cell(cid, lp, sa, ea, shape_index=shape_index)
        if xml:
            if 'source="' in xml or 'target="' in xml:
                bound_count += 1
            cells.append(xml); cid += 1; edge_count += 1
    if bound_count:
        print(f'        bound {bound_count}/{edge_count} edges to native shapes (drag-follow)')

    sing_count = 0
    for g in icon_groups:
        if g.get('is_icon') and len(g['path_ids']) >= 2:
            continue
        for pid in sorted(g['path_ids']):
            p = path_by_idx[pid]
            xml = emit_singleton_solid(cid, p, stencil_size=stencil_size)
            if xml:
                cells.append(xml); cid += 1; sing_count += 1

    for ah in unattached_arrows:
        xml = emit_stencil(cid, ah, stencil_size=stencil_size)
        if xml:
            cells.append(xml); cid += 1; sing_count += 1

    icon_count = 0
    for g in icon_groups:
        if g.get('is_icon') and len(g['path_ids']) >= 2:
            gp = [path_by_idx[pid] for pid in sorted(g['path_ids'])]
            xml = emit_icon(cid, gp, g['bbox'], quantize_threshold=quantize_threshold)
            cells.append(xml); cid += 1; icon_count += 1

    # Native shape cells: emit AFTER stencils + icons so the recognized
    # box (with its editable text inside) renders ON TOP of any leftover
    # path fragments that weren't consumed. Without this, vtracer's
    # interior-fill subpaths render over the text.
    all_paths_by_idx = {p['idx']: p for p in paths}
    semantic_count = 0
    for sc in semantic_cells:
        cluster = sc['cluster']
        fill_hex = sc['fill']
        try:
            r_ = int(fill_hex.lstrip('#')[0:2], 16)
            g_ = int(fill_hex.lstrip('#')[2:4], 16)
            b_ = int(fill_hex.lstrip('#')[4:6], 16)
            stroke = f'#{max(0, r_ - 60):02x}{max(0, g_ - 60):02x}{max(0, b_ - 60):02x}'
            # Auto-contrast font color: dark fills → white text, light fills
            # → black text. Threshold ~ perceived luminance 128.
            luminance = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            if luminance < 128:
                font_color = '#ffffff'
            else:
                font_color = '#070707'
        except Exception:
            stroke = '#666666'
            font_color = '#000000'
        # Override with explicit cluster-derived color if available
        try:
            inferred = fill_color_for_cluster(cluster, all_paths_by_idx)
            if inferred and inferred != '#000000':
                # Only use cluster color if it has reasonable contrast against fill.
                ir = int(inferred.lstrip('#')[0:2], 16)
                ig = int(inferred.lstrip('#')[2:4], 16)
                ib = int(inferred.lstrip('#')[4:6], 16)
                ilum = 0.299 * ir + 0.587 * ig + 0.114 * ib
                if abs(ilum - luminance) > 50:
                    font_color = inferred
        except Exception:
            pass
        # Use the pre-allocated string ID so edges emitted earlier can
        # already reference this shape via source/target.
        xml = emit_native_shape(
            sc['_id'], sc['bbox'], sc['text'], sc['shape'],
            fill=sc['fill'], stroke=stroke,
            font_size=int(sc['font_size']),
            bold=bool(sc['bold']),
            font_family=sc['font_family'],
            font_color=font_color,
        )
        if xml:
            cells.append(xml); semantic_count += 1
    if semantic_count:
        print(f'        emitted {semantic_count} native drawio shapes (editable text inside)')
    text_count = 0
    for c in clusters:
        # Skip clusters already represented as native shapes — their text
        # is now embedded in the shape's value attribute.
        if id(c) in semantic_cluster_ids:
            continue
        color = fill_color_for_cluster(c, all_paths_by_idx)
        xml = emit_text(cid, c, color, font_family=font_family)
        if xml:
            cells.append(xml); cid += 1; text_count += 1

    # Render-driven line repair: write the drawio, render to PNG via Docker,
    # diff against the source, find any LINE segments still missing, emit
    # them as plain edges, repeat until convergence. Catches panel-divider
    # lines, chart axes, etc. that vtracer produced but the pipeline
    # filtered.
    drawio = wrap_drawio_xml(cells, W, H, agent='svg_to_drawio')
    Path(drawio_path).write_text(drawio)
    total_lines = 0
    added_lines = set()  # dedupe by quantized coords
    try:
        import subprocess as _sp
        for iteration in range(3):
            _sp.run([
                'docker', 'run', '--rm',
                '-v', f'{Path(drawio_path).parent.absolute()}:/data',
                'drawio-export-cmu', '--format', 'png', '--scale', '2',
                '--transparent', Path(drawio_path).name,
            ], check=True, capture_output=True, timeout=120)
            rendered_png = (Path(drawio_path).parent / 'export'
                            / f'{Path(drawio_path).stem}-traced.png')
            if not rendered_png.exists():
                break
            text_cluster_bboxes = [c['bbox'] for c in clusters if c.get('text')]
            missing_lines = detect_missing_lines(png_path, str(rendered_png),
                                                  min_length=30, max_thickness=4,
                                                  text_bboxes=text_cluster_bboxes)
            iter_count = 0
            for ml in missing_lines:
                # Dedupe: round coords to nearest 4 px and skip if already added
                key = (ml['orientation'],
                       int(ml['x0'] // 4), int(ml['y0'] // 4),
                       int(ml['x1'] // 4), int(ml['y1'] // 4))
                if key in added_lines:
                    continue
                added_lines.add(key)
                xml = emit_simple_line(cid, ml['x0'], ml['y0'], ml['x1'], ml['y1'],
                                       color=ml.get('color', '#000000'),
                                       width=1.0,
                                       shape_index=shape_index)
                if xml:
                    cells.append(xml); cid += 1; iter_count += 1
            total_lines += iter_count
            print(f'        line-repair iter {iteration+1}: added {iter_count} new')
            if iter_count == 0:
                break  # converged
            drawio = wrap_drawio_xml(cells, W, H, agent='svg_to_drawio')
            Path(drawio_path).write_text(drawio)
        if total_lines:
            print(f'        total lines added: {total_lines}')
    except Exception as e:
        print(f'        line-repair pass skipped: {e}')

    print(f'[10/11] wrote {drawio_path}  ({len(cells)} cells, '
          f'{os.path.getsize(drawio_path) // 1024}KB)')
    print(f'[11/11] z: {panel_count} panel-bgs → {edge_count} edges + '
          f'{sing_count} singletons → {icon_count} icons → {text_count} text')
    return {
        'panels': panel_count, 'edges': edge_count, 'singletons': sing_count,
        'icons': icon_count, 'text': text_count, 'total': len(cells),
    }
