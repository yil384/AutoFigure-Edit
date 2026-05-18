"""drawio cell emitters — one function per z-order layer.

Z-order (bottom → top):
  canvas → panel-bgs → singletons + native edges → icons → text
"""
from collections import Counter

from svg_to_drawio.parse import (
    segments_to_stencil, encode_stencil, path_bbox, split_subpaths,
)
from svg_to_drawio.edges import emit_edge_cell
from svg_to_drawio.icons import quantized_icon_svg, svg_to_image_cell_style

__all__ = [
    'emit_canvas', 'emit_panel_bg', 'emit_stencil', 'emit_singleton_solid',
    'emit_thin_outline', 'emit_icon', 'emit_text', 'emit_edge_cell',
    'emit_container_rect', 'emit_simple_line', 'emit_native_shape',
    'wrap_drawio_xml', 'is_thin_outline',
]


def emit_native_shape(cid, bbox: tuple, text: str, shape: str,
                      fill: str = '#ffffff', stroke: str = '#666666',
                      font_size: int = 10, bold: bool = False,
                      font_family: str = 'Helvetica',
                      font_color: str = '#000000') -> str:
    """Emit ONE drawio native vertex cell with editable text inside. This
    replaces the (stencil container + separate text cell) pair with a
    single human-editable shape — the way someone would actually build
    the diagram by hand in drawio.
    """
    from svg_to_drawio.semantic import shape_style_for
    x0, y0, x1, y1 = bbox
    w = x1 - x0; h = y1 - y0
    if w < 1 or h < 1:
        return ''
    base_style = shape_style_for(shape) or shape_style_for('rect')
    style_parts = [
        base_style.rstrip(';'),
        f'fillColor={fill}',
        f'strokeColor={stroke}',
        f'fontFamily={font_family}',
        f'fontSize={font_size}',
        f'fontColor={font_color}',
        'verticalAlign=middle',
        'align=center',
    ]
    if bold:
        style_parts.append('fontStyle=1')
    style = ';'.join(style_parts) + ';'
    val = (text.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('"', '&quot;'))
    if '\n' in val:
        val = val.replace('\n', '&lt;br&gt;')
    return (
        f'<mxCell id="{cid}" value="{val}" style="{style}" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'as="geometry"/></mxCell>'
    )


def emit_simple_line(cid: int, x0: float, y0: float, x1: float, y1: float,
                     color: str = '#000000', width: float = 1.0,
                     shape_index=None) -> str:
    """Plain straight line/edge. When `shape_index` is provided, attempts
    to bind endpoints to native shapes (source/target + exit/entry ratios)
    so the line follows when shapes are dragged. Otherwise emits absolute
    mxPoint coords (legacy behavior)."""
    from svg_to_drawio.edges import snap_to_shape
    src_snap = tgt_snap = None
    if shape_index:
        src_snap = snap_to_shape((x0, y0), shape_index)
        tgt_snap = snap_to_shape((x1, y1), shape_index)
        # Drop self-loops — same shape on both ends is usually a perimeter
        # stroke fragment, not a real connector.
        if (src_snap is not None and tgt_snap is not None
                and src_snap[0] == tgt_snap[0]):
            src_snap = tgt_snap = None
    parts = ['endArrow=none', 'html=1', 'rounded=0',
             f'strokeColor={color}', f'strokeWidth={width}']
    if src_snap is not None:
        _, ex, ey = src_snap
        parts.append(f'exitX={ex:.3f};exitY={ey:.3f};exitDx=0;exitDy=0')
    if tgt_snap is not None:
        _, ex, ey = tgt_snap
        parts.append(f'entryX={ex:.3f};entryY={ey:.3f};entryDx=0;entryDy=0')
    edge_attrs = 'edge="1" parent="1"'
    if src_snap is not None:
        edge_attrs += f' source="{src_snap[0]}"'
    if tgt_snap is not None:
        edge_attrs += f' target="{tgt_snap[0]}"'
    geom = ''
    if src_snap is None:
        geom += f'<mxPoint x="{x0:.1f}" y="{y0:.1f}" as="sourcePoint"/>'
    if tgt_snap is None:
        geom += f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/>'
    return (
        f'<mxCell id="{cid}" value="" '
        f'style="{";".join(parts)};" {edge_attrs}>'
        f'<mxGeometry relative="1" as="geometry">'
        f'{geom}'
        f'</mxGeometry></mxCell>'
    )


def emit_container_rect(cid: int, bbox: tuple, fill: str = '#eaf2f7',
                        stroke: str = '#a8b6c8',
                        stroke_width: float = 1.0,
                        rounded: bool = True) -> str:
    """Plain rounded-rect cell for missing container borders that vtracer
    didn't trace. Sits in the z-order between panel backgrounds and
    singletons so it doesn't blanket icons or text."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w < 1 or h < 1:
        return ''
    rounded_flag = 1 if rounded else 0
    return (
        f'<mxCell id="{cid}" value="" '
        f'style="rounded={rounded_flag};whiteSpace=wrap;html=1;'
        f'fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};'
        f'arcSize=8;" vertex="1" parent="1">'
        f'<mxGeometry x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'as="geometry"/></mxCell>'
    )


def emit_canvas(cid: int, W: int, H: int) -> str:
    """Invisible locked rect spanning the canvas. Prevents drawio from showing
    a draggable bbox around the diagram and pins the (0,0)-(W,H) frame."""
    return (
        f'<mxCell id="{cid}" value="" '
        f'style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;'
        f'locked=1;movable=0;editable=0;deletable=0;resizable=0;rotatable=0;'
        f'connectable=0;selectable=0;" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="0" y="0" width="{W}" height="{H}" as="geometry"/>'
        f'</mxCell>'
    )


def _hex_rgb_sum(hex_color: str) -> int:
    try:
        s = hex_color.lstrip('#')
        if len(s) != 6:
            return 0
        return int(s[0:2], 16) + int(s[2:4], 16) + int(s[4:6], 16)
    except Exception:
        return 0


def _hex_rgb(hex_color: str) -> tuple[int, int, int]:
    try:
        s = hex_color.lstrip('#')
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except Exception:
        return 0, 0, 0


def _is_container_fill(hex_color: str) -> bool:
    """A 'container' fill is a pale-chroma color (not pure white/gray) used
    by figure builders for rounded-rect group backgrounds. Pure white fills
    like #ffffff are usually fillers INSIDE another shape (e.g. interior of
    a curving arrow or knockout of a black silhouette) — they shouldn't get
    a stroke around their bbox or it'll draw a phantom rectangle inside.
    """
    r, g, b = _hex_rgb(hex_color)
    if r + g + b < 720:
        return False
    # Require a faint blue/pink tint: at least one channel must differ from
    # the other two by 4+ units. Pure-white (255,255,255) and pure-gray
    # (240,240,240) are excluded.
    chroma = max(r, g, b) - min(r, g, b)
    return chroma >= 4


def emit_stencil(cid: int, p: dict, stencil_size: int = 1000) -> str:
    """Single mxCell carrying a path as a drawio shape=stencil(...).

    If the path's fill is near-white (RGB sum ≥ 720) and the bbox is big
    enough to be a container rect (≥80×60), add a thin gray stroke so the
    container is visible. Without this, vtracer-traced container rects
    (e.g. P3 'Hardware Agent' box, fill=#ecf2f8) blend into the canvas
    background and the user sees no border.
    """
    x0, y0, x1, y1 = p['bbox']
    w, h = x1 - x0, y1 - y0
    if w < 0.1 or h < 0.1:
        return ''
    stencil_xml = segments_to_stencil(p['expanded'], (x0, y0, w, h),
                                      stencil_size=stencil_size)
    fill_hex = p['fill']
    is_near_white_container = (_is_container_fill(fill_hex)
                               and w >= 60 and h >= 60)
    if is_near_white_container:
        # Stencil's <fill/> alone paints only the interior; we need <fillstroke/>
        # so drawio also paints the path outline using the cell's strokeColor.
        stencil_xml = stencil_xml.replace('<fill/>', '<fillstroke/>', 1)
    style_parts = [f'shape=stencil({encode_stencil(stencil_xml)})',
                   f'fillColor={fill_hex}',
                   'html=1']
    if is_near_white_container:
        # Match the figure's existing panel-border color exactly so the new
        # outline blends with the other rounded rects vtracer already encoded
        # as thin-outline paths (#8f9cac shows up in 18+ panel paths).
        # 0.5 px is the lightest visible width — keeps borders perceptible
        # without over-darkening the pixels around them.
        style_parts.append('strokeColor=#8f9cac')
        style_parts.append('strokeWidth=0.5')
    else:
        style_parts.append('strokeColor=none')
    if p.get('fill_op', 1.0) < 1.0:
        style_parts.append(f'opacity={int(p["fill_op"]*100)}')
    return (
        f'<mxCell id="{cid}" value="" style="{";".join(style_parts)};" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'as="geometry"/></mxCell>'
    )


def emit_panel_bg(cid: int, p: dict, stencil_size: int = 1000) -> str:
    """Panel background — same as emit_stencil but explicit name keeps the
    layer separation readable in convert.py."""
    return emit_stencil(cid, p, stencil_size=stencil_size)


def is_thin_outline(p: dict, area_ratio: float = 0.1) -> bool:
    """A path is a 'thin outline' (stroke disguised as fill) when its actual
    painted area (outer minus inner holes) is far smaller than its bbox area.
    vtracer encodes panel borders as outer rectangle + inner rectangle holes
    (evenodd) — when we render that as a filled stencil with default nonzero
    fill, drawio paints the whole bbox, blanketing whatever's behind. Such
    paths must render as stroked outlines instead.

    Painted-area computation: largest subpath = outer; all others = holes.
    actual = max(areas) - sum(other areas).
    """
    from svg_to_drawio.parse import split_subpaths
    bb = p['bbox']
    bw = bb[2] - bb[0]; bh = bb[3] - bb[1]
    if bw <= 0 or bh <= 0:
        return False
    bbox_area = bw * bh
    subs = split_subpaths(p['expanded'])
    if not subs:
        return False
    areas = [_polygon_shoelace_area(s) for s in subs]
    if len(areas) == 1:
        actual = areas[0]
    else:
        outer = max(areas)
        actual = outer - (sum(areas) - outer)
        actual = max(0.0, actual)
    return actual / bbox_area < area_ratio


def emit_thin_outline(cid: int, p: dict, stencil_size: int = 1000) -> str:
    """Render a thin-outline path as a stencil that STROKES the path instead
    of filling it. Uses the original fillColor as strokeColor and a 1px width
    so vtracer-encoded panel/icon borders look like the lines they are.

    Inner subpaths are dropped — they're holes in the original evenodd fill,
    not visible strokes.
    """
    from svg_to_drawio.parse import split_subpaths, path_bbox
    x0, y0, x1, y1 = p['bbox']
    w, h = x1 - x0, y1 - y0
    if w < 0.1 or h < 0.1:
        return ''
    # Keep only the outer (largest) subpath so we draw the visible boundary,
    # not the hole boundaries inside it.
    expanded = p['expanded']
    subs = split_subpaths(expanded)
    if len(subs) > 1:
        outer = max(subs, key=_polygon_shoelace_area)
        expanded = outer
        nb = path_bbox(outer)
        x0 = nb[0]; y0 = nb[1]; w = nb[2]; h = nb[3]
        if w < 0.1 or h < 0.1:
            return ''
    sx = stencil_size / w if w else 1
    sy = stencil_size / h if h else 1
    parts = [f'<shape aspect="variable" w="{stencil_size}" h="{stencil_size}">',
             '<foreground>',
             '<path>']
    for seg in expanded:
        cmd = seg[0]
        if cmd == 'M':
            parts.append(f'<move x="{(seg[1]-x0)*sx:.4f}" y="{(seg[2]-y0)*sy:.4f}"/>')
        elif cmd == 'L':
            parts.append(f'<line x="{(seg[1]-x0)*sx:.4f}" y="{(seg[2]-y0)*sy:.4f}"/>')
        elif cmd == 'C':
            parts.append(f'<curve x1="{(seg[1]-x0)*sx:.4f}" y1="{(seg[2]-y0)*sy:.4f}" '
                         f'x2="{(seg[3]-x0)*sx:.4f}" y2="{(seg[4]-y0)*sy:.4f}" '
                         f'x3="{(seg[5]-x0)*sx:.4f}" y3="{(seg[6]-y0)*sy:.4f}"/>')
        elif cmd == 'Q':
            parts.append(f'<quad x1="{(seg[1]-x0)*sx:.4f}" y1="{(seg[2]-y0)*sy:.4f}" '
                         f'x2="{(seg[3]-x0)*sx:.4f}" y2="{(seg[4]-y0)*sy:.4f}"/>')
        elif cmd == 'Z':
            parts.append('<close/>')
    parts.append('</path>')
    parts.append('<stroke/>')
    parts.append('</foreground></shape>')
    stencil_xml = ''.join(parts)
    style_parts = [f'shape=stencil({encode_stencil(stencil_xml)})',
                   f'strokeColor={p["fill"]}',
                   'fillColor=none', 'html=1', 'strokeWidth=1']
    if p.get('fill_op', 1.0) < 1.0:
        style_parts.append(f'opacity={int(p["fill_op"]*100)}')
    return (
        f'<mxCell id="{cid}" value="" style="{";".join(style_parts)};" '
        f'vertex="1" parent="1">'
        f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'as="geometry"/></mxCell>'
    )


def _polygon_shoelace_area(sub):
    pts = []
    for s in sub:
        if s[0] in ('M', 'L'):
            pts.append((s[1], s[2]))
        elif s[0] == 'C':
            pts.append((s[5], s[6]))
        elif s[0] == 'Q':
            pts.append((s[3], s[4]))
    n = len(pts); a = 0.0
    for i in range(n):
        xi, yi = pts[i]; xj, yj = pts[(i + 1) % n]
        a += xi * yj - xj * yi
    return abs(a) / 2


def emit_singleton_solid(cid: int, p: dict, stencil_size: int = 1000,
                         min_dim: float = 30.0) -> str:
    """Singleton stencil with inner subpaths dropped when bbox >= min_dim.

    Why: an evenodd path with a hole becomes a 'donut' stencil — when an arrow
    is dragged across that hole, the panel underneath shows through. Keeping
    only the outermost subpath gives a solid stencil.

    Special case: thin-outline paths (where polygon area << bbox area) are
    really strokes encoded as fill — render them with stroke instead of fill
    so they don't blanket whatever's behind their bbox.
    """
    if is_thin_outline(p):
        return emit_thin_outline(cid, p, stencil_size=stencil_size)
    x0, y0, x1, y1 = p['bbox']
    w, h = x1 - x0, y1 - y0
    expanded = p['expanded']
    if w >= min_dim and h >= min_dim:
        subs = split_subpaths(expanded)
        if len(subs) >= 2:
            largest = max(subs, key=_polygon_shoelace_area)
            expanded = largest
            new_bb = path_bbox(largest)
            p = dict(p, expanded=largest,
                     bbox=(new_bb[0], new_bb[1],
                           new_bb[0] + new_bb[2], new_bb[1] + new_bb[3]))
    return emit_stencil(cid, p, stencil_size=stencil_size)


def emit_icon(cid: int, paths: list, bbox: tuple, quantize_threshold: float = 30.0) -> str:
    """Render an icon (≥2 paths) as a single drawio shape=image cell carrying
    a quantized inline SVG. Atomic — drawio cannot split it."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    svg, _ = quantized_icon_svg(paths, bbox, threshold=quantize_threshold)
    style = svg_to_image_cell_style(svg)
    return (
        f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'as="geometry"/></mxCell>'
    )


def emit_text(cid: int, c: dict, fill_color: str, font_family: str = 'DejaVu Sans') -> str:
    """Editable drawio text cell. For vertical clusters (rotation=-90), swap
    width/height so the rotated text isn't truncated by a narrow horizontal
    bbox.
    """
    text = (c.get('text') or '').strip()
    if not text:
        return ''
    x0, y0, x1, y1 = c['bbox']
    w, h = x1 - x0, y1 - y0
    if w < 2 or h < 2:
        return ''
    fs = c['font_size']
    is_vertical = c.get('vertical', False)
    multi_line = c.get('_multi_line') or '\n' in text
    wrap_mode = 'wrap' if multi_line else 'nowrap'
    style_parts = ['text', 'html=1', 'strokeColor=none', 'fillColor=none',
                   'align=left', 'verticalAlign=middle',
                   f'whiteSpace={wrap_mode}', 'rounded=0',
                   f'fontFamily={font_family}', f'fontSize={fs}',
                   f'fontColor={fill_color}']
    bits = 0
    if c.get('bold'):
        bits |= 1
    if bits:
        style_parts.append(f'fontStyle={bits}')
    if is_vertical:
        style_parts.append('rotation=-90')
        char_w = fs * 0.62
        line_h = fs * 1.4
        gw = max(h, char_w * len(text))
        gh = max(w, line_h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        gx, gy = cx - gw / 2, cy - gh / 2
    else:
        pad_x = max(1, int(fs * 0.1)); pad_y = 1
        gx, gy, gw, gh = x0 - pad_x, y0 - pad_y, w + 2 * pad_x, h + 2 * pad_y
    val = (text.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('"', '&quot;'))
    if multi_line:
        # drawio html=1 renders &lt;br&gt; in the value attribute as a real
        # <br> line break. A literal '<' isn't allowed in XML attributes.
        val = val.replace('\n', '&lt;br&gt;')
    style = ';'.join(style_parts) + ';'
    return (
        f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{gx:.1f}" y="{gy:.1f}" width="{gw:.1f}" height="{gh:.1f}" '
        f'as="geometry"/></mxCell>'
    )


def wrap_drawio_xml(cells: list, W: int, H: int, agent: str = 'svg_to_drawio') -> str:
    """Wrap mxCell strings in a complete .drawio XML document."""
    body = '\n        '.join(cells)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<mxfile host="app.diagrams.net" modified="2026-04-25T00:00:00.000Z" '
        f'agent="{agent}" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced" name="traced">\n'
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


def fill_color_for_cluster(c: dict, all_paths_by_idx: dict) -> str:
    """Most common fill color across the cluster's glyph paths."""
    if c.get('glyphs'):
        fills = [gg['path']['fill'] for gg in c['glyphs']]
    else:
        fills = [all_paths_by_idx[pid]['fill']
                 for pid in c.get('glyph_path_ids', [])
                 if pid in all_paths_by_idx]
    return Counter(fills).most_common(1)[0][0] if fills else '#333333'
