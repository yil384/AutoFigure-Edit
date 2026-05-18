"""Detect line-shaped paths and arrowheads, emit native drawio edges."""
import math

__all__ = [
    'is_line_shaped', 'is_arrowhead_shaped', 'line_endpoints', 'line_thickness',
    'attach_arrowhead', 'emit_edge_cell', 'snap_to_shape',
]


def snap_to_shape(point: tuple, shape_index: list,
                  max_dist: float = 18.0):
    """Find the closest shape whose edge is near `point`. `shape_index`
    is a list of dicts: {'id': str, 'bbox': (x0,y0,x1,y1)}.

    Returns (shape_id, ex, ey) where ex/ey are 0-1 ratios indicating
    which side of the shape the point attaches to, or None if no shape
    is within max_dist.
    """
    px, py = point
    best = None
    best_d = max_dist
    for s in shape_index:
        x0, y0, x1, y1 = s['bbox']
        # Closest point ON the bbox to (px, py)
        cx = max(x0, min(px, x1))
        cy = max(y0, min(py, y1))
        d = math.hypot(px - cx, py - cy)
        if d > best_d:
            continue
        # Compute exit/entry ratios. If point is to the LEFT of bbox -> ex=0,
        # to the right -> ex=1, otherwise ratio along width.
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        if px <= x0 + 2:
            ex = 0.0
        elif px >= x1 - 2:
            ex = 1.0
        else:
            ex = (px - x0) / w
        if py <= y0 + 2:
            ey = 0.0
        elif py >= y1 - 2:
            ey = 1.0
        else:
            ey = (py - y0) / h
        # Snap to the NEAREST CARDINAL edge (cleaner connectors than
        # arbitrary midpoints): if px is roughly at the horizontal center
        # of the shape we want ey to be 0 or 1; if py is roughly at the
        # vertical center we want ex to be 0 or 1.
        dx_left = abs(px - x0); dx_right = abs(px - x1)
        dy_top = abs(py - y0); dy_bot = abs(py - y1)
        nearest = min(dx_left, dx_right, dy_top, dy_bot)
        if nearest == dx_left:
            ex, ey = 0.0, max(0.0, min(1.0, (py - y0) / h))
        elif nearest == dx_right:
            ex, ey = 1.0, max(0.0, min(1.0, (py - y0) / h))
        elif nearest == dy_top:
            ex, ey = max(0.0, min(1.0, (px - x0) / w)), 0.0
        else:
            ex, ey = max(0.0, min(1.0, (px - x0) / w)), 1.0
        best_d = d
        best = (s['id'], ex, ey)
    return best


def _extract_polygon_points(segments):
    """Endpoints of every M/L/C/Q segment — used to recover line endpoints
    and arrowhead corner counts from a filled-polygon path."""
    pts = []
    for seg in segments:
        cmd = seg[0]
        if cmd == 'M' or cmd == 'L':
            pts.append((seg[1], seg[2]))
        elif cmd == 'C':
            pts.append((seg[5], seg[6]))
        elif cmd == 'Q':
            pts.append((seg[3], seg[4]))
    return pts


def is_line_shaped(p, max_thickness: float = 5.0, min_aspect: float = 5.0) -> bool:
    """A 'line' is a filled polygon whose short dimension is small AND whose
    aspect ratio is high — vtracer encodes thin strokes this way."""
    x0, y0, x1, y1 = p['bbox']
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return False
    short, long_ = min(w, h), max(w, h)
    if short > max_thickness:
        return False
    return long_ / max(short, 0.1) >= min_aspect


def line_endpoints(p):
    """Two extremes of a line-shaped path along its long axis, centered on
    the perpendicular axis. Returns ((sx,sy),(tx,ty)) or None."""
    x0, y0, x1, y1 = p['bbox']
    w, h = x1 - x0, y1 - y0
    pts = _extract_polygon_points(p['expanded'])
    if not pts:
        return None
    if w >= h:
        cy = (y0 + y1) / 2
        return (
            (min(pts, key=lambda q: q[0])[0], cy),
            (max(pts, key=lambda q: q[0])[0], cy),
        )
    cx = (x0 + x1) / 2
    return (
        (cx, min(pts, key=lambda q: q[1])[1]),
        (cx, max(pts, key=lambda q: q[1])[1]),
    )


def line_thickness(p) -> float:
    x0, y0, x1, y1 = p['bbox']
    return min(x1 - x0, y1 - y0)


def is_arrowhead_shaped(p, max_dim: float = 14.0, max_aspect: float = 2.5) -> bool:
    """Small filled triangle/quad — used to mark arrow tips next to lines."""
    x0, y0, x1, y1 = p['bbox']
    w, h = x1 - x0, y1 - y0
    if w > max_dim or h > max_dim or w < 2 or h < 2:
        return False
    if max(w, h) / max(min(w, h), 0.5) > max_aspect:
        return False
    pts = _extract_polygon_points(p['expanded'])
    return 3 <= len(pts) <= 8


def attach_arrowhead(line_p, arrowhead_paths, max_dist: float = 6.0):
    """Pair arrowheads with line endpoints by Euclidean distance.
    Returns (start_arrow, end_arrow), each None if no nearby arrowhead."""
    eps = line_endpoints(line_p)
    if eps is None:
        return (None, None)
    p1, p2 = eps
    start = end = None
    for ahp in arrowhead_paths:
        ax = (ahp['bbox'][0] + ahp['bbox'][2]) / 2
        ay = (ahp['bbox'][1] + ahp['bbox'][3]) / 2
        d1 = math.hypot(ax - p1[0], ay - p1[1])
        d2 = math.hypot(ax - p2[0], ay - p2[1])
        if d1 < d2 and d1 < max_dist and start is None:
            start = ahp
        elif d2 <= d1 and d2 < max_dist and end is None:
            end = ahp
    return (start, end)


def emit_edge_cell(cid: int, line_p, start_arrow=None, end_arrow=None,
                   shape_index=None) -> str:
    """Native drawio edge. When `shape_index` is provided and either
    endpoint snaps to a native shape, emit `source`/`target` ID + entry/exit
    so the edge follows when the user drags the shape. Otherwise fall back
    to absolute mxPoint coords (legacy behavior).
    """
    eps = line_endpoints(line_p)
    if eps is None:
        return ''
    (sx, sy), (tx, ty) = eps
    sw = max(0.5, line_thickness(line_p))
    color = line_p['fill']
    parts = ['html=1', 'rounded=0', f'strokeColor={color}', f'strokeWidth={sw:.2f}']
    parts.append('endArrow=classic;endFill=1' if end_arrow is not None else 'endArrow=none')
    if start_arrow is not None:
        parts.append('startArrow=classic;startFill=1')
    src_snap = tgt_snap = None
    if shape_index:
        src_snap = snap_to_shape((sx, sy), shape_index)
        tgt_snap = snap_to_shape((tx, ty), shape_index)
        # Reject self-loops (both endpoints inside the same shape) — those
        # are usually just stroke paths around a box's perimeter, not true
        # connectors. Drop both bindings and fall back to absolute coords.
        if (src_snap is not None and tgt_snap is not None
                and src_snap[0] == tgt_snap[0]):
            src_snap = tgt_snap = None
    if src_snap is not None:
        _, ex, ey = src_snap
        parts.append(f'exitX={ex:.3f};exitY={ey:.3f};exitDx=0;exitDy=0')
    if tgt_snap is not None:
        _, ex, ey = tgt_snap
        parts.append(f'entryX={ex:.3f};entryY={ey:.3f};entryDx=0;entryDy=0')
    style = ';'.join(parts) + ';'
    edge_attrs = 'edge="1" parent="1"'
    if src_snap is not None:
        edge_attrs += f' source="{src_snap[0]}"'
    if tgt_snap is not None:
        edge_attrs += f' target="{tgt_snap[0]}"'
    geom_inner = ''
    if src_snap is None:
        geom_inner += f'<mxPoint x="{sx:.2f}" y="{sy:.2f}" as="sourcePoint"/>'
    if tgt_snap is None:
        geom_inner += f'<mxPoint x="{tx:.2f}" y="{ty:.2f}" as="targetPoint"/>'
    return (
        f'<mxCell id="{cid}" value="" style="{style}" {edge_attrs}>'
        f'<mxGeometry relative="1" as="geometry">'
        f'{geom_inner}'
        f'</mxGeometry></mxCell>'
    )
