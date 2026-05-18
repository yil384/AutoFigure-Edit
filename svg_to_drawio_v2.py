"""svg_to_drawio_v2.py - Faithful SVG → drawio XML converter.

Handles SVG <path> elements with M/L/H/V/C/S/Q/T/A/Z commands (absolute + relative).
Each path becomes a drawio `shape=stencil(...)` mxCell preserving:
 - Multi-subpath paths (holes rendered via drawio's default winding; visually
   correct because SVG paths in the output are typically painted in order and
   subsequent paths cover "hole" regions)
 - Arc (A) commands converted to cubic Bezier via the W3C arc-to-bezier algorithm
 - Fill color
 - Exact coordinates (bbox + normalized stencil coords)

Output: drawio file openable in diagrams.net where each SVG <path> is an
individually-selectable, editable vector shape.

Usage:
    python svg_to_drawio_v2.py input.svg [-o output.drawio]
"""
from __future__ import annotations

import re
import sys
import math
import base64
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET


# ----------------------------------------------------------------------------
# SVG path parsing
# ----------------------------------------------------------------------------

def _tokenize(d: str):
    """Yield tokens for an SVG path d attribute.

    Commands are single letters (M/m/L/l/H/h/V/v/C/c/S/s/Q/q/T/t/A/a/Z/z).
    Numbers can be separated by comma or whitespace, and can be concatenated
    with a leading minus (e.g., "3-4" → ["3","-4"]). Flags in A/a are 0 or 1.
    """
    # Pattern: commands, floats (including exponents)
    token_re = re.compile(
        r'([MmLlHhVvCcSsQqTtAaZz])'
        r'|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    )
    for m in token_re.finditer(d):
        if m.group(1):
            yield m.group(1)
        else:
            yield m.group(2)


def _parse_numbers(tokens: list, i: int, n: int):
    """Grab n floats from tokens[i:]. Returns (values, new_i)."""
    vals = [float(tokens[i + k]) for k in range(n)]
    return vals, i + n


ARG_COUNTS = {
    'M': 2, 'm': 2,
    'L': 2, 'l': 2,
    'H': 1, 'h': 1,
    'V': 1, 'v': 1,
    'C': 6, 'c': 6,
    'S': 4, 's': 4,
    'Q': 4, 'q': 4,
    'T': 2, 't': 2,
    'A': 7, 'a': 7,
    'Z': 0, 'z': 0,
}


def parse_path(d: str):
    """Parse SVG path into list of (cmd, *args) absolute-coordinate subpath items.

    Emits segments as:
      ('M', x, y)
      ('L', x, y)
      ('C', x1, y1, x2, y2, x, y)
      ('Q', x1, y1, x, y)
      ('A', rx, ry, x_axis_rot, large_arc_flag, sweep_flag, x, y)
      ('Z',)

    Handles relative commands by maintaining current point and resolves all
    output as absolute coordinates.
    """
    tokens = list(_tokenize(d))
    i = 0
    out = []
    cx = cy = 0.0  # current point
    start_x = start_y = 0.0  # current subpath start
    prev_cmd = None
    prev_ctrl = None  # last C/S control for S smoothing, or Q/T for T

    def consume(cmd: str, args: list):
        nonlocal cx, cy, start_x, start_y, prev_ctrl
        if cmd in ('M', 'm'):
            x, y = args
            if cmd == 'm':
                x += cx; y += cy
            cx, cy = x, y
            start_x, start_y = x, y
            out.append(('M', x, y))
            prev_ctrl = None
        elif cmd in ('L', 'l'):
            x, y = args
            if cmd == 'l':
                x += cx; y += cy
            cx, cy = x, y
            out.append(('L', x, y))
            prev_ctrl = None
        elif cmd in ('H', 'h'):
            x = args[0]
            if cmd == 'h':
                x += cx
            cx = x
            out.append(('L', cx, cy))
            prev_ctrl = None
        elif cmd in ('V', 'v'):
            y = args[0]
            if cmd == 'v':
                y += cy
            cy = y
            out.append(('L', cx, cy))
            prev_ctrl = None
        elif cmd in ('C', 'c'):
            x1, y1, x2, y2, x, y = args
            if cmd == 'c':
                x1 += cx; y1 += cy
                x2 += cx; y2 += cy
                x += cx; y += cy
            out.append(('C', x1, y1, x2, y2, x, y))
            prev_ctrl = (x2, y2)
            cx, cy = x, y
        elif cmd in ('S', 's'):
            x2, y2, x, y = args
            if cmd == 's':
                x2 += cx; y2 += cy
                x += cx; y += cy
            # Smooth: x1 = reflection of prev C control around current point
            if prev_cmd_val() in ('C', 'c', 'S', 's'):
                x1 = 2 * cx - prev_ctrl[0]
                y1 = 2 * cy - prev_ctrl[1]
            else:
                x1, y1 = cx, cy
            out.append(('C', x1, y1, x2, y2, x, y))
            prev_ctrl = (x2, y2)
            cx, cy = x, y
        elif cmd in ('Q', 'q'):
            x1, y1, x, y = args
            if cmd == 'q':
                x1 += cx; y1 += cy
                x += cx; y += cy
            out.append(('Q', x1, y1, x, y))
            prev_ctrl = (x1, y1)
            cx, cy = x, y
        elif cmd in ('T', 't'):
            x, y = args
            if cmd == 't':
                x += cx; y += cy
            if prev_cmd_val() in ('Q', 'q', 'T', 't'):
                x1 = 2 * cx - prev_ctrl[0]
                y1 = 2 * cy - prev_ctrl[1]
            else:
                x1, y1 = cx, cy
            out.append(('Q', x1, y1, x, y))
            prev_ctrl = (x1, y1)
            cx, cy = x, y
        elif cmd in ('A', 'a'):
            rx, ry, phi, fa, fs, x, y = args
            if cmd == 'a':
                x += cx; y += cy
            out.append(('A', rx, ry, phi, int(round(fa)), int(round(fs)), x, y))
            cx, cy = x, y
            prev_ctrl = None
        elif cmd in ('Z', 'z'):
            out.append(('Z',))
            cx, cy = start_x, start_y
            prev_ctrl = None

    # need to remember last command for S/T smoothing
    _last = [None]
    def prev_cmd_val():
        return _last[0]

    while i < len(tokens):
        tok = tokens[i]
        if not tok.isalpha():
            # implicit continuation of previous command
            if prev_cmd is None:
                i += 1
                continue
            cmd = 'L' if prev_cmd == 'M' else ('l' if prev_cmd == 'm' else prev_cmd)
            count = ARG_COUNTS.get(cmd, 0)
            if count == 0:
                i += 1
                continue
            args, i = _parse_numbers(tokens, i, count)
            consume(cmd, args)
            _last[0] = cmd
            continue

        cmd = tok
        i += 1
        count = ARG_COUNTS[cmd]
        if count == 0:
            consume(cmd, [])
            _last[0] = cmd
            prev_cmd = cmd
        else:
            # A repeats; parse blocks of `count` args until next command
            while i < len(tokens) and not tokens[i].isalpha():
                args, i = _parse_numbers(tokens, i, count)
                consume(cmd, args)
                _last[0] = cmd
                prev_cmd = cmd
    return out


# ----------------------------------------------------------------------------
# Arc -> Cubic Bezier conversion (W3C SVG 1.1 Appendix F.6.5)
# ----------------------------------------------------------------------------

def _arc_to_cubics(x1, y1, rx, ry, phi_deg, fa, fs, x2, y2):
    """Convert SVG arc A to a list of cubic Bezier segments.

    Returns list of (x1, y1, x2, y2, x, y) for each cubic segment where (x, y)
    is the end point and (x1,y1), (x2,y2) are the two control points.

    Algorithm: https://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
    """
    rx = abs(rx); ry = abs(ry)
    if rx == 0 or ry == 0:
        return [(x1, y1, x2, y2, x2, y2)]  # degenerate → straight line approximated
    phi = math.radians(phi_deg)
    cos_phi = math.cos(phi); sin_phi = math.sin(phi)

    # Step 1: (x1', y1') — transformed start point
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Step 2: correct radii if too small
    rx_sq = rx * rx; ry_sq = ry * ry
    x1p_sq = x1p * x1p; y1p_sq = y1p * y1p
    radii_check = x1p_sq / rx_sq + y1p_sq / ry_sq
    if radii_check > 1:
        s = math.sqrt(radii_check)
        rx *= s; ry *= s
        rx_sq = rx * rx; ry_sq = ry * ry

    # Step 3: compute (cx', cy')
    sign = -1 if fa == fs else 1
    num = rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq
    denom = rx_sq * y1p_sq + ry_sq * x1p_sq
    if denom == 0:
        coef = 0.0
    else:
        coef = sign * math.sqrt(max(0.0, num / denom))
    cxp = coef * (rx * y1p / ry)
    cyp = coef * -(ry * x1p / rx)

    # Step 4: compute (cx, cy)
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2

    # Step 5: compute theta1 and delta_theta
    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        mag = math.sqrt(ux*ux + uy*uy) * math.sqrt(vx*vx + vy*vy)
        if mag == 0:
            return 0.0
        val = max(-1.0, min(1.0, dot / mag))
        ang = math.acos(val)
        if ux * vy - uy * vx < 0:
            ang = -ang
        return ang

    theta1 = angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry,
                   (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if fs == 0 and dtheta > 0:
        dtheta -= 2 * math.pi
    elif fs == 1 and dtheta < 0:
        dtheta += 2 * math.pi

    # Split into segments of <= pi/2
    num_segs = int(math.ceil(abs(dtheta) / (math.pi / 2)))
    if num_segs == 0:
        num_segs = 1
    seg_angle = dtheta / num_segs

    cubics = []
    t = theta1
    for _ in range(num_segs):
        t_end = t + seg_angle
        # Unit circle cubic Bezier approximation for arc from t → t+seg_angle
        # Alpha = 4/3 * tan(seg/4)
        alpha = (4.0 / 3.0) * math.tan(seg_angle / 4.0)

        # Endpoints on unit circle (ellipse before affine)
        cos_t = math.cos(t); sin_t = math.sin(t)
        cos_te = math.cos(t_end); sin_te = math.sin(t_end)

        # Control points on unit circle
        p0 = (cos_t, sin_t)
        p1 = (cos_t - alpha * sin_t, sin_t + alpha * cos_t)
        p2 = (cos_te + alpha * sin_te, sin_te - alpha * cos_te)
        p3 = (cos_te, sin_te)

        def transform(p):
            px, py = p
            # scale by radii
            ex, ey = px * rx, py * ry
            # rotate by phi
            rxp = cos_phi * ex - sin_phi * ey
            ryp = sin_phi * ex + cos_phi * ey
            # translate by center
            return (rxp + cx, ryp + cy)

        _p1 = transform(p1)
        _p2 = transform(p2)
        _p3 = transform(p3)
        cubics.append((_p1[0], _p1[1], _p2[0], _p2[1], _p3[0], _p3[1]))
        t = t_end
    return cubics


# ----------------------------------------------------------------------------
# Path expansion: convert all commands to M/L/C/Q/Z with arc→cubic
# ----------------------------------------------------------------------------

def expand_path(segments):
    """Expand path segments converting A (arcs) to sequences of cubic Beziers.

    Returns list of segments using only M/L/C/Q/Z commands.
    """
    out = []
    cx = cy = 0.0
    for seg in segments:
        cmd = seg[0]
        if cmd == 'M':
            _, x, y = seg
            out.append(('M', x, y))
            cx, cy = x, y
        elif cmd == 'L':
            _, x, y = seg
            out.append(('L', x, y))
            cx, cy = x, y
        elif cmd == 'C':
            _, x1, y1, x2, y2, x, y = seg
            out.append(('C', x1, y1, x2, y2, x, y))
            cx, cy = x, y
        elif cmd == 'Q':
            _, x1, y1, x, y = seg
            out.append(('Q', x1, y1, x, y))
            cx, cy = x, y
        elif cmd == 'A':
            _, rx, ry, phi, fa, fs, x, y = seg
            cubics = _arc_to_cubics(cx, cy, rx, ry, phi, fa, fs, x, y)
            for c in cubics:
                out.append(('C', c[0], c[1], c[2], c[3], c[4], c[5]))
            cx, cy = x, y
        elif cmd == 'Z':
            out.append(('Z',))
    return out


# ----------------------------------------------------------------------------
# Bounding box
# ----------------------------------------------------------------------------

def path_bbox(segments):
    """Compute a conservative bbox using the segment control/end points only.

    This slightly over-estimates for curves (control points extend beyond the
    curve extent), but it's safe and fast.
    """
    xs, ys = [], []
    cx = cy = 0.0
    start_x = start_y = 0.0
    for seg in segments:
        cmd = seg[0]
        if cmd == 'M':
            _, x, y = seg
            cx, cy = x, y; start_x, start_y = x, y
            xs.append(x); ys.append(y)
        elif cmd == 'L':
            _, x, y = seg
            cx, cy = x, y
            xs.append(x); ys.append(y)
        elif cmd == 'C':
            _, x1, y1, x2, y2, x, y = seg
            xs.extend([x1, x2, x]); ys.extend([y1, y2, y])
            cx, cy = x, y
        elif cmd == 'Q':
            _, x1, y1, x, y = seg
            xs.extend([x1, x]); ys.extend([y1, y])
            cx, cy = x, y
        elif cmd == 'Z':
            cx, cy = start_x, start_y
    if not xs:
        return (0, 0, 1, 1)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return (x0, y0, max(1e-6, x1 - x0), max(1e-6, y1 - y0))


# ----------------------------------------------------------------------------
# Stencil XML emission (drawio format)
# ----------------------------------------------------------------------------

def segments_to_stencil(segments, bbox, stencil_size=1000):
    """Build drawio stencil XML for a path.

    Coords in stencil are normalized to stencil_size x stencil_size then the
    mxCell's geometry scales it to actual pixels. Using a larger stencil_size
    (e.g. 1000) preserves subpixel precision.

    Returns stencil XML string.
    """
    x0, y0, w, h = bbox
    sx = stencil_size / w
    sy = stencil_size / h

    def fx(x):
        return (x - x0) * sx
    def fy(y):
        return (y - y0) * sy

    parts = [f'<shape aspect="variable" w="{stencil_size}" h="{stencil_size}">',
             '<foreground>',
             '<path>']
    for seg in segments:
        cmd = seg[0]
        if cmd == 'M':
            _, x, y = seg
            parts.append(f'<move x="{fx(x):.4f}" y="{fy(y):.4f}"/>')
        elif cmd == 'L':
            _, x, y = seg
            parts.append(f'<line x="{fx(x):.4f}" y="{fy(y):.4f}"/>')
        elif cmd == 'C':
            _, x1, y1, x2, y2, x, y = seg
            parts.append(f'<curve x1="{fx(x1):.4f}" y1="{fy(y1):.4f}" '
                         f'x2="{fx(x2):.4f}" y2="{fy(y2):.4f}" '
                         f'x3="{fx(x):.4f}" y3="{fy(y):.4f}"/>')
        elif cmd == 'Q':
            _, x1, y1, x, y = seg
            parts.append(f'<quad x1="{fx(x1):.4f}" y1="{fy(y1):.4f}" '
                         f'x2="{fx(x):.4f}" y2="{fy(y):.4f}"/>')
        elif cmd == 'Z':
            parts.append('<close/>')
    parts.append('</path>')
    parts.append('<fill/>')
    parts.append('</foreground>')
    parts.append('</shape>')
    return ''.join(parts)


def encode_stencil(stencil_xml: str) -> str:
    """Encode stencil XML as drawio-style raw-deflate + base64."""
    compressed = zlib.compress(stencil_xml.encode('utf-8'), 9)
    # Strip zlib 2-byte header and 4-byte Adler32 trailer → raw deflate
    raw = compressed[2:-4]
    return base64.b64encode(raw).decode('ascii')


# ----------------------------------------------------------------------------
# SVG document traversal
# ----------------------------------------------------------------------------

_NS_RE = re.compile(r'^\{[^}]+\}')


def _strip_ns(tag: str) -> str:
    return _NS_RE.sub('', tag)


def _escape_style(s: str) -> str:
    return s.replace('"', '&quot;')


def split_subpaths(segments):
    """Split segments at each M (moveto) into a list of subpaths.
    Each subpath is a list of segments starting with M.
    """
    subs = []
    cur = []
    for seg in segments:
        if seg[0] == 'M' and cur:
            subs.append(cur)
            cur = [seg]
        else:
            cur.append(seg)
    if cur:
        subs.append(cur)
    return subs


def convert_svg(svg_path: str, drawio_path: str,
                stencil_size: int = 1000) -> int:
    """Convert SVG file to drawio XML. Returns number of cells emitted.

    Each SVG <path> → one drawio mxCell with `shape=stencil(...)`. Subpaths
    (multiple M commands inside a single path) are kept together in the
    stencil to preserve the evenodd hole rendering via the source's
    opposite-winding convention (which drawio's default nonzero fill handles
    correctly when outer is CW and holes are CCW).
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    w_attr = root.get('width', '1376')
    h_attr = root.get('height', '768')
    def strip_unit(s):
        m = re.match(r'^\s*([0-9.]+)', s)
        return float(m.group(1)) if m else 0.0
    W = int(strip_unit(w_attr))
    H = int(strip_unit(h_attr))
    if W == 0 or H == 0:
        vb = root.get('viewBox', '').split()
        if len(vb) == 4:
            W = int(float(vb[2]))
            H = int(float(vb[3]))

    cells = []
    cid = 100
    skipped = 0

    for elem in root.iter():
        tag = _strip_ns(elem.tag)
        if tag != 'path':
            continue
        d = elem.get('d', '')
        if not d.strip():
            skipped += 1
            continue
        fill = elem.get('fill', '#000000')
        if fill == 'none':
            skipped += 1
            continue
        fill_opacity = elem.get('fill-opacity', '1')
        try:
            fill_op = float(fill_opacity)
        except ValueError:
            fill_op = 1.0

        segments = parse_path(d)
        if not segments:
            skipped += 1
            continue
        expanded = expand_path(segments)
        bbox = path_bbox(expanded)
        x0, y0, w, h = bbox
        if w < 0.1 or h < 0.1:
            skipped += 1
            continue
        stencil_xml = segments_to_stencil(expanded, bbox, stencil_size=stencil_size)
        stencil_b64 = encode_stencil(stencil_xml)
        style_parts = [
            f'shape=stencil({stencil_b64})',
            f'fillColor={fill}',
            'strokeColor=none',
            'html=1',
        ]
        if fill_op < 1.0:
            style_parts.append(f'opacity={int(fill_op * 100)}')
        style = ';'.join(style_parts) + ';'
        cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" '
            f'width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
            f'</mxCell>'
        )
        cid += 1

    body = '\n        '.join(cells)
    drawio = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-04-24T00:00:00.000Z" '
        'agent="svg_to_drawio_v2.py" version="24.7.0" type="device">\n'
        '  <diagram id="svg_traced_01" name="traced">\n'
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
    print(f'converted {len(cells)} cells (skipped {skipped}) → {drawio_path}')
    return len(cells)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='input SVG path')
    ap.add_argument('-o', '--output', default=None, help='output drawio path')
    args = ap.parse_args()
    out = args.output or Path(args.input).with_suffix('.drawio').as_posix()
    convert_svg(args.input, out)
