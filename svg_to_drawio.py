"""svg_to_drawio.py – Deterministic SVG → draw.io XML converter.

Parses an SVG file produced by the autofigure pipeline and converts each
element (rect, text, path, circle, line, image, group) into an independent
editable mxCell in draw.io format.  Coordinates are preserved exactly.

Handles:
  - Inline style attributes (fill="...", stroke="...")
  - CSS class-based styling (<style> blocks with .className)
  - Nested <g> groups with transform="translate(x, y)"
  - AF-labeled placeholder groups (<g id="AF01">)

Usage:
    python svg_to_drawio.py input.svg [-o output.drawio]
"""

import os, re, sys
from xml.etree import ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"


# ---------------------------------------------------------------------------
# SVG path "d" attribute parser
# ---------------------------------------------------------------------------

def _parse_path_commands(d: str):
    """Parse SVG path 'd' into [(command, [float, ...]), ...]."""
    tokens = re.findall(
        r'[MmLlHhVvQqCcSsTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?',
        d,
    )
    cmds = []
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            args = []
            while i < len(tokens) and not tokens[i].isalpha():
                args.append(float(tokens[i]))
                i += 1
            cmds.append((cmd, args))
        else:
            i += 1
    return cmds


def _path_to_points(d: str):
    """Convert SVG path 'd' to a list of (x, y) waypoints.

    Gracefully handles malformed paths (odd argument counts, missing
    coordinates) which can occur when LLMs generate SVG.
    """
    cmds = _parse_path_commands(d)
    pts = []
    cx, cy = 0.0, 0.0

    def _safe_pairs(args, start=0, step=2):
        """Yield (idx, x, y) for complete coordinate pairs only."""
        for j in range(start, len(args) - 1, step):
            yield j, args[j], args[j + 1]

    for cmd, args in cmds:
        try:
            if cmd == "M":
                if len(args) < 2:
                    continue
                cx, cy = args[0], args[1]
                pts.append((cx, cy))
                for _, x, y in _safe_pairs(args, 2):
                    cx, cy = x, y
                    pts.append((cx, cy))
            elif cmd == "m":
                if len(args) < 2:
                    continue
                if not pts:
                    cx, cy = args[0], args[1]
                else:
                    cx += args[0]
                    cy += args[1]
                pts.append((cx, cy))
                for _, x, y in _safe_pairs(args, 2):
                    cx += x
                    cy += y
                    pts.append((cx, cy))
            elif cmd == "L":
                for _, x, y in _safe_pairs(args):
                    cx, cy = x, y
                    pts.append((cx, cy))
            elif cmd == "l":
                for _, x, y in _safe_pairs(args):
                    cx += x
                    cy += y
                    pts.append((cx, cy))
            elif cmd == "H":
                for v in args:
                    cx = v
                    pts.append((cx, cy))
            elif cmd == "h":
                for v in args:
                    cx += v
                    pts.append((cx, cy))
            elif cmd == "V":
                for v in args:
                    cy = v
                    pts.append((cx, cy))
            elif cmd == "v":
                for v in args:
                    cy += v
                    pts.append((cx, cy))
            elif cmd == "Q":
                for j in range(0, len(args) - 3, 4):
                    qx, qy = args[j], args[j + 1]
                    cx, cy = args[j + 2], args[j + 3]
                    pts.append((qx, qy))
                    pts.append((cx, cy))
            elif cmd == "q":
                for j in range(0, len(args) - 3, 4):
                    qx, qy = cx + args[j], cy + args[j + 1]
                    cx, cy = cx + args[j + 2], cy + args[j + 3]
                    pts.append((qx, qy))
                    pts.append((cx, cy))
            elif cmd == "C":
                for j in range(0, len(args) - 5, 6):
                    cx, cy = args[j + 4], args[j + 5]
                    pts.append((cx, cy))
            elif cmd == "c":
                for j in range(0, len(args) - 5, 6):
                    cx, cy = cx + args[j + 4], cy + args[j + 5]
                    pts.append((cx, cy))
            elif cmd in ("Z", "z"):
                if pts:
                    pts.append(pts[0])
        except (IndexError, TypeError):
            # Skip malformed path segment
            continue
    return pts


# ---------------------------------------------------------------------------
# CSS class resolver
# ---------------------------------------------------------------------------

def _parse_css_classes(root):
    """Extract CSS class→property map from <style> blocks in the SVG."""
    css = {}
    for style_elem in root.iter(f"{NS}style"):
        text = style_elem.text or ""
        # Parse rules like: .fill-black { fill: #000; }
        for m in re.finditer(r'\.([\w-]+)\s*\{([^}]*)\}', text):
            cls_name = m.group(1)
            props = {}
            for prop in m.group(2).split(";"):
                prop = prop.strip()
                if ":" in prop:
                    k, v = prop.split(":", 1)
                    props[k.strip()] = v.strip()
            css[cls_name] = props
    return css


def _resolve_classes(elem, css_map):
    """Resolve CSS classes on element into inline-equivalent properties."""
    cls_attr = elem.get("class", "")
    if not cls_attr:
        return {}
    resolved = {}
    for cls in cls_attr.split():
        if cls in css_map:
            resolved.update(css_map[cls])
    return resolved


def _get_prop(elem, prop_name, css_props, default=None):
    """Get a property: inline attribute wins, then CSS class, then default."""
    val = elem.get(prop_name)
    if val is not None:
        return val
    # Map CSS property names to SVG attribute names
    css_name = prop_name.replace("_", "-")
    if css_name in css_props:
        return css_props[css_name]
    return default


# ---------------------------------------------------------------------------
# Transform handling
# ---------------------------------------------------------------------------

def _parse_translate(transform_attr):
    """Extract (tx, ty) from transform="translate(x, y)" string."""
    if not transform_attr:
        return (0.0, 0.0)
    m = re.search(r'translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)', transform_attr)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    m = re.search(r'translate\(\s*([-\d.]+)\s*\)', transform_attr)
    if m:
        return (float(m.group(1)), 0.0)
    return (0.0, 0.0)


def _build_transform_map(root):
    """Build element-id → accumulated (tx, ty) offset map by walking the tree."""
    offsets = {}  # id(elem) -> (tx, ty)

    def _walk(elem, parent_tx, parent_ty):
        tx, ty = _parse_translate(elem.get("transform"))
        abs_tx = parent_tx + tx
        abs_ty = parent_ty + ty
        offsets[id(elem)] = (abs_tx, abs_ty)
        for child in elem:
            _walk(child, abs_tx, abs_ty)

    _walk(root, 0.0, 0.0)
    return offsets


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fl(elem, attr, default=0.0) -> float:
    v = elem.get(attr)
    if v is None:
        return default
    # Strip units like "px", handle percentages
    v = v.strip()
    if v.endswith("%"):
        return default  # can't resolve percentages without context
    v = re.sub(r'px$', '', v)
    try:
        return float(v)
    except ValueError:
        return default


def _strip(tag: str) -> str:
    return tag.replace(NS, "")


# ---------------------------------------------------------------------------
# Per-element converters  (return dict or None)
# All coordinates should be ABSOLUTE (transform already applied by caller)
# ---------------------------------------------------------------------------

def _convert_rect(elem, css, pw, ph):
    x, y = _fl(elem, "x"), _fl(elem, "y")
    w, h = _fl(elem, "width"), _fl(elem, "height")

    if w <= 0 or h <= 0:
        return None

    # Skip full-page background rect (no stroke, matches page size)
    fill_v = _get_prop(elem, "fill", css, "none")
    stroke_v = _get_prop(elem, "stroke", css, "none")
    if abs(w - pw) < 2 and abs(h - ph) < 2 and stroke_v == "none":
        return None

    sw = _get_prop(elem, "stroke-width", css, "1")
    rx = _fl(elem, "rx")

    parts = []
    if rx > 0 and min(w, h) > 0:
        parts.append("rounded=1")
        arc = min(100, int(rx / min(w, h) * 200))
        parts.append(f"arcSize={arc}")
    else:
        parts.append("rounded=0")

    parts.append(f"fillColor={fill_v}" if fill_v != "none" else "fillColor=none")
    if stroke_v != "none":
        parts += [f"strokeColor={stroke_v}", f"strokeWidth={sw}"]
    else:
        parts.append("strokeColor=none")

    return dict(vertex=True, value="", style=";".join(parts) + ";",
                x=x, y=y, w=w, h=h)


def _convert_text(elem, css):
    text = elem.text or ""
    if not text.strip():
        return None

    x, y = _fl(elem, "x"), _fl(elem, "y")
    anchor = _get_prop(elem, "text-anchor", css, "start")
    fs_raw = _get_prop(elem, "font-size", css, "16")
    font_size = float(re.sub(r'px$', '', str(fs_raw)))
    font_weight = _get_prop(elem, "font-weight", css, "normal")
    fill = _get_prop(elem, "fill", css, "#000000")
    family = (_get_prop(elem, "font-family", css, "Arial") or "Arial").split(",")[0].strip()

    char_w = font_size * 0.55
    tw = max(len(text) * char_w, font_size * 2)
    th = font_size * 1.3

    if anchor == "middle":
        gx = x - tw / 2
        align = "center"
    elif anchor == "end":
        gx = x - tw
        align = "right"
    else:
        gx = x
        align = "left"

    gy = y - font_size * 0.85
    fs_bit = 1 if font_weight == "bold" else 0

    style = (
        f"text;html=0;align={align};verticalAlign=middle;"
        f"fontSize={int(font_size)};fontFamily={family};"
        f"fontColor={fill};fontStyle={fs_bit};"
        f"strokeColor=none;fillColor=none;overflow=visible;whiteSpace=nowrap;"
    )
    return dict(vertex=True, value=_esc(text), style=style,
                x=gx, y=gy, w=tw, h=th)


def _convert_circle(elem, css):
    cx_v, cy_v = _fl(elem, "cx"), _fl(elem, "cy")
    r = _fl(elem, "r", 5)
    if r <= 0:
        return None
    fill = _get_prop(elem, "fill", css, "#000000")
    stroke = _get_prop(elem, "stroke", css, "none")

    parts = ["ellipse"]
    parts.append(f"fillColor={fill}" if fill != "none" else "fillColor=none")
    if stroke != "none":
        sw = _get_prop(elem, "stroke-width", css, "1")
        parts += [f"strokeColor={stroke}", f"strokeWidth={sw}"]
    else:
        parts.append("strokeColor=none")

    return dict(vertex=True, value="", style=";".join(parts) + ";",
                x=cx_v - r, y=cy_v - r, w=r * 2, h=r * 2)


def _convert_line(elem, css):
    x1, y1 = _fl(elem, "x1"), _fl(elem, "y1")
    x2, y2 = _fl(elem, "x2"), _fl(elem, "y2")
    stroke = _get_prop(elem, "stroke", css, "#000000")
    sw = _get_prop(elem, "stroke-width", css, "1")
    has_arrow = elem.get("marker-end") is not None

    parts = [f"strokeColor={stroke}", f"strokeWidth={sw}"]
    parts.append("endArrow=classic;endFill=1" if has_arrow else "endArrow=none")
    parts.append("startArrow=none")

    return dict(edge=True, value="", style=";".join(parts) + ";",
                source_pt=(x1, y1), target_pt=(x2, y2), waypoints=[])


def _convert_path(elem, css):
    d = elem.get("d", "")
    if not d:
        return None

    fill = _get_prop(elem, "fill", css, "none")
    stroke = _get_prop(elem, "stroke", css, "none")

    # Skip filled shapes inside <defs> (marker arrow polygons)
    if fill != "none" and stroke == "none":
        return None

    # Skip stroked paths with fill=none but stroke=none (empty)
    if stroke == "none" and fill == "none":
        return None

    sw = _get_prop(elem, "stroke-width", css, "1")
    marker_end = elem.get("marker-end", "")
    marker_start = elem.get("marker-start", "")

    pts = _path_to_points(d)
    if len(pts) < 2:
        return None

    has_curve = any(c in ("Q", "q", "C", "c") for c, _ in _parse_path_commands(d))

    parts = [f"strokeColor={stroke}", f"strokeWidth={sw}"]
    if has_curve:
        parts.append("curved=1")

    if "url(#" in marker_end:
        parts += ["endArrow=classic", "endFill=1"]
    else:
        parts.append("endArrow=none")

    if "url(#" in marker_start:
        parts += ["startArrow=classic", "startFill=1"]
    else:
        parts.append("startArrow=none")

    return dict(edge=True, value="", style=";".join(parts) + ";",
                source_pt=pts[0], target_pt=pts[-1],
                waypoints=pts[1:-1] if len(pts) > 2 else [])


def _convert_polyline(elem, css):
    """Convert <polyline> to edge with waypoints."""
    points_attr = elem.get("points", "")
    if not points_attr:
        return None
    nums = re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)', points_attr)
    pts = [(float(nums[i]), float(nums[i+1])) for i in range(0, len(nums) - 1, 2)]
    if len(pts) < 2:
        return None

    fill = _get_prop(elem, "fill", css, "none")
    stroke = _get_prop(elem, "stroke", css, "#000000")
    sw = _get_prop(elem, "stroke-width", css, "1")

    parts = [f"strokeColor={stroke}", f"strokeWidth={sw}", "endArrow=none", "startArrow=none"]
    if fill != "none":
        parts.append(f"fillColor={fill}")

    return dict(edge=True, value="", style=";".join(parts) + ";",
                source_pt=pts[0], target_pt=pts[-1],
                waypoints=pts[1:-1] if len(pts) > 2 else [])


def _convert_image(elem, css):
    x, y = _fl(elem, "x"), _fl(elem, "y")
    w, h = _fl(elem, "width", 100), _fl(elem, "height", 100)
    href = elem.get("href") or elem.get(f"{XLINK}href", "")
    svg_id = elem.get("id", "")

    if not href:
        return None

    # draw.io style parser splits on ";", so ";base64" in data URIs
    # breaks parsing.  draw.io uses "data:image/png," (no ;base64).
    drawio_href = href.replace(";base64,", ",")

    style = (
        f"shape=image;verticalLabelPosition=bottom;"
        f"labelBackgroundColor=default;verticalAlign=top;"
        f"aspect=fixed;imageAspect=0;"
        f"image={drawio_href}"
    )
    return dict(vertex=True, value="", style=style,
                x=x, y=y, w=w, h=h, svg_id=svg_id)


def _convert_af_group(g_elem, css, tx, ty):
    """Convert an <AF##> placeholder group to a gray rect mxCell."""
    gid = g_elem.get("id", "")
    rect = g_elem.find(f"{NS}rect")
    txt = g_elem.find(f"{NS}text")

    if rect is None:
        return None

    x = _fl(rect, "x") + tx
    y = _fl(rect, "y") + ty
    w, h = _fl(rect, "width", 100), _fl(rect, "height", 100)
    label = (txt.text or "") if txt is not None else ""

    # Adaptive font size based on placeholder size
    fs = max(8, min(60, int(min(w, h) * 0.4)))

    style = (
        f"rounded=0;fillColor=#808080;strokeColor=#000000;strokeWidth=2;"
        f"fontColor=#FFFFFF;fontSize={fs};fontFamily=Arial;"
        f"align=center;verticalAlign=middle;whiteSpace=wrap;overflow=visible;"
    )
    return dict(vertex=True, value=_esc(label), style=style,
                x=x, y=y, w=w, h=h, svg_id=gid)


# ---------------------------------------------------------------------------
# Apply transform offset to cell coordinates
# ---------------------------------------------------------------------------

def _apply_offset(cell, tx, ty):
    """Shift cell coordinates by (tx, ty) transform offset."""
    if tx == 0 and ty == 0:
        return cell
    if cell.get("vertex"):
        cell["x"] = cell["x"] + tx
        cell["y"] = cell["y"] + ty
    elif cell.get("edge"):
        sx, sy = cell["source_pt"]
        cell["source_pt"] = (sx + tx, sy + ty)
        tx2, ty2 = cell["target_pt"]
        cell["target_pt"] = (tx2 + tx, ty2 + ty)
        if cell.get("waypoints"):
            cell["waypoints"] = [(wx + tx, wy + ty) for wx, wy in cell["waypoints"]]
    return cell


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def svg_to_drawio(svg_path, output_path=None):
    """Convert SVG → draw.io XML.  Returns (output_path, error)."""
    if not os.path.exists(svg_path):
        return None, f"SVG not found: {svg_path}"

    if output_path is None:
        output_path = os.path.splitext(svg_path)[0] + ".drawio"

    try:
        tree = ET.parse(svg_path)
    except ET.ParseError as e:
        return None, f"SVG parse error: {e}"

    root = tree.getroot()

    # Page dimensions
    vb = root.get("viewBox")
    if vb:
        parts = vb.split()
        pw, ph = float(parts[2]), float(parts[3])
    else:
        pw = float(root.get("width", "1000"))
        ph = float(root.get("height", "1000"))

    # Parse CSS classes from <style> blocks
    css_map = _parse_css_classes(root)

    # Build transform offset map for every element
    offset_map = _build_transform_map(root)

    cells = []
    skip = set()     # ids of elements to skip (children of AF groups, defs, etc.)
    next_id = 2

    # Skip everything inside <defs>
    for defs in root.iter(f"{NS}defs"):
        for child in defs.iter():
            skip.add(id(child))

    # --- Pass 1: AF placeholder groups (preserve AF IDs for icon replacement) ---
    for g in root.iter(f"{NS}g"):
        gid = g.get("id", "")
        if not gid.startswith("AF"):
            continue
        tx, ty = offset_map.get(id(g), (0, 0))
        cell = _convert_af_group(g, css_map, tx, ty)
        if cell:
            cell["id"] = gid  # Keep original AF01/AF02 ID for icon replacement
            cells.append(cell)
        for child in g.iter():
            skip.add(id(child))

    # --- Pass 2: all other elements (with transform offsets) ---
    for elem in root.iter():
        if id(elem) in skip:
            continue
        tag = _strip(elem.tag)

        css = _resolve_classes(elem, css_map)
        tx, ty = offset_map.get(id(elem), (0, 0))

        cell = None
        if tag == "rect":
            cell = _convert_rect(elem, css, pw, ph)
        elif tag == "text":
            cell = _convert_text(elem, css)
        elif tag == "circle":
            cell = _convert_circle(elem, css)
        elif tag == "line":
            cell = _convert_line(elem, css)
        elif tag == "path":
            cell = _convert_path(elem, css)
        elif tag == "polyline":
            cell = _convert_polyline(elem, css)
        elif tag == "image":
            cell = _convert_image(elem, css)

        if cell:
            _apply_offset(cell, tx, ty)
            # Preserve meaningful SVG IDs (AF groups and icon images)
            svg_id = cell.get("svg_id", "")
            if svg_id and (svg_id.startswith("AF") or svg_id.startswith("icon_")):
                cell["id"] = svg_id
            else:
                cell["id"] = str(next_id)
            cells.append(cell)
            next_id += 1

    # --- Build XML ---
    lines = [
        f'<mxGraphModel dx="0" dy="0" grid="0" gridSize="1" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{int(pw)}" pageHeight="{int(ph)}" '
        f'math="0" shadow="0">',
        "  <root>",
        '    <mxCell id="0"/>',
        '    <mxCell id="1" parent="0"/>',
    ]

    for c in cells:
        cid = c["id"]
        val = c.get("value", "")
        sty = c["style"]

        if c.get("vertex"):
            lines.append(
                f'    <mxCell id="{cid}" value="{val}" '
                f'style="{sty}" vertex="1" parent="1">'
            )
            lines.append(
                f'      <mxGeometry x="{c["x"]:.1f}" y="{c["y"]:.1f}" '
                f'width="{c["w"]:.1f}" height="{c["h"]:.1f}" as="geometry"/>'
            )
            lines.append("    </mxCell>")

        elif c.get("edge"):
            sx, sy = c["source_pt"]
            tx, ty = c["target_pt"]
            wps = c.get("waypoints", [])

            lines.append(
                f'    <mxCell id="{cid}" value="" '
                f'style="{sty}" edge="1" parent="1">'
            )
            lines.append('      <mxGeometry relative="1" as="geometry">')
            lines.append(
                f'        <mxPoint x="{sx:.1f}" y="{sy:.1f}" as="sourcePoint"/>'
            )
            lines.append(
                f'        <mxPoint x="{tx:.1f}" y="{ty:.1f}" as="targetPoint"/>'
            )
            if wps:
                lines.append('        <Array as="points">')
                for wx, wy in wps:
                    lines.append(f'          <mxPoint x="{wx:.1f}" y="{wy:.1f}"/>')
                lines.append("        </Array>")
            lines.append("      </mxGeometry>")
            lines.append("    </mxCell>")

    lines += ["  </root>", "</mxGraphModel>"]

    model_xml = "\n".join(lines)
    mxfile = (
        '<mxfile host="app.diagrams.net" type="device">\n'
        f'  <diagram name="Page-1" id="page1">\n'
        f"    {model_xml}\n"
        f"  </diagram>\n"
        f"</mxfile>"
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mxfile)

    n_vert = sum(1 for c in cells if c.get("vertex"))
    n_edge = sum(1 for c in cells if c.get("edge"))
    print(f"Converted {len(cells)} elements ({n_vert} vertices, {n_edge} edges)")
    print(f"Page: {int(pw)}x{int(ph)}")
    print(f"Output: {output_path}")

    return output_path, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert SVG to draw.io XML")
    ap.add_argument("svg", help="Input SVG file")
    ap.add_argument("-o", "--output", help="Output .drawio file")
    args = ap.parse_args()

    out, err = svg_to_drawio(args.svg, args.output)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
    print("Done.")
