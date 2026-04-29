"""High-precision PNG to SVG conversion with geometric primitive fitting.

Pipeline:
  1. Color quantization (reduce to ~36 colors)
  2. vtracer initial tracing (raw cubic bezier paths)
  3. Post-processing: optimize each path's commands
     - Straight segments → L (line)
     - Circular/elliptical arcs → A (arc)
     - Quadratic curves → Q
     - Keep genuine cubics as C
  4. Output clean SVG

Returns (result, error) tuples per project convention.
"""

import os
import re
import sys
import math
import subprocess
import tempfile
from typing import List, Tuple, Optional, NamedTuple
from collections import defaultdict

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
T = type(None)  # placeholder
Point = Tuple[float, float]


class SvgPath(NamedTuple):
    fill: str          # hex color e.g. "#eaf2fa"
    fill_rule: str     # "evenodd" or "nonzero"
    commands: list      # list of (cmd_type, points) tuples


# ---------------------------------------------------------------------------
# 1. Color Quantization
# ---------------------------------------------------------------------------

def quantize_colors(image_path, n_colors=36, output_path=None):
    """Reduce image to n_colors using k-means clustering.

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32).reshape(-1, 3)

        # k-means clustering
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42,
                                batch_size=1000, n_init=3)
        kmeans.fit(arr)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        labels = kmeans.predict(arr)
        quantized = centers[labels].reshape(img.size[1], img.size[0], 3)

        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_q{n_colors}{ext}"

        Image.fromarray(quantized).save(output_path)
        return output_path, None
    except Exception as e:
        return None, f"Color quantization failed: {e}"


# ---------------------------------------------------------------------------
# 2. vtracer Initial Tracing
# ---------------------------------------------------------------------------

def run_vtracer(image_path, output_path=None, color_precision=6,
                filter_speckle=2, segment_length=3.5, path_precision=4,
                gradient_step=0, mode="spline"):
    """Run vtracer to get initial SVG with cubic bezier paths.

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = f"{base}_vtracer.svg"

    cmd = [
        "vtracer",
        "-i", image_path,
        "-o", output_path,
        "--colormode", "color",
        "--hierarchical", "stacked",
        "--mode", mode,
        "-p", str(color_precision),
        "-f", str(filter_speckle),
        "-l", str(segment_length),
        "--path_precision", str(path_precision),
    ]
    if gradient_step > 0:
        cmd.extend(["-g", str(gradient_step)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None, f"vtracer failed: {result.stderr}"
        return output_path, None
    except FileNotFoundError:
        return None, "vtracer not found. Install with: cargo install vtracer"
    except Exception as e:
        return None, f"vtracer error: {e}"


# ---------------------------------------------------------------------------
# 3. SVG Path Parsing
# ---------------------------------------------------------------------------

def parse_svg_paths(svg_content):
    """Parse SVG content into list of SvgPath objects.

    Returns:
        (list[SvgPath], None) on success, (None, error) on failure.
    """
    paths = []
    # Match <path ... /> elements
    path_re = re.compile(
        r'<path\s+([^>]*?)/>',
        re.DOTALL
    )
    for m in path_re.finditer(svg_content):
        attrs = m.group(1)

        # Extract fill color
        fill_m = re.search(r'fill="([^"]*)"', attrs)
        fill = fill_m.group(1) if fill_m else "#000000"

        # Normalize fill to hex
        fill = _normalize_color(fill)

        # Extract fill-rule
        rule_m = re.search(r'fill-rule="([^"]*)"', attrs)
        fill_rule = rule_m.group(1) if rule_m else "evenodd"

        # Extract d attribute
        d_m = re.search(r'd="([^"]*)"', attrs)
        if not d_m:
            continue
        d = d_m.group(1)

        # Extract transform
        transform_m = re.search(r'transform="([^"]*)"', attrs)
        tx, ty = 0.0, 0.0
        if transform_m:
            tr = transform_m.group(1)
            tr_m = re.search(r'translate\(([-\d.]+),\s*([-\d.]+)\)', tr)
            if tr_m:
                tx, ty = float(tr_m.group(1)), float(tr_m.group(2))

        # Parse path commands
        commands = _parse_path_d(d, tx, ty)
        if commands:
            paths.append(SvgPath(fill=fill, fill_rule=fill_rule, commands=commands))

    return paths, None


def _normalize_color(color_str):
    """Convert rgb(), named colors, etc. to #rrggbb hex."""
    if color_str.startswith("#"):
        # Already hex — normalize to 6-digit
        h = color_str.lstrip("#")
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        return f"#{h.lower()}"

    rgb_m = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
    if rgb_m:
        r, g, b = int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3))
        return f"#{r:02x}{g:02x}{b:02x}"

    # Fallback
    return color_str


def _parse_path_d(d, tx=0.0, ty=0.0):
    """Parse SVG path d attribute into command list.

    Each command: (type, [points...])
    - ('M', [(x, y)])
    - ('L', [(x, y)])
    - ('C', [(cp1x, cp1y), (cp2x, cp2y), (x, y)])
    - ('Q', [(cpx, cpy), (x, y)])
    - ('A', [(rx, ry, rotation, large_arc, sweep, x, y)])
    - ('Z', [])

    All coordinates are absolute and translated by (tx, ty).
    """
    commands = []
    # Tokenize: split into command letters and numbers
    tokens = re.findall(r'[MCLQAZmclqaz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d)

    i = 0
    current_cmd = None
    cx, cy = 0.0, 0.0  # current position (for relative commands)

    while i < len(tokens):
        token = tokens[i]

        if token in 'MCLQAZmclqaz':
            current_cmd = token
            i += 1
            continue

        if current_cmd is None:
            i += 1
            continue

        cmd_upper = current_cmd.upper()
        is_relative = current_cmd.islower()

        if cmd_upper == 'M':
            x, y = float(tokens[i]), float(tokens[i+1])
            i += 2
            if is_relative:
                x += cx; y += cy
            x += tx; y += ty
            commands.append(('M', [(x, y)]))
            cx, cy = x, y
            # Subsequent coords after M are treated as L
            current_cmd = 'l' if is_relative else 'L'

        elif cmd_upper == 'L':
            x, y = float(tokens[i]), float(tokens[i+1])
            i += 2
            if is_relative:
                x += cx; y += cy
            x += tx; y += ty
            commands.append(('L', [(x, y)]))
            cx, cy = x, y

        elif cmd_upper == 'C':
            cp1x, cp1y = float(tokens[i]), float(tokens[i+1])
            cp2x, cp2y = float(tokens[i+2]), float(tokens[i+3])
            x, y = float(tokens[i+4]), float(tokens[i+5])
            i += 6
            if is_relative:
                cp1x += cx; cp1y += cy
                cp2x += cx; cp2y += cy
                x += cx; y += cy
            cp1x += tx; cp1y += ty
            cp2x += tx; cp2y += ty
            x += tx; y += ty
            commands.append(('C', [(cp1x, cp1y), (cp2x, cp2y), (x, y)]))
            cx, cy = x, y

        elif cmd_upper == 'Q':
            cpx, cpy = float(tokens[i]), float(tokens[i+1])
            x, y = float(tokens[i+2]), float(tokens[i+3])
            i += 4
            if is_relative:
                cpx += cx; cpy += cy
                x += cx; y += cy
            cpx += tx; cpy += ty
            x += tx; y += ty
            commands.append(('Q', [(cpx, cpy), (x, y)]))
            cx, cy = x, y

        elif cmd_upper == 'A':
            rx = float(tokens[i])
            ry = float(tokens[i+1])
            rotation = float(tokens[i+2])
            large_arc = int(float(tokens[i+3]))
            sweep = int(float(tokens[i+4]))
            x, y = float(tokens[i+5]), float(tokens[i+6])
            i += 7
            if is_relative:
                x += cx; y += cy
            x += tx; y += ty
            commands.append(('A', [(rx, ry, rotation, large_arc, sweep, x, y)]))
            cx, cy = x, y

        elif cmd_upper == 'Z':
            commands.append(('Z', []))
            # Z doesn't consume coordinates
            # Reset to start of current sub-path
            for c in reversed(commands[:-1]):
                if c[0] == 'M':
                    cx, cy = c[1][0]
                    break

        else:
            i += 1

    return commands


# ---------------------------------------------------------------------------
# 4. Bezier Math Utilities
# ---------------------------------------------------------------------------

def cubic_bezier_point(p0, p1, p2, p3, t):
    """Evaluate cubic bezier at parameter t."""
    u = 1 - t
    return (
        u*u*u * p0[0] + 3*u*u*t * p1[0] + 3*u*t*t * p2[0] + t*t*t * p3[0],
        u*u*u * p0[1] + 3*u*u*t * p1[1] + 3*u*t*t * p2[1] + t*t*t * p3[1],
    )


def sample_cubic(p0, cp1, cp2, p3, n=20):
    """Sample n+1 points along a cubic bezier curve."""
    return [cubic_bezier_point(p0, cp1, cp2, p3, t/n) for t in range(n+1)]


def point_to_line_dist(px, py, x1, y1, x2, y2):
    """Distance from point (px,py) to line through (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy
    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1)
    return abs(dy * px - dx * py + x2*y1 - y2*x1) / math.sqrt(length_sq)


# ---------------------------------------------------------------------------
# 5. Geometric Primitive Detection
# ---------------------------------------------------------------------------

def is_line(p0, cp1, cp2, p3, tolerance=0.5):
    """Check if cubic bezier is effectively a straight line.

    Tests if both control points are within `tolerance` pixels of the
    line from p0 to p3.
    """
    d1 = point_to_line_dist(cp1[0], cp1[1], p0[0], p0[1], p3[0], p3[1])
    d2 = point_to_line_dist(cp2[0], cp2[1], p0[0], p0[1], p3[0], p3[1])
    return max(d1, d2) < tolerance


def is_quadratic(p0, cp1, cp2, p3, tolerance=0.3):
    """Check if cubic bezier can be reduced to a quadratic.

    A cubic B(t) with control points CP1, CP2 is equivalent to a quadratic
    with control point Q if: CP1 = P0 + 2/3*(Q-P0) and CP2 = P3 + 2/3*(Q-P3).
    Solving: Q = (3*CP1 - P0) / 2 = (3*CP2 - P3) / 2.
    If both solutions agree within tolerance, it's quadratic.
    """
    q1x = (3 * cp1[0] - p0[0]) / 2
    q1y = (3 * cp1[1] - p0[1]) / 2
    q2x = (3 * cp2[0] - p3[0]) / 2
    q2y = (3 * cp2[1] - p3[1]) / 2

    dist = math.hypot(q1x - q2x, q1y - q2y)
    if dist < tolerance:
        # Return the averaged control point
        qx = (q1x + q2x) / 2
        qy = (q1y + q2y) / 2
        return True, (qx, qy)
    return False, None


def fit_circle(points):
    """Fit a circle to a set of points using Taubin SVD method.

    Returns:
        (cx, cy, radius, residual) — center, radius, and mean residual.
    """
    if len(points) < 3:
        return None

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Center the data for numerical stability
    mx, my = xs.mean(), ys.mean()
    xs_c = xs - mx
    ys_c = ys - my

    # Taubin method
    z = xs_c**2 + ys_c**2
    zmean = z.mean()

    # Build moment matrix
    z_norm = z - zmean
    zx = np.column_stack([z_norm, xs_c, ys_c, np.ones(len(xs))])

    # SVD
    try:
        _, S, Vt = np.linalg.svd(zx, full_matrices=False)
        # Solution is the last row of Vt
        a = Vt[-1]
    except np.linalg.LinAlgError:
        return None

    # Extract circle parameters: a[0]*(x²+y²) + a[1]*x + a[2]*y + a[3] = 0
    # Center: (-a[1]/(2*a[0]), -a[2]/(2*a[0]))
    if abs(a[0]) < 1e-10:
        return None

    cx = -a[1] / (2 * a[0]) + mx
    cy = -a[2] / (2 * a[0]) + my
    r = math.sqrt(abs(a[1]**2 + a[2]**2 - 4*a[0]*a[3])) / (2 * abs(a[0]))

    if r < 0.1 or r > 10000:
        return None

    # Compute residual (mean absolute distance from circle)
    dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    residual = np.mean(np.abs(dists - r))

    return cx, cy, r, residual


def fit_ellipse(points):
    """Fit an ellipse to points using direct least squares (Halir-Flusser).

    Returns:
        (cx, cy, rx, ry, rotation_deg, residual) or None.
    """
    if len(points) < 6:
        return None

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Normalize for numerical stability
    mx, my = xs.mean(), ys.mean()
    sx = (xs.max() - xs.min()) / 2
    sy = (ys.max() - ys.min()) / 2
    if sx < 1e-6 or sy < 1e-6:
        return None
    xn = (xs - mx) / sx
    yn = (ys - my) / sy

    # Design matrix for conic: ax² + bxy + cy² + dx + ey + f = 0
    D1 = np.column_stack([xn**2, xn*yn, yn**2])
    D2 = np.column_stack([xn, yn, np.ones(len(xn))])

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Constraint matrix for ellipse: 4ac - b² > 0
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

    try:
        S3_inv = np.linalg.inv(S3)
        M = np.linalg.inv(C1) @ (S1 - S2 @ S3_inv @ S2.T)
        eigvals, eigvecs = np.linalg.eig(M)

        # Find eigenvector satisfying ellipse constraint
        cond = 4 * eigvecs[0] * eigvecs[2] - eigvecs[1]**2
        valid = cond > 0
        if not np.any(valid):
            return None

        # Pick the one with positive constraint
        idx = np.where(valid)[0]
        # Choose eigenvector with smallest positive eigenvalue
        best = idx[np.argmin(np.abs(eigvals[idx]))]
        a1 = eigvecs[:, best]
        a2 = -S3_inv @ S2.T @ a1

        # Denormalize coefficients
        # [a, b, c, d, e, f] in original coordinates
        a_coeff = np.array([
            a1[0] / (sx*sx),
            a1[1] / (sx*sy),
            a1[2] / (sy*sy),
            -2*a1[0]*mx/(sx*sx) - a1[1]*my/(sx*sy) + a2[0]/sx,
            -a1[1]*mx/(sx*sy) - 2*a1[2]*my/(sy*sy) + a2[1]/sy,
            a1[0]*mx*mx/(sx*sx) + a1[1]*mx*my/(sx*sy) + a1[2]*my*my/(sy*sy)
            - a2[0]*mx/sx - a2[1]*my/sy + a2[2],
        ])

        a, b, c, d, e, f = a_coeff

        # Extract ellipse parameters from conic coefficients
        det = b*b - 4*a*c
        if det >= 0:
            return None  # Not an ellipse

        cx = (2*c*d - b*e) / det
        cy = (2*a*e - b*d) / det

        # Semi-axes and rotation
        num = 2 * (a*e*e + c*d*d - b*d*e + det*f)
        s1 = a + c
        s2_sq = (a-c)**2 + b*b
        if s2_sq < 0:
            return None
        s2 = math.sqrt(float(s2_sq))

        if abs(det) < 1e-12:
            return None

        rx_sq = float(-num * (s1 + s2) / (det * det))
        ry_sq = float(-num * (s1 - s2) / (det * det))

        if rx_sq <= 0 or ry_sq <= 0:
            return None

        rx = math.sqrt(rx_sq)
        ry = math.sqrt(ry_sq)

        # Rotation angle
        a, b, c = float(a), float(b), float(c)
        if abs(b) < 1e-10 and a < c:
            rotation = 0.0
        elif abs(b) < 1e-10:
            rotation = math.pi / 2
        else:
            rotation = math.atan2(c - a - s2, b) / 2

        rotation_deg = math.degrees(rotation)

        # Compute residual
        dists = []
        cos_r, sin_r = math.cos(-rotation), math.sin(-rotation)
        for x, y in zip(xs, ys):
            dx, dy = x - float(cx), y - float(cy)
            xr = dx * cos_r - dy * sin_r
            yr = dx * sin_r + dy * cos_r
            angle = math.atan2(yr / ry, xr / rx) if (rx > 0 and ry > 0) else 0
            ex = rx * math.cos(angle)
            ey = ry * math.sin(angle)
            dists.append(math.hypot(xr - ex, yr - ey))

        residual = float(np.mean(dists))
        return float(cx), float(cy), rx, ry, rotation_deg, residual

    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None


def try_arc_fit(p0, cp1, cp2, p3, tolerance=0.5):
    """Try to fit a cubic bezier as a circular or elliptical arc.

    Returns:
        ('A', params) if arc fits within tolerance, or None.
        params = (rx, ry, rotation, large_arc, sweep, end_x, end_y)
    """
    # Sample points along the cubic bezier
    points = sample_cubic(p0, cp1, cp2, p3, n=16)

    # Skip very short segments
    chord = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
    if chord < 1.0:
        return None

    # Try circle fit first (simpler, more common)
    circle = fit_circle(points)
    if circle is not None:
        cx, cy, r, residual = circle
        if residual < tolerance:
            # Convert to SVG arc parameters
            arc_params = _circle_to_arc(p0, p3, cx, cy, r)
            if arc_params is not None:
                return arc_params

    # Try ellipse fit
    ellipse = fit_ellipse(points)
    if ellipse is not None:
        ecx, ecy, rx, ry, rot, residual = ellipse
        if residual < tolerance and rx > 0.1 and ry > 0.1:
            arc_params = _ellipse_to_arc(p0, p3, ecx, ecy, rx, ry, rot)
            if arc_params is not None:
                return arc_params

    return None


def _circle_to_arc(p0, p3, cx, cy, r):
    """Convert circle fit to SVG arc parameters."""
    # Calculate angles
    angle0 = math.atan2(p0[1] - cy, p0[0] - cx)
    angle3 = math.atan2(p3[1] - cy, p3[0] - cx)

    # Determine sweep direction (clockwise vs counter-clockwise)
    # Use cross product of the vectors
    d_angle = angle3 - angle0
    # Normalize to [-pi, pi]
    while d_angle > math.pi:
        d_angle -= 2 * math.pi
    while d_angle < -math.pi:
        d_angle += 2 * math.pi

    sweep = 1 if d_angle > 0 else 0
    large_arc = 1 if abs(d_angle) > math.pi else 0

    return (r, r, 0, large_arc, sweep, p3[0], p3[1])


def _ellipse_to_arc(p0, p3, cx, cy, rx, ry, rotation_deg):
    """Convert ellipse fit to SVG arc parameters."""
    rot_rad = math.radians(rotation_deg)
    cos_r = math.cos(-rot_rad)
    sin_r = math.sin(-rot_rad)

    # Transform to ellipse-local coordinates
    dx0, dy0 = p0[0] - cx, p0[1] - cy
    lx0 = dx0 * cos_r - dy0 * sin_r
    ly0 = dx0 * sin_r + dy0 * cos_r

    dx3, dy3 = p3[0] - cx, p3[1] - cy
    lx3 = dx3 * cos_r - dy3 * sin_r
    ly3 = dx3 * sin_r + dy3 * cos_r

    # Angles on the ellipse
    angle0 = math.atan2(ly0 / ry, lx0 / rx) if rx > 0 and ry > 0 else 0
    angle3 = math.atan2(ly3 / ry, lx3 / rx) if rx > 0 and ry > 0 else 0

    d_angle = angle3 - angle0
    while d_angle > math.pi:
        d_angle -= 2 * math.pi
    while d_angle < -math.pi:
        d_angle += 2 * math.pi

    sweep = 1 if d_angle > 0 else 0
    large_arc = 1 if abs(d_angle) > math.pi else 0

    return (rx, ry, rotation_deg, large_arc, sweep, p3[0], p3[1])


# ---------------------------------------------------------------------------
# 6. Path Optimization
# ---------------------------------------------------------------------------

def optimize_path(path, line_tol=0.5, arc_tol=0.4, quad_tol=0.3):
    """Optimize a single SvgPath by fitting geometric primitives.

    Returns:
        (optimized_SvgPath, None).
    """
    optimized_cmds = []
    current_pos = (0.0, 0.0)

    for cmd_type, pts in path.commands:
        if cmd_type == 'M':
            optimized_cmds.append(('M', pts))
            current_pos = pts[0]

        elif cmd_type == 'Z':
            optimized_cmds.append(('Z', []))

        elif cmd_type == 'L':
            optimized_cmds.append(('L', pts))
            current_pos = pts[0]

        elif cmd_type == 'Q':
            optimized_cmds.append(('Q', pts))
            current_pos = pts[-1]

        elif cmd_type == 'A':
            optimized_cmds.append(('A', pts))
            current_pos = (pts[0][-2], pts[0][-1]) if pts else current_pos

        elif cmd_type == 'C':
            p0 = current_pos
            cp1, cp2, p3 = pts[0], pts[1], pts[2]

            # Try line first (cheapest)
            if is_line(p0, cp1, cp2, p3, tolerance=line_tol):
                optimized_cmds.append(('L', [p3]))
                current_pos = p3
                continue

            # Try quadratic reduction
            is_quad, q_cp = is_quadratic(p0, cp1, cp2, p3, tolerance=quad_tol)
            if is_quad:
                optimized_cmds.append(('Q', [q_cp, p3]))
                current_pos = p3
                continue

            # Try arc fitting
            arc = try_arc_fit(p0, cp1, cp2, p3, tolerance=arc_tol)
            if arc is not None:
                optimized_cmds.append(('A', [arc]))
                current_pos = p3
                continue

            # Keep as cubic
            optimized_cmds.append(('C', pts))
            current_pos = p3

        else:
            optimized_cmds.append((cmd_type, pts))

    return SvgPath(fill=path.fill, fill_rule=path.fill_rule,
                   commands=optimized_cmds), None


def optimize_all_paths(paths, line_tol=0.5, arc_tol=0.4, quad_tol=0.3):
    """Optimize all paths in the SVG.

    Returns:
        (list[SvgPath], None).
    """
    optimized = []
    stats = defaultdict(int)

    for path in paths:
        opt_path, _ = optimize_path(path, line_tol, arc_tol, quad_tol)
        optimized.append(opt_path)

        for cmd_type, _ in opt_path.commands:
            stats[cmd_type] += 1

    total = sum(stats.values())
    print(f"  Optimization stats ({total} total commands):")
    for cmd in ['M', 'L', 'A', 'Q', 'C', 'Z']:
        if stats[cmd] > 0:
            pct = stats[cmd] / total * 100
            print(f"    {cmd}: {stats[cmd]:6d} ({pct:.1f}%)")

    return optimized, None


# ---------------------------------------------------------------------------
# 7. Color Merging
# ---------------------------------------------------------------------------

def merge_similar_colors(paths, threshold=8):
    """Merge paths with very similar colors (within threshold in RGB space).

    Returns:
        (list[SvgPath], None).
    """
    # Collect unique colors
    colors = list(set(p.fill for p in paths))

    # Build merge map
    merge_map = {}
    for c in colors:
        if c in merge_map:
            continue
        r1, g1, b1 = _hex_to_rgb(c)
        for c2 in colors:
            if c2 == c or c2 in merge_map:
                continue
            r2, g2, b2 = _hex_to_rgb(c2)
            dist = math.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
            if dist < threshold:
                merge_map[c2] = c

    if merge_map:
        merged = []
        for p in paths:
            new_fill = merge_map.get(p.fill, p.fill)
            merged.append(SvgPath(fill=new_fill, fill_rule=p.fill_rule,
                                  commands=p.commands))
        n_merged = len(set(merge_map.values()))
        print(f"  Merged {len(merge_map)} colors into {n_merged} groups")
        return merged, None

    return paths, None


def _hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# 8. SVG Output
# ---------------------------------------------------------------------------

def paths_to_svg(paths, width, height, precision=4):
    """Convert optimized paths back to SVG string.

    Returns:
        (svg_string, None).
    """
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
    ]

    for path in paths:
        d = _commands_to_d(path.commands, precision)
        if not d:
            continue
        lines.append(
            f'  <path fill="{path.fill}" fill-rule="{path.fill_rule}" '
            f'd="{d}"/>'
        )

    lines.append('</svg>')
    return '\n'.join(lines), None


def _commands_to_d(commands, precision=4):
    """Convert command list to SVG d attribute string."""
    parts = []
    fmt = f"{{:.{precision}f}}"

    for cmd_type, pts in commands:
        if cmd_type == 'M':
            x, y = pts[0]
            parts.append(f"M {fmt.format(x)} {fmt.format(y)}")

        elif cmd_type == 'L':
            x, y = pts[0]
            parts.append(f"L {fmt.format(x)} {fmt.format(y)}")

        elif cmd_type == 'C':
            cp1, cp2, end = pts
            parts.append(
                f"C {fmt.format(cp1[0])} {fmt.format(cp1[1])} "
                f"{fmt.format(cp2[0])} {fmt.format(cp2[1])} "
                f"{fmt.format(end[0])} {fmt.format(end[1])}"
            )

        elif cmd_type == 'Q':
            cp, end = pts[0], pts[1]
            parts.append(
                f"Q {fmt.format(cp[0])} {fmt.format(cp[1])} "
                f"{fmt.format(end[0])} {fmt.format(end[1])}"
            )

        elif cmd_type == 'A':
            params = pts[0]
            rx, ry, rot, large, sweep, x, y = params
            parts.append(
                f"A {fmt.format(rx)} {fmt.format(ry)} "
                f"{fmt.format(rot)} {int(large)} {int(sweep)} "
                f"{fmt.format(x)} {fmt.format(y)}"
            )

        elif cmd_type == 'Z':
            parts.append("Z")

    return " ".join(parts)


def _path_bbox(path):
    """Compute bounding box (x, y, w, h) of a path from its commands."""
    xs, ys = [], []
    for cmd, pts in path.commands:
        if cmd in ('M', 'L'):
            xs.append(pts[0][0]); ys.append(pts[0][1])
        elif cmd == 'C':
            for p in pts:
                xs.append(p[0]); ys.append(p[1])
        elif cmd == 'Q':
            for p in pts:
                xs.append(p[0]); ys.append(p[1])
        elif cmd == 'A':
            rx, ry, rot, la, sw, x, y = pts[0]
            xs.append(x); ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _color_luminance(hex_color):
    """Compute relative luminance of a hex color string like '#rrggbb'."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return 1.0
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def _remove_text_paths(paths, word_bboxes, margin=3):
    """Remove traced paths whose anchor points fall inside text word bboxes.

    Uses the actual M (move-to) and L (line-to) endpoint coordinates rather
    than the full control-point bounding box, since bezier control points can
    be far from the visible curve.

    Removes paths where the majority of anchor points lie inside word regions,
    unless the path spans a very large area (likely a background fill).

    Args:
        paths: list of SvgPath
        word_bboxes: list of (x1, y1, x2, y2) tuples
        margin: pixel margin around word bboxes

    Returns:
        (filtered_paths, n_removed)
    """
    if not word_bboxes:
        return paths, 0

    # Build list of expanded word regions
    regions = []
    for (x1, y1, x2, y2) in word_bboxes:
        regions.append((x1 - margin, y1 - margin, x2 + margin, y2 + margin))

    def _point_in_any_region(x, y):
        for rx1, ry1, rx2, ry2 in regions:
            if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                return True
        return False

    kept = []
    removed = 0
    for p in paths:
        # Get the first move-to point (path start position)
        start_pt = None
        anchors = []
        for cmd, pts in p.commands:
            if cmd == 'M':
                if start_pt is None:
                    start_pt = pts[0]
                anchors.append(pts[0])
            elif cmd == 'L':
                anchors.append(pts[0])
            elif cmd == 'C':
                anchors.append(pts[2])
            elif cmd == 'Q':
                anchors.append(pts[1])
            elif cmd == 'A':
                anchors.append((pts[0][5], pts[0][6]))

        if start_pt is None or not anchors:
            kept.append(p)
            continue

        # Path must start inside a word region
        if not _point_in_any_region(start_pt[0], start_pt[1]):
            kept.append(p)
            continue

        # Path must be dark (luminance < 0.70 — catches black, gray, colored text)
        if _color_luminance(p.fill) >= 0.70:
            kept.append(p)
            continue

        # Path must be small (text glyph sized, not a large graphic element)
        xs = [x for x, y in anchors]
        ys = [y for x, y in anchors]
        span_w = max(xs) - min(xs) if xs else 0
        span_h = max(ys) - min(ys) if ys else 0
        if span_w > 40 or span_h > 30:
            kept.append(p)
            continue

        removed += 1

    return kept, removed


# ---------------------------------------------------------------------------
# 9. Text Detection (for semantic mode)
# ---------------------------------------------------------------------------

def detect_text_regions(image_path):
    """Detect text using pytesseract with bounding boxes and styling.

    Returns line-level items, word-level bboxes, and per-word details.

    Returns:
        ({lines: [...], word_bboxes: [...], words: [...]}, None) on success.
        Each line: {text, x, y, w, h, font_size, text_color, bg_color, conf}
        Each word_bbox: (x, y, x2, y2) tight bounding box
        Each word: {text, x, y, w, h, font_size, text_color, bg_color, conf}
    """
    try:
        import pytesseract
    except ImportError:
        return None, "pytesseract not installed (pip install pytesseract)"

    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)
        ih, iw = arr.shape[:2]

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Collect words grouped into lines, and individual word bboxes
        from collections import defaultdict as _dd
        line_groups = _dd(list)
        word_bboxes = []
        all_words = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if not text or conf < 55:
                continue
            x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])

            # Add word-level bbox (tight around each word)
            word_bboxes.append((x, y, x + bw, y + bh))

            # Sample text color (dark pixels in bounding box)
            region = arr[max(0,y):min(ih,y+bh), max(0,x):min(iw,x+bw)]
            tc, bg = (0,0,0), (255,255,255)
            if region.size > 0:
                gray = np.mean(region, axis=2)
                dark_mask = gray < np.percentile(gray, 25)
                if dark_mask.any():
                    tc = tuple(int(v) for v in np.median(region[dark_mask], axis=0))
                light_mask = gray > np.percentile(gray, 75)
                if light_mask.any():
                    bg = tuple(int(v) for v in np.median(region[light_mask], axis=0))

            word_info = {
                'text': text, 'conf': conf,
                'x': x, 'y': y, 'w': bw, 'h': bh,
                'font_size': bh,
                'text_color': tc, 'bg_color': bg,
            }
            line_groups[key].append(word_info)
            all_words.append(word_info)

        # Build line-level items
        lines = []
        for key in sorted(line_groups.keys()):
            words = sorted(line_groups[key], key=lambda w: w['x'])
            full_text = ' '.join(w['text'] for w in words)
            x0 = words[0]['x']
            y0 = min(w['y'] for w in words)
            x1 = max(w['x'] + w['w'] for w in words)
            y1 = max(w['y'] + w['h'] for w in words)
            tc = tuple(int(np.mean([w['text_color'][c] for w in words])) for c in range(3))
            bg = tuple(int(np.mean([w['bg_color'][c] for w in words])) for c in range(3))
            avg_conf = np.mean([w['conf'] for w in words])

            lines.append({
                'text': full_text,
                'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0,
                'font_size': y1 - y0,
                'text_color': tc, 'bg_color': bg,
                'conf': avg_conf,
                'words': words,  # keep per-word details in line
            })

        return {'lines': lines, 'word_bboxes': word_bboxes, 'words': all_words}, None
    except Exception as e:
        return None, f"Text detection failed: {e}"


def _text_to_svg_elements(text_lines, with_background=True):
    """Convert detected text lines to SVG elements with optional background rects.

    When with_background=True, adds a background rect behind each text element
    to cleanly cover the traced text underneath.
    """
    elements = []
    pad = 1  # padding around text background rect
    for line in text_lines:
        tc = line['text_color']
        bg = line['bg_color']
        fill = f"#{tc[0]:02x}{tc[1]:02x}{tc[2]:02x}"
        bg_fill = f"#{bg[0]:02x}{bg[1]:02x}{bg[2]:02x}"
        baseline_y = line['y'] + line['h'] * 0.82
        fs = max(6, line['font_size'] * 0.85)
        text = (line['text']
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

        if with_background:
            elements.append(
                f'  <rect x="{line["x"]-pad}" y="{line["y"]-pad}" '
                f'width="{line["w"]+2*pad}" height="{line["h"]+2*pad}" '
                f'fill="{bg_fill}"/>'
            )
        elements.append(
            f'  <text x="{line["x"]}" y="{baseline_y:.1f}" '
            f'font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{fs:.1f}" fill="{fill}">{text}</text>'
        )
    return elements


# ---------------------------------------------------------------------------
# 10. CV-based Semantic Element Detection
# ---------------------------------------------------------------------------

def detect_rectangles_cv(image_path, min_area=500, min_side=20, n_colors=32):
    """Detect rectangles via color-based region segmentation.

    Quantizes image to n_colors, then finds connected components of each color
    that form rectangular shapes (fill_ratio > 0.85 of bounding rect).
    More robust than edge-based detection for filled colored boxes.

    Returns:
        (list_of_rects, None) on success, (None, error) on failure.
        Each rect: {x, y, w, h, fill_color: (r,g,b), stroke_color: (r,g,b)|None,
                    rx: float (corner radius)}
    """
    try:
        import cv2
    except ImportError:
        return None, "opencv-python not installed (pip install opencv-python)"

    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Cannot read image: {image_path}"
        ih, iw = img.shape[:2]

        # Quantize image to fewer colors for region detection
        Z = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, n_colors, None, criteria, 3,
                                         cv2.KMEANS_PP_CENTERS)
        centers_u8 = np.uint8(centers)
        q_labels = labels.reshape(ih, iw)

        rects = []
        seen_regions = []

        for ci in range(n_colors):
            mask = (q_labels == ci).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                if rect_area == 0 or w < min_side or h < min_side:
                    continue
                # Check rectangularity
                fill_ratio = area / rect_area
                if fill_ratio < 0.85:
                    continue
                # Skip near-full-image rectangles (background)
                if w > iw * 0.9 and h > ih * 0.9:
                    continue

                # Deduplicate
                cx, cy = x + w // 2, y + h // 2
                is_dup = False
                for sx, sy, sw, sh in seen_regions:
                    if (abs(cx - sx) < max(w, sw) * 0.3 and
                        abs(cy - sy) < max(h, sh) * 0.3):
                        is_dup = True
                        break
                if is_dup:
                    continue
                seen_regions.append((cx, cy, w, h))

                # Fill color from quantized center
                color_bgr = centers_u8[ci]
                fill_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

                # Check if there's a visible border
                # Sample 1px inside boundary from original image
                stroke_rgb = None
                border_pixels = []
                for bx in range(x, min(x + w, iw), 2):
                    if 0 <= y < ih:
                        border_pixels.append(img[y, bx])
                    if 0 <= y + h - 1 < ih:
                        border_pixels.append(img[min(y + h - 1, ih - 1), bx])
                for by in range(y, min(y + h, ih), 2):
                    if 0 <= x < iw:
                        border_pixels.append(img[by, x])
                    if 0 <= x + w - 1 < iw:
                        border_pixels.append(img[by, min(x + w - 1, iw - 1)])
                if border_pixels:
                    border_arr = np.array(border_pixels)
                    stroke_bgr = np.median(border_arr, axis=0).astype(int)
                    stroke_rgb = (int(stroke_bgr[2]), int(stroke_bgr[1]),
                                  int(stroke_bgr[0]))
                    if all(abs(int(stroke_rgb[c]) - int(fill_rgb[c])) < 25
                           for c in range(3)):
                        stroke_rgb = None

                # Corner radius from fill ratio deviation
                rx = 0.0
                if fill_ratio < 0.95:
                    rx = min(w, h) * (1 - fill_ratio) * 2

                rects.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'fill_color': fill_rgb,
                    'stroke_color': stroke_rgb,
                    'rx': round(rx, 1),
                })

        # Sort by area (largest first) for proper z-ordering
        rects.sort(key=lambda r: r['w'] * r['h'], reverse=True)
        return rects, None

    except Exception as e:
        return None, f"Rectangle detection failed: {e}"


def detect_lines_cv(image_path, rects=None, min_length=40):
    """Detect connecting lines/arrows using Hough Line Transform.

    Filters to find lines that are NOT part of rectangle borders.
    Optionally uses detected rectangles to filter out border segments.

    Returns:
        (list_of_lines, None) on success, (None, error) on failure.
        Each line: {x1, y1, x2, y2, color: (r,g,b), stroke_width: float,
                    has_arrow: bool}
    """
    try:
        import cv2
    except ImportError:
        return None, "opencv-python not installed (pip install opencv-python)"

    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, f"Cannot read image: {image_path}"
        ih, iw = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # If we have rects, mask out their border regions to avoid detecting box edges
        if rects:
            mask = np.ones_like(edges) * 255
            for r in rects:
                x, y, w, h = r['x'], r['y'], r['w'], r['h']
                border = 4  # mask border region
                # Mask top/bottom edges
                cv2.rectangle(mask, (x - border, y - border),
                              (x + w + border, y + border), 0, -1)
                cv2.rectangle(mask, (x - border, y + h - border),
                              (x + w + border, y + h + border), 0, -1)
                # Mask left/right edges
                cv2.rectangle(mask, (x - border, y - border),
                              (x + border, y + h + border), 0, -1)
                cv2.rectangle(mask, (x + w - border, y - border),
                              (x + w + border, y + h + border), 0, -1)
            edges = cv2.bitwise_and(edges, mask.astype(np.uint8))

        # Probabilistic Hough Line Transform (high threshold for clean results)
        hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                       threshold=80, minLineLength=min_length,
                                       maxLineGap=5)

        if hough_lines is None:
            return [], None

        lines = []
        seen_lines = []

        for hl in hough_lines:
            x1, y1, x2, y2 = hl[0]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < min_length:
                continue

            # Check if this line is mostly horizontal or vertical
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            is_hv = (dx < 3 or dy < 3 or
                     (dx > 0 and dy / dx < 0.1) or
                     (dy > 0 and dx / dy < 0.1))

            # For academic figures, most connecting lines are horizontal or vertical
            # Allow diagonal lines but prefer H/V
            angle = math.atan2(abs(y2 - y1), abs(x2 - x1))

            # Check if line is inside a rectangle (skip these)
            inside_rect = False
            if rects:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                for r in rects:
                    rx, ry, rw, rh = r['x'], r['y'], r['w'], r['h']
                    margin = 5
                    if (rx + margin < mx < rx + rw - margin and
                        ry + margin < my < ry + rh - margin):
                        # Check if both endpoints are inside
                        if (rx + margin < x1 < rx + rw - margin and
                            ry + margin < y1 < ry + rh - margin and
                            rx + margin < x2 < rx + rw - margin and
                            ry + margin < y2 < ry + rh - margin):
                            inside_rect = True
                            break
            if inside_rect:
                continue

            # Deduplicate similar lines
            is_dup = False
            for sl in seen_lines:
                dist1 = math.hypot(x1 - sl[0], y1 - sl[1])
                dist2 = math.hypot(x2 - sl[2], y2 - sl[3])
                dist3 = math.hypot(x1 - sl[2], y1 - sl[3])
                dist4 = math.hypot(x2 - sl[0], y2 - sl[1])
                if min(dist1 + dist2, dist3 + dist4) < 15:
                    is_dup = True
                    break
            if is_dup:
                continue
            seen_lines.append((x1, y1, x2, y2))

            # Sample line color
            n_samples = max(3, int(length / 5))
            colors = []
            for t in range(n_samples):
                frac = t / max(1, n_samples - 1)
                sx = int(x1 + frac * (x2 - x1))
                sy = int(y1 + frac * (y2 - y1))
                sx = max(0, min(iw - 1, sx))
                sy = max(0, min(ih - 1, sy))
                colors.append(img[sy, sx])
            if colors:
                avg_bgr = np.median(colors, axis=0).astype(int)
                color_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
            else:
                color_rgb = (0, 0, 0)

            # Estimate stroke width by checking perpendicular extent of dark pixels
            stroke_width = 1.5
            if is_hv:
                if dx < 3:  # vertical line
                    col = max(0, min(iw - 1, (x1 + x2) // 2))
                    row_start = min(y1, y2)
                    row_end = max(y1, y2)
                    mid_row = (row_start + row_end) // 2
                    mid_row = max(0, min(ih - 1, mid_row))
                    # Scan left and right from the line
                    w_count = 0
                    for d in range(-10, 11):
                        c = col + d
                        if 0 <= c < iw and gray[mid_row, c] < 180:
                            w_count += 1
                    stroke_width = max(1, w_count * 0.8)
                elif dy < 3:  # horizontal line
                    row = max(0, min(ih - 1, (y1 + y2) // 2))
                    col_start = min(x1, x2)
                    col_end = max(x1, x2)
                    mid_col = (col_start + col_end) // 2
                    mid_col = max(0, min(iw - 1, mid_col))
                    h_count = 0
                    for d in range(-10, 11):
                        r = row + d
                        if 0 <= r < ih and gray[r, mid_col] < 180:
                            h_count += 1
                    stroke_width = max(1, h_count * 0.8)

            # Simple arrow detection: check if endpoints have triangular regions
            has_arrow = _detect_arrowhead(img, gray, x1, y1, x2, y2)

            lines.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'color': color_rgb,
                'stroke_width': round(min(stroke_width, 5), 1),
                'has_arrow': has_arrow,
            })

        return lines, None

    except Exception as e:
        return None, f"Line detection failed: {e}"


def _detect_arrowhead(img, gray, x1, y1, x2, y2):
    """Check if line endpoint has an arrowhead shape."""
    ih, iw = gray.shape
    # Check at the end point (x2, y2)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 5:
        return False

    # Normalize direction
    ndx, ndy = dx / length, dy / length

    # Sample a small region around the endpoint
    region_size = 8
    ex, ey = int(x2), int(y2)
    y_lo = max(0, ey - region_size)
    y_hi = min(ih, ey + region_size)
    x_lo = max(0, ex - region_size)
    x_hi = min(iw, ex + region_size)

    if y_hi <= y_lo or x_hi <= x_lo:
        return False

    region = gray[y_lo:y_hi, x_lo:x_hi]
    dark_pixels = np.sum(region < 128)
    total_pixels = region.size

    # Arrowheads typically have a higher dark pixel density than regular line segments
    if total_pixels > 0 and dark_pixels / total_pixels > 0.25:
        return True
    return False


def _path_bbox(path):
    """Compute bounding box of a path from its command points."""
    xs, ys = [], []
    for cmd, pts in path.commands:
        if cmd == 'Z':
            continue
        if cmd == 'A':
            for p in pts:
                xs.append(p[-2])
                ys.append(p[-1])
        else:
            for p in pts:
                xs.append(p[0])
                ys.append(p[1])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_overlap(b1, b2, threshold=0.5):
    """Check if two bounding boxes overlap by at least threshold fraction."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if x2 <= x1 or y2 <= y1:
        return False

    inter = (x2 - x1) * (y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])

    if area1 <= 0:
        return False

    return inter / area1 >= threshold


def filter_paths_by_regions(paths, regions, overlap_threshold=0.6):
    """Remove traced paths whose bounding boxes overlap significantly with
    detected semantic regions (rects, text areas).

    Args:
        paths: list of SvgPath
        regions: list of (x, y, x2, y2) bounding boxes for semantic elements
        overlap_threshold: fraction of path bbox that must overlap to be removed

    Returns:
        (kept_paths, removed_count)
    """
    if not regions:
        return paths, 0

    kept = []
    removed = 0
    for p in paths:
        bbox = _path_bbox(p)
        if bbox is None:
            kept.append(p)
            continue

        should_remove = False
        for reg in regions:
            if _bbox_overlap(bbox, reg, overlap_threshold):
                should_remove = True
                break

        if should_remove:
            removed += 1
        else:
            kept.append(p)

    return kept, removed


def _rects_to_svg_elements(rects):
    """Convert detected rectangles to SVG <rect> elements."""
    elements = []
    for r in rects:
        fill = f"#{r['fill_color'][0]:02x}{r['fill_color'][1]:02x}{r['fill_color'][2]:02x}"
        stroke = "none"
        stroke_w = ""
        if r['stroke_color']:
            stroke = f"#{r['stroke_color'][0]:02x}{r['stroke_color'][1]:02x}{r['stroke_color'][2]:02x}"
            stroke_w = ' stroke-width="1"'
        rx_attr = f' rx="{r["rx"]}"' if r['rx'] > 0 else ''
        elements.append(
            f'  <rect x="{r["x"]}" y="{r["y"]}" width="{r["w"]}" height="{r["h"]}" '
            f'fill="{fill}" stroke="{stroke}"{stroke_w}{rx_attr}/>'
        )
    return elements


def _lines_to_svg_elements(lines, defs_needed):
    """Convert detected lines to SVG <line> or <polyline> elements.

    Args:
        lines: detected line dicts
        defs_needed: set — adds marker IDs if arrows are detected

    Returns list of SVG element strings.
    """
    elements = []
    for i, ln in enumerate(lines):
        color = f"#{ln['color'][0]:02x}{ln['color'][1]:02x}{ln['color'][2]:02x}"
        sw = ln['stroke_width']
        marker = ""
        if ln['has_arrow']:
            marker_id = f"arrow_{i}"
            defs_needed.add((marker_id, color))
            marker = f' marker-end="url(#{marker_id})"'
        elements.append(
            f'  <line x1="{ln["x1"]}" y1="{ln["y1"]}" x2="{ln["x2"]}" y2="{ln["y2"]}" '
            f'stroke="{color}" stroke-width="{sw}"{marker}/>'
        )
    return elements


def _arrow_defs_svg(defs_set):
    """Generate SVG <defs> for arrow markers."""
    if not defs_set:
        return ""
    parts = ['  <defs>']
    for marker_id, color in defs_set:
        parts.append(
            f'    <marker id="{marker_id}" markerWidth="10" markerHeight="7" '
            f'refX="10" refY="3.5" orient="auto">'
        )
        parts.append(
            f'      <polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>'
        )
        parts.append('    </marker>')
    parts.append('  </defs>')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# 11. Full Semantic Pipeline
# ---------------------------------------------------------------------------

QUALITY_PRESETS = {
    'fast':     {'color_precision': 6, 'upscale': 1.0, 'speckle': 1, 'min_dlen': 60, 'precision': 1},
    'balanced': {'color_precision': 8, 'upscale': 1.0, 'speckle': 0, 'min_dlen': 60, 'precision': 1},
    'high':     {'color_precision': 8, 'upscale': 1.5, 'speckle': 0, 'min_dlen': 60, 'precision': 1},
    'max':      {'color_precision': 8, 'upscale': 2.0, 'speckle': 0, 'min_dlen': 60, 'precision': 1},
}


def _paint_text_regions(image_path, text_lines, max_bg_std=12.0):
    """Blur text regions to remove letter shapes while preserving background.

    Uses heavy Gaussian blur on text regions so vtracer doesn't trace
    individual letters. The actual text is rendered via SVG <text> elements.
    Blur preserves background color distribution better than flat-fill.

    Returns:
        (painted_image_path, painted_line_indices, None) on success.
    """
    try:
        import cv2
    except ImportError:
        return None, set(), "opencv-python required"

    img = cv2.imread(image_path)
    if img is None:
        return None, set(), f"Cannot read: {image_path}"
    ih, iw = img.shape[:2]

    painted_lines = set()
    for li, tl in enumerate(text_lines):
        x, y, w, h = tl['x'], tl['y'], tl['w'], tl['h']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 <= x1 or y2 <= y1:
            continue

        region = img[y1:y2, x1:x2]
        if region.size == 0:
            continue

        gray = np.mean(region, axis=2)
        light = gray > np.percentile(gray, 60)
        if light.any():
            bg_std = float(np.std(region[light].reshape(-1, 3), axis=0).mean())
        else:
            bg_std = float(np.std(region.reshape(-1, 3), axis=0).mean())

        if bg_std <= max_bg_std and tl.get('conf', 100) > 70:
            painted_lines.add(li)
            # Heavy blur to remove letter shapes
            ksize = max(5, h) | 1  # odd kernel size, at least text height
            blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
            img[y1:y2, x1:x2] = blurred

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        out_path = tmp.name
    cv2.imwrite(out_path, img)
    return out_path, painted_lines, None


def _inpaint_regions(image_path, word_bboxes, lines=None, pad=2):
    """Use OpenCV inpainting to cleanly remove text and lines from image.

    Creates a mask of text word bboxes and detected lines, then uses
    Telea inpainting to reconstruct the background seamlessly.

    Args:
        image_path: source image
        word_bboxes: list of (x1, y1, x2, y2) word bounding boxes
        lines: optional list of line dicts to also inpaint
        pad: padding around each word bbox

    Returns:
        (inpainted_image_path, None) on success, (None, error) on failure.
    """
    try:
        import cv2
    except ImportError:
        return None, "opencv-python required"

    img = cv2.imread(image_path)
    if img is None:
        return None, f"Cannot read: {image_path}"
    ih, iw = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Build inpainting mask
    mask = np.zeros((ih, iw), dtype=np.uint8)

    # Mask text regions — use adaptive thresholding within each word bbox
    # to mask only dark (text) pixels, not the background
    for (x1, y1, x2, y2) in word_bboxes:
        x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
        x2p, y2p = min(iw, x2 + pad), min(ih, y2 + pad)
        if x2p <= x1p or y2p <= y1p:
            continue

        region_gray = gray[y1p:y2p, x1p:x2p]
        # Otsu threshold to find text pixels (dark on light)
        thresh_val = cv2.threshold(region_gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Dilate slightly to catch anti-aliased edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh_val = cv2.dilate(thresh_val, kernel, iterations=1)
        mask[y1p:y2p, x1p:x2p] = np.maximum(mask[y1p:y2p, x1p:x2p], thresh_val)

    # Mask line regions (only if requested)
    if lines:
        for ln in lines:
            x1, y1 = int(ln['x1']), int(ln['y1'])
            x2, y2 = int(ln['x2']), int(ln['y2'])
            sw = max(2, int(ln['stroke_width'] + 4))
            cv2.line(mask, (x1, y1), (x2, y2), 255, sw)

    # Count masked pixels
    n_masked = np.count_nonzero(mask)
    if n_masked == 0:
        return image_path, None  # nothing to inpaint

    # Inpaint using Telea method (better for text removal)
    inpainted = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        out_path = tmp.name
    cv2.imwrite(out_path, inpainted)
    return out_path, None


def _detect_font_style(image_path, word_bboxes):
    """Detect if text in image is primarily serif or sans-serif.

    Checks stroke width variation at character stems — serif fonts have
    thicker/thinner variations and decorative strokes.

    Returns 'serif' or 'sans-serif'.
    """
    try:
        import cv2
    except ImportError:
        return 'sans-serif'

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'sans-serif'
    ih, iw = img.shape

    serif_score = 0
    total = 0
    for (x1, y1, x2, y2) in word_bboxes[:30]:  # sample first 30 words
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        region = img[y1:y2, x1:x2]
        # Binarize
        _, bw = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark = (bw == 0)
        if not dark.any():
            continue

        # Check horizontal stroke width variation per column
        col_widths = []
        for col in range(dark.shape[1]):
            w = np.sum(dark[:, col])
            if w > 0:
                col_widths.append(w)
        if len(col_widths) < 3:
            continue

        total += 1
        cv_val = np.std(col_widths) / (np.mean(col_widths) + 1e-6)
        # Serifs have higher stroke width variation (>0.6 typical)
        if cv_val > 0.65:
            serif_score += 1

    if total == 0:
        return 'sans-serif'
    return 'serif' if serif_score / total > 0.5 else 'sans-serif'


def png_to_svg_semantic(input_path, output_path=None, quality='balanced',
                        color_precision=None, upscale=None,
                        min_path_dlen=None, svg_precision=None,
                        merge_threshold=0, include_text=True, svgo=True,
                        text_mode='overlay'):
    """Full semantic SVG with visible <text>, <rect>, <line> elements.

    Architecture:
      1. Detect rectangles, lines, text from original image
      2. Inpaint text+line regions to cleanly remove them from source
      3. Trace the inpainted image (no text/line shapes in paths)
      4. Render VISIBLE semantic elements on top of traced background
      5. SVGO on paths, inject semantic elements after

    The result is a truly semantic SVG where:
      - Text is rendered as crisp <text> elements (not pixelated traces)
      - Rectangles are visible <rect> elements (editable in SVG editors)
      - Lines are visible <line> elements (editable connections)
      - Complex graphics (icons, gradients) remain as traced vector paths

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_semantic.svg"

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['balanced'])
    cp = color_precision if color_precision is not None else preset['color_precision']
    sf = upscale if upscale is not None else preset['upscale']
    sp = preset['speckle']
    min_dl = min_path_dlen if min_path_dlen is not None else preset['min_dlen']
    prec = svg_precision if svg_precision is not None else preset['precision']

    img = Image.open(input_path)
    width, height = img.size
    print(f"Input: {input_path} ({width}x{height})")
    print(f"Quality: {quality} (cp={cp}, upscale={sf}x, sp={sp})")

    # === Step 1: Detect semantic elements ===
    print("\n[1] Detecting rectangles...")
    rects, err = detect_rectangles_cv(input_path)
    if err:
        print(f"  Warning: {err}")
        rects = []
    else:
        print(f"  Found {len(rects)} rectangles")

    print("\n[2] Detecting lines/arrows...")
    lines, err = detect_lines_cv(input_path, rects=rects)
    if err:
        print(f"  Warning: {err}")
        lines = []
    else:
        arrows = sum(1 for ln in lines if ln['has_arrow'])
        print(f"  Found {len(lines)} lines ({arrows} with arrows)")

    print("\n[3] Detecting text...")
    text_result, err = detect_text_regions(input_path)
    if err:
        print(f"  Warning: {err}")
        text_lines = []
        word_bboxes = []
        all_words = []
    else:
        text_lines = text_result['lines']
        word_bboxes = text_result['word_bboxes']
        all_words = text_result.get('words', [])
        print(f"  Found {len(text_lines)} text lines, {len(word_bboxes)} words")

    # Detect font style
    font_style = 'sans-serif'
    if word_bboxes and include_text:
        font_style = _detect_font_style(input_path, word_bboxes)
        print(f"  Font style: {font_style}")
    if font_style == 'serif':
        font_family = "Georgia, 'Times New Roman', serif"
    else:
        font_family = "Arial, Helvetica, sans-serif"

    # === Step 2: Prepare trace input ===
    trace_input = input_path
    inpaint_path = None

    # === Step 3: Trace the inpainted image ===
    upscale_path = None
    if sf > 1.0:
        print(f"\n[5] Upscaling {sf}x...")
        up_img = Image.open(trace_input)
        uw, uh = up_img.size
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            upscale_path = tmp.name
        up_img.resize((int(uw * sf), int(uh * sf)),
                      Image.LANCZOS).save(upscale_path)
        trace_input = upscale_path

    print(f"\n[6] vtracer trace (cp={cp}, sp={sp})...")
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        vt_path = tmp.name
    vt_result, err = run_vtracer(
        trace_input, output_path=vt_path,
        color_precision=cp, path_precision=4,
        filter_speckle=sp, segment_length=3.5,
        gradient_step=1, mode='spline',
    )
    if err:
        return None, f"vtracer failed: {err}"
    print(f"  Raw: {os.path.getsize(vt_path)/1024:.0f}KB")

    # Parse and scale back
    print("\n[7] Parsing paths...")
    with open(vt_path) as f:
        svg_content = f.read()
    paths, err = parse_svg_paths(svg_content)
    if err:
        return None, err
    print(f"  Parsed: {len(paths)} paths")

    if sf > 1.0:
        scaled = []
        for p in paths:
            new_cmds = []
            for cmd, pts in p.commands:
                if cmd == 'M':
                    new_cmds.append(('M', [(pts[0][0]/sf, pts[0][1]/sf)]))
                elif cmd == 'L':
                    new_cmds.append(('L', [(pts[0][0]/sf, pts[0][1]/sf)]))
                elif cmd == 'C':
                    new_cmds.append(('C', [(x/sf, y/sf) for x, y in pts]))
                elif cmd == 'Q':
                    new_cmds.append(('Q', [(x/sf, y/sf) for x, y in pts]))
                elif cmd == 'A':
                    rx, ry, rot, la, sw, x, y = pts[0]
                    new_cmds.append(('A', [(rx/sf, ry/sf, rot, la, sw, x/sf, y/sf)]))
                elif cmd == 'Z':
                    new_cmds.append(('Z', []))
            scaled.append(SvgPath(fill=p.fill, fill_rule=p.fill_rule, commands=new_cmds))
        paths = scaled

    # Filter tiny paths
    d_lens = [len(_commands_to_d(p.commands, precision=prec)) for p in paths]
    filtered = [p for p, dl in zip(paths, d_lens) if dl >= min_dl]
    print(f"  Kept: {len(filtered)} paths (removed {len(paths)-len(filtered)} tiny)")

    if merge_threshold > 0:
        filtered, _ = merge_similar_colors(filtered, threshold=merge_threshold)

    # === Step 4: Build paths-only SVG for SVGO ===
    print(f"\n[8] Assembling SVG...")

    paths_svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
    ]

    for path in filtered:
        d = _commands_to_d(path.commands, precision=prec)
        if d:
            paths_svg_lines.append(
                f'  <path fill="{path.fill}" fill-rule="{path.fill_rule}" d="{d}"/>'
            )

    paths_svg_lines.append('</svg>')

    paths_only_path = output_path + '.paths.svg'
    with open(paths_only_path, 'w') as f:
        f.write('\n'.join(paths_svg_lines))
    pre_size = os.path.getsize(paths_only_path)
    print(f"  Paths SVG: {pre_size/1024:.0f}KB")

    # SVGO
    if svgo:
        print(f"\n[9] SVGO optimization...")
        svgo_out = paths_only_path + '.tmp'
        svgo_cfg = output_path + '.svgo.config.mjs'
        try:
            with open(svgo_cfg, 'w') as f:
                f.write('export default { plugins: [{ name: "preset-default", '
                        'params: { overrides: { removeUnknownsAndDefaults: false, '
                        'collapseGroups: false, minifyStyles: false, '
                        'removeUselessDefs: false, removeHiddenElems: false, '
                        'cleanupIds: false }}}]};\n')
            result = subprocess.run(
                ['npx', 'svgo', f'--precision={prec}',
                 '--config', svgo_cfg,
                 '-i', paths_only_path, '-o', svgo_out],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 and os.path.exists(svgo_out):
                import shutil
                shutil.move(svgo_out, paths_only_path)
                post_size = os.path.getsize(paths_only_path)
                print(f"  SVGO: {pre_size/1024:.0f}KB -> {post_size/1024:.0f}KB "
                      f"({post_size/pre_size*100:.0f}%)")
            else:
                print(f"  SVGO failed: {result.stderr[:200] if result.stderr else 'unknown'}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  SVGO not available or timed out")
        finally:
            for fp in [svgo_out, svgo_cfg]:
                if os.path.exists(fp):
                    try:
                        os.unlink(fp)
                    except OSError:
                        pass

    # === Step 5: Inject semantic elements ===
    print(f"\n[10] Injecting semantic elements...")

    with open(paths_only_path) as f:
        optimized_svg = f.read()

    optimized_svg = optimized_svg.rstrip()
    if optimized_svg.endswith('</svg>'):
        optimized_svg = optimized_svg[:-6]

    semantic_lines = []

    # Lines (invisible overlay for editability — traced paths render them)
    if lines:
        semantic_lines.append('  <!-- Detected connections (editable overlay) -->')
        for ln in lines:
            c = ln['color']
            color = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            sw = ln['stroke_width']
            arrow_attr = ' data-arrow="true"' if ln['has_arrow'] else ''
            semantic_lines.append(
                f'  <line x1="{ln["x1"]}" y1="{ln["y1"]}" '
                f'x2="{ln["x2"]}" y2="{ln["y2"]}" '
                f'stroke="{color}" stroke-width="{sw}" '
                f'stroke-opacity="0.01"{arrow_attr} '
                f'class="semantic-line" data-semantic="line"/>'
            )

    # Text (invisible overlay — traced paths render text visually,
    # these <text> elements provide selectability/searchability/editability)
    font_mult = 0.75
    if include_text and text_lines:
        semantic_lines.append('  <!-- Detected text (invisible overlay for selection) -->')
        for tl in text_lines:
            words = tl.get('words', [])
            if not words:
                words = [tl]
            for w in words:
                tc = w['text_color']
                fill = f"#{tc[0]:02x}{tc[1]:02x}{tc[2]:02x}"
                fs = max(5, w['font_size'] * font_mult)
                baseline_y = w['y'] + w['h'] * 0.75

                text = (w['text']
                        .replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;'))
                semantic_lines.append(
                    f'  <text x="{w["x"]}" y="{baseline_y:.1f}" '
                    f'font-family="{font_family}" '
                    f'font-size="{fs:.1f}" fill="{fill}" fill-opacity="0.01" '
                    f'textLength="{w["w"]}" lengthAdjust="spacingAndGlyphs" '
                    f'class="semantic-text" data-semantic="text">{text}</text>'
                )

    # Rectangles (invisible overlay for editability)
    if rects:
        semantic_lines.append('  <!-- Detected rectangles (editable overlay) -->')
        for r in rects:
            fill = f"#{r['fill_color'][0]:02x}{r['fill_color'][1]:02x}{r['fill_color'][2]:02x}"
            stroke_attr = ' stroke="none"'
            if r['stroke_color']:
                sc = r['stroke_color']
                stroke_attr = (f' stroke="#{sc[0]:02x}{sc[1]:02x}{sc[2]:02x}"'
                               ' stroke-width="1"')
            rx_attr = f' rx="{r["rx"]}"' if r['rx'] > 0 else ''
            semantic_lines.append(
                f'  <rect x="{r["x"]}" y="{r["y"]}" '
                f'width="{r["w"]}" height="{r["h"]}" '
                f'fill="{fill}" fill-opacity="0.01"{stroke_attr}{rx_attr} '
                f'class="semantic-rect" data-semantic="rect"/>'
            )

    final_svg = optimized_svg + '\n' + '\n'.join(semantic_lines) + '\n</svg>'

    with open(output_path, 'w') as f:
        f.write(final_svg)

    # Cleanup temp files
    try:
        os.unlink(paths_only_path)
    except OSError:
        pass
    for tmp in [vt_path, upscale_path, inpaint_path]:
        if tmp and tmp != input_path:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    out_size = os.path.getsize(output_path)
    n_text = sum(len(tl.get('words', [])) or 1 for tl in text_lines) if text_lines else 0
    print(f"\n  Output: {output_path} ({out_size/1024:.0f}KB)")
    print(f"  Semantic: {len(rects)} <rect>, {len(lines)} <line>, "
          f"{n_text} <text> (all visible) + {len(filtered)} traced paths")
    print(f"  Text: per-word, font={font_style}, mult={font_mult}")
    return output_path, None


# ---------------------------------------------------------------------------
# 11. Original Pipeline
# ---------------------------------------------------------------------------

def png_to_svg(input_path, output_path=None, n_colors=0,
               line_tol=0.3, arc_tol=0.6, quad_tol=0.4,
               vtracer_precision=6, vtracer_speckle=0,
               vtracer_segment=8.0, vtracer_gradient=1,
               color_precision=8, mode="pixel"):
    """Convert PNG to high-precision SVG.

    Args:
        mode: vtracer tracing mode - "pixel" (highest precision, ~0.998 SSIM),
              "spline" (smooth curves, ~0.964 SSIM), or "polygon".

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}.svg"

    # Get image dimensions
    img = Image.open(input_path)
    width, height = img.size
    print(f"Input: {input_path} ({width}x{height})")

    # Step 1: Color quantization (skip if n_colors=0)
    if n_colors > 0:
        print("\n[1] Color quantization...")
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            q_path = tmp.name
        q_result, err = quantize_colors(input_path, n_colors=n_colors,
                                         output_path=q_path)
        if err:
            print(f"  Warning: {err}, using original")
            q_path = input_path
        else:
            q_img = np.array(Image.open(q_path))
            unique = len(set(
                tuple(q_img[y, x]) for y in range(0, q_img.shape[0], 4)
                for x in range(0, q_img.shape[1], 4)
            ))
            print(f"  Quantized to ~{unique} colors: {q_path}")
    else:
        print("\n[1] Skipping color quantization (using original colors)")
        q_path = input_path

    # Step 2: vtracer tracing
    print("\n[2] vtracer tracing...")
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        vt_path = tmp.name
    vt_result, err = run_vtracer(
        q_path, output_path=vt_path,
        color_precision=color_precision,
        path_precision=vtracer_precision,
        filter_speckle=vtracer_speckle,
        segment_length=vtracer_segment,
        gradient_step=vtracer_gradient,
        mode=mode,
    )
    if err:
        return None, f"vtracer failed: {err}"
    vt_size = os.path.getsize(vt_path)
    print(f"  vtracer output: {vt_size/1024:.0f}KB")

    # For pixel mode: use vtracer output directly (no parse/re-serialize overhead)
    if mode == "pixel":
        print("\n[3-6] Pixel mode: using vtracer output directly (maximum precision)")
        import shutil
        shutil.copy2(vt_path, output_path)
    else:
        # Step 3: Parse SVG paths
        print("\n[3] Parsing SVG paths...")
        with open(vt_path) as f:
            svg_content = f.read()
        paths, err = parse_svg_paths(svg_content)
        if err:
            return None, err
        print(f"  Parsed {len(paths)} paths")

        # Count original command distribution
        orig_stats = defaultdict(int)
        for p in paths:
            for cmd_type, _ in p.commands:
                orig_stats[cmd_type] += 1
        total = sum(orig_stats.values())
        print(f"  Original commands ({total} total):")
        for cmd in ['M', 'L', 'A', 'Q', 'C', 'Z']:
            if orig_stats[cmd] > 0:
                print(f"    {cmd}: {orig_stats[cmd]:6d} ({orig_stats[cmd]/total*100:.1f}%)")

        # Step 4: Optimize paths (geometric primitive fitting)
        print("\n[4] Optimizing paths (geometric primitive fitting)...")
        opt_paths, err = optimize_all_paths(paths, line_tol, arc_tol, quad_tol)
        if err:
            return None, err

        # Step 5: Merge similar colors
        print("\n[5] Merging similar colors...")
        merged_paths, _ = merge_similar_colors(opt_paths, threshold=8)
        unique_colors = len(set(p.fill for p in merged_paths))
        print(f"  Final unique colors: {unique_colors}")

        # Step 6: Output SVG
        print("\n[6] Writing SVG...")
        svg_str, _ = paths_to_svg(merged_paths, width, height,
                                   precision=vtracer_precision)
        with open(output_path, 'w') as f:
            f.write(svg_str)

    out_size = os.path.getsize(output_path)
    print(f"  Output: {output_path} ({out_size/1024:.0f}KB)")

    # Cleanup temp files
    for tmp_path in [q_path, vt_path]:
        if tmp_path != input_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return output_path, None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def compute_ssim(original_path, svg_path, renderer="resvg"):
    """Compute SSIM between original image and rendered SVG.

    Args:
        renderer: "resvg" (recommended, pixel-perfect) or "chrome" (legacy).

    Returns:
        (ssim_score, None) on success, (None, error) on failure.
    """
    try:
        from skimage.metrics import structural_similarity
        from skimage.io import imread as sk_imread
        from skimage.transform import resize
        from skimage.color import rgb2gray

        img = Image.open(original_path)
        width, height = img.size

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            render_path = tmp.name

        if renderer == "resvg":
            resvg_paths = [
                os.path.expanduser("~/.cargo/bin/resvg"),
                "/usr/local/bin/resvg",
                "resvg",
            ]
            resvg = None
            for p in resvg_paths:
                if os.path.exists(p) or p == "resvg":
                    resvg = p
                    break
            if not resvg:
                return None, "resvg not found. Install with: cargo install resvg"

            result = subprocess.run([
                resvg, svg_path, render_path,
                '-w', str(width), '-h', str(height)
            ], capture_output=True, timeout=60)
            if result.returncode != 0:
                return None, f"resvg failed: {result.stderr.decode()}"

        elif renderer == "chrome":
            chrome_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/usr/bin/google-chrome",
                "/usr/bin/chromium-browser",
            ]
            chrome = None
            for p in chrome_paths:
                if os.path.exists(p):
                    chrome = p
                    break
            if not chrome:
                return None, "Chrome not found for SVG rendering"

            subprocess.run([
                chrome, '--headless', '--disable-gpu',
                f'--screenshot={render_path}',
                f'--window-size={width},{height}',
                '--default-background-color=00000000',
                f'file://{os.path.abspath(svg_path)}'
            ], capture_output=True, timeout=30)
        else:
            return None, f"Unknown renderer: {renderer}"

        orig = sk_imread(original_path)
        rendered = sk_imread(render_path)

        if orig.shape != rendered.shape:
            rendered = resize(
                rendered, orig.shape[:2],
                anti_aliasing=True, preserve_range=True
            ).astype(np.uint8)

        orig_g = rgb2gray(orig[:, :, :3])
        rendered_g = rgb2gray(rendered[:, :, :3])
        score = structural_similarity(orig_g, rendered_g, data_range=1.0)

        os.unlink(render_path)
        return score, None

    except ImportError:
        return None, "scikit-image not installed (pip install scikit-image)"
    except Exception as e:
        return None, f"SSIM computation failed: {e}"


# ---------------------------------------------------------------------------
# 13. SciPrism-style Converter (v2)
#     Color quantization → per-color contour → geometric primitive fitting
# ---------------------------------------------------------------------------

def _smooth_contour(pts, sigma=1.2):
    """Gaussian-smooth contour points (circular boundary aware)."""
    from scipy.ndimage import gaussian_filter1d
    n = len(pts)
    if n < 5:
        return pts
    # Pad circularly for smooth wrapping
    pad = min(n, int(sigma * 4))
    xs = np.concatenate([pts[-pad:, 0], pts[:, 0], pts[:pad, 0]])
    ys = np.concatenate([pts[-pad:, 1], pts[:, 1], pts[:pad, 1]])
    xs = gaussian_filter1d(xs, sigma)
    ys = gaussian_filter1d(ys, sigma)
    return np.column_stack([xs[pad:pad+n], ys[pad:pad+n]])


def _detect_corners(pts, angle_thresh=25.0, min_seg_len=3, window=5):
    """Detect corner indices by measuring angle deviation from 180°.

    Uses a multi-scale window to avoid detecting pixel-level staircase
    as corners. The angle is measured between vectors spanning `window`
    points on each side of the candidate.

    Returns list of indices where corners occur.
    """
    n = len(pts)
    if n < 5:
        return [0, n - 1]

    corners = [0]
    half = max(1, window)
    for i in range(half, n - half):
        v1 = pts[i] - pts[max(0, i - half)]
        v2 = pts[min(n - 1, i + half)] - pts[i]
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-6 or len2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
        angle = math.degrees(math.acos(cos_a))
        if angle < (180 - angle_thresh):
            if not corners or (i - corners[-1]) >= min_seg_len:
                corners.append(i)

    if corners[-1] != n - 1:
        corners.append(n - 1)
    return corners


def _fit_segment_primitive(pts, arc_tol=0.8, line_tol=0.5):
    """Fit the best geometric primitive to a contour segment.

    Returns list of (cmd, params) SVG commands.
    Tries: line → circular arc → elliptical arc → quadratic bezier → cubic bezier.
    """
    n = len(pts)
    if n <= 1:
        return []
    if n == 2:
        return [('L', [(pts[1][0], pts[1][1])])]

    p0, pN = pts[0], pts[-1]

    # --- Try straight line ---
    chord = np.linalg.norm(pN - p0)
    if chord < 1e-6:
        return [('L', [(pN[0], pN[1])])]

    # Max distance from line p0→pN
    direction = (pN - p0) / chord
    perp = np.array([-direction[1], direction[0]])
    max_dev = 0.0
    for pt in pts[1:-1]:
        dev = abs(np.dot(pt - p0, perp))
        max_dev = max(max_dev, dev)

    if max_dev < line_tol:
        return [('L', [(pN[0], pN[1])])]

    # --- Try arc fit (circle then ellipse) ---
    pts_list = [(float(p[0]), float(p[1])) for p in pts]

    # Subsample for fitting if too many points
    if n > 30:
        idx = np.linspace(0, n - 1, 30, dtype=int)
        fit_pts = [pts_list[i] for i in idx]
    else:
        fit_pts = pts_list

    # Circle fit
    circle = fit_circle(fit_pts)
    if circle is not None:
        cx, cy, r, residual = circle
        if residual < arc_tol and r > 0.3 and r < 5000:
            # Verify against ALL points
            max_r = max(abs(math.hypot(p[0]-cx, p[1]-cy) - r) for p in pts_list)
            if max_r < arc_tol * 2:
                arc = _circle_to_arc(pts_list[0], pts_list[-1], cx, cy, r)
                if arc is not None:
                    rx, ry, rot, la, sw, ex, ey = arc
                    # SciPrism style: always large_arc=0, split if needed
                    if la == 0:
                        return [('A', [(rx, ry, rot, 0, sw, ex, ey)])]
                    else:
                        # Split into two arcs at midpoint
                        mid_idx = n // 2
                        mid = pts_list[mid_idx]
                        arc1 = _circle_to_arc(pts_list[0], mid, cx, cy, r)
                        arc2 = _circle_to_arc(mid, pts_list[-1], cx, cy, r)
                        if arc1 and arc2:
                            cmds = []
                            cmds.append(('A', [(arc1[0], arc1[1], arc1[2], 0, arc1[4], mid[0], mid[1])]))
                            cmds.append(('A', [(arc2[0], arc2[1], arc2[2], 0, arc2[4], ex, ey)]))
                            return cmds

    # Ellipse fit
    if n >= 6:
        ellipse = fit_ellipse(fit_pts)
        if ellipse is not None:
            ecx, ecy, rx, ry, rot, residual = ellipse
            if residual < arc_tol and rx > 0.2 and ry > 0.2 and rx < 5000 and ry < 5000:
                arc = _ellipse_to_arc(pts_list[0], pts_list[-1], ecx, ecy, rx, ry, rot)
                if arc is not None:
                    arx, ary, arot, la, sw, ex, ey = arc
                    if la == 0:
                        return [('A', [(arx, ary, arot, 0, sw, ex, ey)])]
                    else:
                        mid_idx = n // 2
                        mid = pts_list[mid_idx]
                        arc1 = _ellipse_to_arc(pts_list[0], mid, ecx, ecy, rx, ry, rot)
                        arc2 = _ellipse_to_arc(mid, pts_list[-1], ecx, ecy, rx, ry, rot)
                        if arc1 and arc2:
                            cmds = []
                            cmds.append(('A', [(arc1[0], arc1[1], arc1[2], 0, arc1[4], mid[0], mid[1])]))
                            cmds.append(('A', [(arc2[0], arc2[1], arc2[2], 0, arc2[4], ex, ey)]))
                            return cmds

    # --- Try quadratic bezier ---
    if n >= 3:
        # Fit single Q bezier: find control point
        # Using least-squares: B(t) = (1-t)²P0 + 2t(1-t)CP + t²PN
        ts = np.linspace(0, 1, n)
        # Weights for control point
        w0 = (1 - ts) ** 2
        wc = 2 * ts * (1 - ts)
        wn = ts ** 2
        # residual = pts - w0*P0 - wn*PN, solve for CP via wc
        res_x = pts[:, 0] - w0 * p0[0] - wn * pN[0]
        res_y = pts[:, 1] - w0 * p0[1] - wn * pN[1]
        wc_sum = np.sum(wc ** 2)
        if wc_sum > 1e-10:
            cpx = np.sum(wc * res_x) / wc_sum
            cpy = np.sum(wc * res_y) / wc_sum
            # Check fit quality
            fitted_x = w0 * p0[0] + wc * cpx + wn * pN[0]
            fitted_y = w0 * p0[1] + wc * cpy + wn * pN[1]
            max_err = max(math.hypot(pts[i, 0] - fitted_x[i], pts[i, 1] - fitted_y[i])
                         for i in range(n))
            if max_err < arc_tol * 1.5:
                return [('Q', [(cpx, cpy), (pN[0], pN[1])])]

    # --- Fallback: cubic bezier ---
    if n >= 4:
        ts = np.linspace(0, 1, n)
        w0 = (1 - ts) ** 3
        w1 = 3 * ts * (1 - ts) ** 2
        w2 = 3 * ts ** 2 * (1 - ts)
        w3 = ts ** 3
        # Solve for cp1, cp2 via least squares
        res_x = pts[:, 0] - w0 * p0[0] - w3 * pN[0]
        res_y = pts[:, 1] - w0 * p0[1] - w3 * pN[1]
        A_mat = np.column_stack([w1, w2])
        try:
            sol_x, _, _, _ = np.linalg.lstsq(A_mat, res_x, rcond=None)
            sol_y, _, _, _ = np.linalg.lstsq(A_mat, res_y, rcond=None)
            return [('C', [(sol_x[0], sol_y[0]), (sol_x[1], sol_y[1]), (pN[0], pN[1])])]
        except np.linalg.LinAlgError:
            pass

    # Final fallback: line
    return [('L', [(pN[0], pN[1])])]


def _contour_to_commands(contour_pts, precision=4, arc_tol=0.8, line_tol=0.5,
                         dp_epsilon=0.4, angle_thresh=25.0):
    """Convert a contour (Nx2 array) to SVG path commands.

    Smooths the contour, detects corners, splits into segments,
    and fits the best primitive to each segment.

    Returns list of (cmd, params) tuples starting with M.
    """
    pts = contour_pts.astype(float)
    n = len(pts)
    if n < 3:
        if n == 2:
            return [('M', [(pts[0][0], pts[0][1])]),
                    ('L', [(pts[1][0], pts[1][1])]), ('Z', [])]
        return []

    # Simplify with Douglas-Peucker
    if dp_epsilon > 0:
        import cv2
        simplified = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.float32),
                                       epsilon=dp_epsilon, closed=True)
        pts = simplified.reshape(-1, 2).astype(float)
        n = len(pts)
        if n < 3:
            return []

    # Detect corners
    corners = _detect_corners(pts, angle_thresh=angle_thresh)

    # Build commands
    commands = [('M', [(pts[0][0], pts[0][1])])]

    for i in range(len(corners) - 1):
        seg_start = corners[i]
        seg_end = corners[i + 1]
        segment = pts[seg_start:seg_end + 1]
        if len(segment) < 2:
            continue
        cmds = _fit_segment_primitive(segment, arc_tol=arc_tol, line_tol=line_tol)
        commands.extend(cmds)

    # Close the path: last corner back to first point
    last_pt = pts[corners[-1]]
    first_pt = pts[0]
    if math.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1]) > 1.0:
        # Fit the closing segment
        closing_seg = np.vstack([pts[corners[-1]:], pts[:1]])
        if len(closing_seg) >= 2:
            cmds = _fit_segment_primitive(closing_seg, arc_tol=arc_tol, line_tol=line_tol)
            commands.extend(cmds)

    commands.append(('Z', []))
    return commands


# ---------------------------------------------------------------------------
# V4: Shapely pixel-boundary tracing with arc-fitted corners
# ---------------------------------------------------------------------------

def _pixels_to_polygon(mask):
    """Convert a binary mask to a Shapely polygon via run-length pixel boxes.

    Each True pixel at (x, y) maps to the unit box [x, y] → [x+1, y+1].
    unary_union merges all boxes into a precise pixel-boundary polygon with
    integer coordinates on pixel edges (not pixel centers).

    Returns:
        Shapely geometry (Polygon/MultiPolygon) or None if empty.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union

    boxes = []
    h, w = mask.shape
    for y in range(h):
        row = mask[y]
        x = 0
        while x < w:
            if row[x]:
                x_start = x
                while x < w and row[x]:
                    x += 1
                boxes.append(shapely_box(x_start, y, x, y + 1))
            else:
                x += 1
    return unary_union(boxes) if boxes else None


def _ring_to_arc_path(coords, radius):
    """Convert a polygon ring (list of coordinates) to an SVG path string
    with arcs at corners for smooth anti-aliasing.

    The input coordinates are typically on integer pixel edges (rectilinear).
    At each 90-degree corner, a small circular arc of the given radius is
    inserted.  Between corners, straight L commands connect the arc endpoints.

    This produces paths that are pixel-accurate at 1:1 rendering but appear
    smooth when zoomed, similar to SciPrism's sub-pixel arc approach.

    Returns:
        SVG path d-string, or None if too few points.
    """
    n = len(coords)
    if n < 4:
        return None

    r = radius
    parts = []

    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        x2, y2 = coords[(i + 2) % n]

        dx1, dy1 = x1 - x0, y1 - y0
        len1 = max(abs(dx1), abs(dy1))
        dx2, dy2 = x2 - x1, y2 - y1

        if len1 == 0:
            continue

        # First point: M to just before the first corner
        if i == 0:
            sx = x1 - (dx1 / len1) * min(r, len1 / 2)
            sy = y1 - (dy1 / len1) * min(r, len1 / 2)
            parts.append(f"M{sx:.1f},{sy:.1f}")

        # Straight segment leading to this corner
        if i > 0:
            ex = x1 - (dx1 / len1) * min(r, len1 / 2)
            ey = y1 - (dy1 / len1) * min(r, len1 / 2)
            parts.append(f"L{ex:.1f},{ey:.1f}")

        # Arc at the corner
        len2 = max(abs(dx2), abs(dy2))
        if len2 > 0:
            actual_r = min(r, len1 / 2, len2 / 2)
            if actual_r > 0.01:
                ax = x1 + (dx2 / len2) * min(r, len2 / 2)
                ay = y1 + (dy2 / len2) * min(r, len2 / 2)
                cross = dx1 * dy2 - dy1 * dx2
                sweep = 1 if cross > 0 else 0
                parts.append(
                    f"A{actual_r:.1f},{actual_r:.1f} 0 0 {sweep} "
                    f"{ax:.1f},{ay:.1f}"
                )

    parts.append("Z")
    return " ".join(parts)


def _polygon_to_compound_path(polygon, dp_tol=0.3, radius=0.5):
    """Convert a Shapely Polygon/MultiPolygon to a compound SVG path string
    with evenodd fill semantics.

    Steps:
      1. Optional Douglas-Peucker simplification (collapses redundant points
         on straight runs while preserving corners).
      2. Arc-fitted corners on each ring (exterior + holes).

    Returns:
        SVG d-string (possibly empty).
    """
    from shapely.geometry import MultiPolygon, Polygon

    if polygon is None or polygon.is_empty:
        return ""

    if dp_tol > 0:
        polygon = polygon.simplify(dp_tol, preserve_topology=True)

    polys = []
    if isinstance(polygon, MultiPolygon):
        polys = list(polygon.geoms)
    elif isinstance(polygon, Polygon):
        polys = [polygon]
    else:
        return ""

    subpaths = []
    for poly in polys:
        # Exterior
        coords = list(poly.exterior.coords)
        if coords[-1] == coords[0]:
            coords = coords[:-1]
        if len(coords) >= 4:
            d = _ring_to_arc_path(coords, radius)
            if d:
                subpaths.append(d)

        # Holes
        for interior in poly.interiors:
            coords = list(interior.coords)
            if coords[-1] == coords[0]:
                coords = coords[:-1]
            if len(coords) >= 4:
                d = _ring_to_arc_path(coords, radius)
                if d:
                    subpaths.append(d)

    return " ".join(subpaths)


def _two_stage_quantize(arr, n_achroma=10, n_chroma=22, denoise='bilateral'):
    """Two-stage color quantization that preserves chromatic diversity.

    Standard quantization (median-cut, K-means) allocates colors proportional
    to pixel count.  Images with mostly white/gray backgrounds waste most color
    slots on near-white variations, losing oranges, greens, purples, etc.

    This function splits pixels into achromatic (low chroma) and chromatic
    (high chroma) groups, allocates color slots separately, and reassigns every
    pixel to the combined palette.

    Args:
        denoise: Denoising method before quantization.
            'bilateral' — bilateral filter (d=7, sigma=25). Fast, preserves
                edges, but leaves many small noise fragments after quantization.
            'edge' — edge-preserving filter (sigma_s=60, sigma_r=0.4). Creates
                larger coherent color regions with fewer fragments, at the cost
                of slightly lower SSIM (~0.975 vs 0.989). Results in ~40% fewer
                subpaths and visually sharper output.

    Returns:
        (quantized_array, palette) — uint8 arrays.
    """
    import cv2
    from sklearn.cluster import MiniBatchKMeans
    from scipy.spatial.distance import cdist

    h, w = arr.shape[:2]

    # Denoise to reduce JPEG artifacts while preserving edges
    if denoise == 'edge':
        denoised = cv2.edgePreservingFilter(arr, flags=2, sigma_s=60,
                                             sigma_r=0.4)
    else:
        denoised = cv2.bilateralFilter(arr, 7, 25, 25)
    pixels = denoised.reshape(-1, 3).astype(float)

    # Split by chroma (max channel - min channel)
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    chroma_threshold = 20

    achroma_px = pixels[chroma <= chroma_threshold]
    chroma_px = pixels[chroma > chroma_threshold]

    # K-means on each group
    km_a = MiniBatchKMeans(n_clusters=n_achroma, random_state=42,
                           batch_size=1024)
    km_a.fit(achroma_px)

    actual_n_chroma = min(n_chroma, len(chroma_px))
    if actual_n_chroma < 2:
        palette = km_a.cluster_centers_.astype(np.uint8)
    else:
        km_c = MiniBatchKMeans(n_clusters=actual_n_chroma, random_state=42,
                               batch_size=1024)
        km_c.fit(chroma_px)
        palette = np.vstack([km_a.cluster_centers_,
                             km_c.cluster_centers_]).astype(np.uint8)

    # Assign every pixel to nearest palette color
    dists = cdist(pixels, palette.astype(float))
    labels = dists.argmin(axis=1)
    qarr = palette[labels].reshape(h, w, 3)

    return qarr, palette


def _contour_to_l_commands(contour, scale=0.5):
    """Convert an OpenCV contour to an SVG subpath using only M/L/Z commands.

    SciPrism-style: exact pixel-edge coordinates with 1 decimal place.
    Points are in upscaled space; multiplied by `scale` (0.5 for 2x upscale)
    to get half-pixel coordinates (X.0 or X.5).

    Returns d-string fragment like "M1.0,2.5L3.0,2.5L3.0,4.0Z" or None.
    """
    pts = contour.reshape(-1, 2)
    if len(pts) < 3:
        return None

    # Scale to display coordinates
    coords = pts.astype(float) * scale
    parts = [f"M{coords[0][0]:.1f},{coords[0][1]:.1f}"]
    for i in range(1, len(coords)):
        parts.append(f"L{coords[i][0]:.1f},{coords[i][1]:.1f}")
    parts.append("Z")
    return "".join(parts)


def _contour_pts_to_d(pts_2x, dp_epsilon=1.0, arc_tol=0.5, line_tol=0.3,
                      min_fit_pts=5):
    """Convert OpenCV contour points (in 2x-upscaled space) to SVG path d-string.

    Points are in 2x-upscaled coordinates; divided by 2 to get SVG coordinates.
    Applies DP simplification, then corner detection + conservative curve fitting.
    Short segments (< min_fit_pts) use L commands only; longer segments get
    arc/bezier fitting.
    """
    import cv2

    if len(pts_2x) < 3:
        return None

    # DP simplification in 2x space
    simplified = cv2.approxPolyDP(
        pts_2x.reshape(-1, 1, 2).astype(np.float32),
        epsilon=dp_epsilon, closed=True)
    # Scale to original coordinates
    pts = simplified.reshape(-1, 2).astype(float) / 2.0
    n = len(pts)
    if n < 3:
        return None

    # Detect corners and fit primitives
    corners = _detect_corners(pts, angle_thresh=30.0)
    cmds = [('M', [(pts[0][0], pts[0][1])])]

    for ci in range(len(corners) - 1):
        seg = pts[corners[ci]:corners[ci + 1] + 1]
        if len(seg) < 2:
            continue
        if len(seg) >= min_fit_pts:
            cmds.extend(_fit_segment_primitive(seg, arc_tol=arc_tol,
                                               line_tol=line_tol))
        else:
            for p in seg[1:]:
                cmds.append(('L', [(p[0], p[1])]))

    # Closing segment
    last_pt = pts[corners[-1]]
    first_pt = pts[0]
    if math.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1]) > 0.5:
        closing = np.vstack([pts[corners[-1]:], pts[:1]])
        if len(closing) >= 2:
            if len(closing) >= min_fit_pts:
                cmds.extend(_fit_segment_primitive(closing, arc_tol=arc_tol,
                                                   line_tol=line_tol))
            else:
                for p in closing[1:]:
                    cmds.append(('L', [(p[0], p[1])]))

    cmds.append(('Z', []))
    return _commands_to_d(cmds)


def _commands_to_d(commands, precision=2):
    """Serialize a list of (cmd, params) tuples to an SVG path d-string."""
    parts = []
    fmt = f"{{:.{precision}f}}"
    for cmd, params in commands:
        if cmd == 'Z':
            parts.append('Z')
        elif cmd == 'M':
            x, y = params[0]
            parts.append(f"M{fmt.format(x)},{fmt.format(y)}")
        elif cmd == 'L':
            x, y = params[0]
            parts.append(f"L{fmt.format(x)},{fmt.format(y)}")
        elif cmd == 'Q':
            cx, cy = params[0]
            ex, ey = params[1]
            parts.append(f"Q{fmt.format(cx)},{fmt.format(cy)} "
                         f"{fmt.format(ex)},{fmt.format(ey)}")
        elif cmd == 'C':
            c1x, c1y = params[0]
            c2x, c2y = params[1]
            ex, ey = params[2]
            parts.append(f"C{fmt.format(c1x)},{fmt.format(c1y)} "
                         f"{fmt.format(c2x)},{fmt.format(c2y)} "
                         f"{fmt.format(ex)},{fmt.format(ey)}")
        elif cmd == 'A':
            rx, ry, rot, la, sw, ex, ey = params[0]
            parts.append(f"A{fmt.format(rx)},{fmt.format(ry)} "
                         f"{fmt.format(rot)} {int(la)} {int(sw)} "
                         f"{fmt.format(ex)},{fmt.format(ey)}")
    return " ".join(parts)


def _superpixel_quantize(arr, n_colors=27, n_segments=3000, compactness=20):
    """Spatial-coherent quantization via SLIC superpixels.

    Instead of per-pixel color assignment (which creates noisy fragments),
    this segments the image into superpixels, computes mean color per
    superpixel, then clusters those means into n_colors.  Each superpixel
    is assigned to its nearest palette color, producing large coherent
    regions with clean boundaries from the start.

    Returns:
        (label_img, palette) — label_img is int32 with color indices,
        palette is uint8 array of shape (n_colors, 3).
    """
    import cv2
    from sklearn.cluster import MiniBatchKMeans
    from scipy.spatial.distance import cdist

    h, w = arr.shape[:2]

    # SLIC superpixel segmentation
    slic = cv2.ximgproc.createSuperpixelSLIC(
        arr, algorithm=cv2.ximgproc.SLICO,
        region_size=max(10, int(np.sqrt(h * w / n_segments))),
        ruler=float(compactness)
    )
    slic.iterate(10)
    slic.enforceLabelConnectivity(min_element_size=25)
    sp_labels = slic.getLabels()
    n_sp = slic.getNumberOfSuperpixels()
    print(f"  SLIC: {n_sp} superpixels")

    # Mean color per superpixel (vectorized)
    flat_labels = sp_labels.ravel()
    flat_pixels = arr.reshape(-1, 3).astype(np.float64)
    sp_means = np.zeros((n_sp, 3), dtype=np.float64)
    sp_counts = np.bincount(flat_labels, minlength=n_sp)
    for c in range(3):
        sp_means[:, c] = np.bincount(flat_labels, weights=flat_pixels[:, c],
                                      minlength=n_sp)
    nonzero = sp_counts > 0
    sp_means[nonzero] /= sp_counts[nonzero, np.newaxis]

    # Two-stage K-means on superpixel means (preserves chromatic diversity)
    chroma = sp_means.max(axis=1) - sp_means.min(axis=1)
    chroma_thresh = 20
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma

    achroma_mask = chroma <= chroma_thresh
    chroma_mask = chroma > chroma_thresh

    km_a = MiniBatchKMeans(n_clusters=min(n_achroma, achroma_mask.sum()),
                           random_state=42, batch_size=256)
    km_a.fit(sp_means[achroma_mask])

    actual_n_chroma = min(n_chroma, chroma_mask.sum())
    if actual_n_chroma >= 2:
        km_c = MiniBatchKMeans(n_clusters=actual_n_chroma,
                               random_state=42, batch_size=256)
        km_c.fit(sp_means[chroma_mask])
        palette = np.vstack([km_a.cluster_centers_,
                             km_c.cluster_centers_]).astype(np.uint8)
    else:
        palette = km_a.cluster_centers_.astype(np.uint8)

    # Assign each superpixel to nearest palette color
    dists = cdist(sp_means, palette.astype(float))
    sp_color_idx = dists.argmin(axis=1)

    # Build label image
    label_img = sp_color_idx[sp_labels]

    return label_img, palette


def _merge_small_components(label_img, min_area=10, palette=None,
                            max_color_dist=None):
    """Merge small connected components into their surrounding color.

    For each color, finds connected components smaller than min_area and
    reassigns their pixels to the most common neighboring color.

    If palette and max_color_dist are provided, only merges when the color
    distance between the component's color and the neighbor's color is
    below max_color_dist.  This preserves high-contrast small features
    (text, thin lines) while removing noise fragments.

    Returns:
        Modified label_img (in-place).
    """
    import cv2

    h, w = label_img.shape
    kernel = np.ones((3, 3), np.uint8)
    n_colors = label_img.max() + 1

    for ci in range(n_colors):
        mask = (label_img == ci).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)

        for li in range(1, n_labels):
            area = stats[li, cv2.CC_STAT_AREA]
            if area >= min_area:
                continue
            # Find this component's pixels
            comp_mask = (labels == li).astype(np.uint8)
            # Dilate to find border pixels
            dilated = cv2.dilate(comp_mask, kernel, iterations=1)
            border = (dilated - comp_mask).astype(bool)
            # Most common neighbor label (excluding self)
            border_labels = label_img[border]
            if len(border_labels) == 0:
                continue
            candidates = border_labels[border_labels != ci]
            if len(candidates) == 0:
                continue
            vals, cnts = np.unique(candidates, return_counts=True)
            best_label = vals[cnts.argmax()]

            # Color-distance guard: skip if colors are too different
            if palette is not None and max_color_dist is not None:
                c1 = palette[ci].astype(float)
                c2 = palette[best_label].astype(float)
                dist = np.sqrt(np.sum((c1 - c2) ** 2))
                if dist > max_color_dist:
                    continue

            label_img[comp_mask.astype(bool)] = best_label

    return label_img


def png_to_svg_v6(input_path, output_path=None, n_colors=32,
                  min_cc_area=10, denoise='edge',
                  svgo=True):
    """SciPrism-quality PNG→SVG via multilabel-potrace shared-boundary tracing.

    Uses the same algorithm as SciPrism: multilabel-potrace traces boundaries
    between ALL adjacent color regions simultaneously, producing exact shared
    coordinates — no gaps, no dilation needed.

    Pipeline:
      1. Edge-preserving denoise → two-stage color quantization
      2. Color-guarded CC merge (removes low-contrast noise only)
      3. Split into 8-connected components → multilabel-potrace (polygon mode)
      4. Merge same-color paths into compound paths (one per color)
      5. SVGO compression

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time
    import cv2

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v6.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Edge-preserving denoise + two-stage quantization
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma
    print(f"\n[1] Quantizing to {n_colors} colors "
          f"({n_achroma} achromatic + {n_chroma} chromatic, "
          f"denoise={denoise})...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma,
                                         denoise=denoise)

    colors_flat = qarr.reshape(-1, 3)
    unique_colors = np.unique(colors_flat, axis=0)
    label_img = np.zeros((h, w), dtype=np.int32)
    for ci, c in enumerate(unique_colors):
        label_img[np.all(qarr == c, axis=2)] = ci
    print(f"  {len(unique_colors)} colors in {time.time()-t0:.1f}s")

    # 3. Color-guarded CC merge — remove noise fragments with similar colors
    print(f"\n[2] Merging small components (min_area={min_cc_area})...")
    t0 = time.time()
    _merge_small_components(label_img, min_area=min_cc_area,
                           palette=unique_colors,
                           max_color_dist=40)

    used_labels = np.unique(label_img)
    remap = np.zeros(len(unique_colors), dtype=np.int32)
    new_colors = []
    for new_idx, old_idx in enumerate(used_labels):
        remap[old_idx] = new_idx
        new_colors.append(unique_colors[old_idx])
    new_colors = np.array(new_colors, dtype=np.uint8)
    label_img_new = remap[label_img]
    print(f"  Done in {time.time()-t0:.1f}s")

    # Count pixels per color (for stacking order)
    counts = np.bincount(label_img_new.ravel(), minlength=len(new_colors))
    order = np.argsort(-counts)  # largest area first

    # 4. multilabel-potrace: shared-boundary tracing (SciPrism algorithm)
    #    Traces boundaries between ALL adjacent color regions simultaneously,
    #    producing exact shared coordinates — no gaps, no dilation needed.
    print(f"\n[3] multilabel-potrace shared-boundary tracing...")
    t0 = time.time()

    try:
        import multilabel_potrace_svg as mlp
    except ImportError:
        return None, ("multilabel-potrace not installed. "
                      "Install from https://gitlab.com/1a7r0ch3/multilabel-potrace")

    # Split label image into 8-connected components (mlp requirement)
    n_labels = len(new_colors)
    comp_img = np.zeros((h, w), dtype=np.int32)
    comp_colors = []
    comp_id = 0
    for ci in range(n_labels):
        mask = (label_img_new == ci).astype(np.uint8)
        n_cc, cc_labels = cv2.connectedComponents(mask, connectivity=8)
        for cc in range(1, n_cc):
            comp_img[cc_labels == cc] = comp_id
            comp_colors.append(new_colors[ci])
            comp_id += 1
    print(f"  {comp_id} 8-connected components")

    comp_colors_arr = np.array(comp_colors, dtype=np.uint8)
    label_16 = np.ascontiguousarray(comp_img.astype(np.uint16))

    # Write raw mlp output to temp file, then post-process
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmp_path = tmp.name

    mlp.multilabel_potrace_svg(
        label_16, tmp_path,
        straight_line_tol=0.0,   # exact pixel boundaries (polygon mode)
        smoothing=0.0,           # no curves — L-only like SciPrism
        curve_fusion_tol=0.0,
        comp_colors=comp_colors_arr,
        line_width=0,
    )

    # Post-process: merge same-color paths into compound paths
    import xml.etree.ElementTree as ET2
    from collections import defaultdict
    ET2.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ET2.parse(tmp_path)
    root = tree.getroot()
    os.unlink(tmp_path)

    color_groups = defaultdict(list)
    for elem in root:
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'path':
            fill = elem.get('fill', '')
            d = elem.get('d', '')
            if d:
                color_groups[fill].append(d)

    # Convert rgb() fills to hex, count subpaths per color
    import re as _re
    def _rgb_to_hex(rgb_str):
        m = _re.match(r'rgb\((\d+),(\d+),(\d+)\)', rgb_str)
        if m:
            return f"#{int(m.group(1)):02x}{int(m.group(2)):02x}{int(m.group(3)):02x}"
        return rgb_str

    # Build SVG with compound paths (one per color, largest-area first)
    color_data = []
    for rgb_fill, ds in color_groups.items():
        hex_fill = _rgb_to_hex(rgb_fill)
        total_d_len = sum(len(d) for d in ds)
        color_data.append((hex_fill, ds, total_d_len))
    color_data.sort(key=lambda x: -x[2])  # largest first

    bg_color = color_data[0][0] if color_data else "#ffffff"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{bg_color}"/>',
    ]

    total_subpaths = 0
    path_count = 0
    for hex_fill, ds, _ in color_data:
        if hex_fill == bg_color:
            continue
        compound_d = ''.join(ds)
        svg_lines.append(
            f'<path fill="{hex_fill}" fill-rule="evenodd" '
            f'd="{compound_d}"/>'
        )
        path_count += 1
        total_subpaths += len(ds)

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    print(f"  {path_count} paths, {total_subpaths} subpaths "
          f"in {time.time()-t0:.1f}s")

    # 5. Write output
    with open(output_path, 'w') as f:
        f.write(svg_content)

    pre_size = os.path.getsize(output_path)
    print(f"\n  Pre-SVGO size: {pre_size/1024:.0f}KB")

    # 6. SVGO compression
    if svgo:
        print(f"\n[4] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed: {result.stderr[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  SVGO skipped: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {path_count} paths, "
          f"{total_subpaths} subpaths, {len(new_colors)} colors")

    return output_path, None


def _subpixel_contour_to_commands(pts, arc_tol=0.3, line_tol=0.2,
                                  angle_thresh=60.0, min_fit_pts=4):
    """Convert sub-pixel contour points to SVG path commands (L/A/Q/C).

    Unlike _contour_to_commands, this operates on smooth sub-pixel points
    from marching squares — no DP simplification needed.

    Args:
        pts: Nx2 float array of (x, y) sub-pixel coordinates.
        arc_tol: Max residual for arc fitting.
        line_tol: Max deviation for line fitting.
        angle_thresh: Corner detection angle threshold (degrees).
        min_fit_pts: Minimum points in a segment to attempt curve fitting.

    Returns:
        List of (cmd, params) tuples, or empty list.
    """
    n = len(pts)
    if n < 3:
        return []

    # Detect corners using multi-scale window
    corners = _detect_corners(pts, angle_thresh=angle_thresh,
                              min_seg_len=5, window=5)

    commands = [('M', [(pts[0][0], pts[0][1])])]

    for i in range(len(corners) - 1):
        seg = pts[corners[i]:corners[i + 1] + 1]
        if len(seg) < 2:
            continue
        if len(seg) >= min_fit_pts:
            cmds = _fit_segment_primitive(seg, arc_tol=arc_tol,
                                           line_tol=line_tol)
            commands.extend(cmds)
        else:
            # Too few points — use lines
            for p in seg[1:]:
                commands.append(('L', [(p[0], p[1])]))

    # Closing segment
    last_pt = pts[corners[-1]]
    first_pt = pts[0]
    if math.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1]) > 0.3:
        closing = np.vstack([pts[corners[-1]:], pts[:1]])
        if len(closing) >= 2:
            if len(closing) >= min_fit_pts:
                cmds = _fit_segment_primitive(closing, arc_tol=arc_tol,
                                               line_tol=line_tol)
                commands.extend(cmds)
            else:
                for p in closing[1:]:
                    commands.append(('L', [(p[0], p[1])]))

    commands.append(('Z', []))
    return commands


def png_to_svg_v8(input_path, output_path=None, n_colors=32,
                  min_cc_area=10, denoise='edge', svgo=True,
                  sigma=0.5, arc_tol=0.3, line_tol=0.2):
    """SciPrism-quality PNG→SVG: sub-pixel marching squares + curve fitting.

    Key innovation: instead of tracing pixel-edge boundaries (OpenCV), uses
    Gaussian blur + marching squares (skimage.find_contours) to extract
    smooth sub-pixel contour points. Then fits L/A/Q/C curves to these
    smooth points — same strategy as SciPrism.

    Pipeline:
      1. Edge-preserving denoise → two-stage color quantization
      2. Color-guarded CC merge
      3. Per-color: Gaussian blur → marching squares at level 0.5
      4. Per-contour: corner detection → curve fitting (line/arc/bezier)
      5. Compound paths per color with fill-rule="evenodd"
      6. SVGO compression

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time
    import cv2
    from skimage.measure import find_contours

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v8.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Quantize
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma
    print(f"\n[1] Quantizing to {n_colors} colors (denoise={denoise})...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma, denoise=denoise)

    colors_flat = qarr.reshape(-1, 3)
    unique_colors = np.unique(colors_flat, axis=0)
    label_img = np.zeros((h, w), dtype=np.int32)
    for ci, c in enumerate(unique_colors):
        label_img[np.all(qarr == c, axis=2)] = ci
    print(f"  {len(unique_colors)} colors in {time.time()-t0:.1f}s")

    # 3. CC merge
    print(f"\n[2] Merging small components (min_area={min_cc_area})...")
    t0 = time.time()
    _merge_small_components(label_img, min_area=min_cc_area,
                           palette=unique_colors, max_color_dist=40)

    used_labels = np.unique(label_img)
    remap = np.zeros(len(unique_colors), dtype=np.int32)
    new_colors = []
    for new_idx, old_idx in enumerate(used_labels):
        remap[old_idx] = new_idx
        new_colors.append(unique_colors[old_idx])
    new_colors = np.array(new_colors, dtype=np.uint8)
    label_img_new = remap[label_img]
    print(f"  Done in {time.time()-t0:.1f}s")

    counts = np.bincount(label_img_new.ravel(), minlength=len(new_colors))
    order = np.argsort(-counts)

    # 4. Sub-pixel contour tracing + curve fitting
    print(f"\n[3] Sub-pixel contours (sigma={sigma}) + curve fitting...")
    t0 = time.time()

    bg_idx = order[0]
    bg_color = f"#{new_colors[bg_idx][0]:02x}" \
               f"{new_colors[bg_idx][1]:02x}{new_colors[bg_idx][2]:02x}"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{bg_color}"/>',
    ]

    total_paths = 0
    total_subpaths = 0
    min_contour_len = 6  # minimum points for a meaningful contour

    for color_idx in order:
        if color_idx == bg_idx or counts[color_idx] < 4:
            continue

        hex_color = f"#{new_colors[color_idx][0]:02x}" \
                    f"{new_colors[color_idx][1]:02x}" \
                    f"{new_colors[color_idx][2]:02x}"

        # Binary mask → dilate → Gaussian blur → marching squares
        mask = (label_img_new == color_idx).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated.astype(np.float32),
                                   (0, 0), sigmaX=sigma)
        contours = find_contours(blurred, level=0.5)

        if not contours:
            continue

        # Convert all contours to SVG subpath commands
        # find_contours returns (row, col); we need (x=col, y=row)
        all_d_parts = []
        for contour in contours:
            if len(contour) < min_contour_len:
                continue
            # (row, col) → (x, y)
            pts = contour[:, ::-1].copy()

            cmds = _subpixel_contour_to_commands(
                pts, arc_tol=arc_tol, line_tol=line_tol)
            if cmds:
                d = _commands_to_d(cmds, precision=4)
                all_d_parts.append(d)
                total_subpaths += 1

        if all_d_parts:
            compound_d = " ".join(all_d_parts)
            svg_lines.append(
                f'<path fill="{hex_color}" fill-rule="evenodd" '
                f'd="{compound_d}"/>')
            total_paths += 1

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    print(f"  {total_paths} paths, {total_subpaths} subpaths "
          f"in {time.time()-t0:.1f}s")

    # 5. Write output
    with open(output_path, 'w') as f:
        f.write(svg_content)

    pre_size = os.path.getsize(output_path)
    print(f"\n  Pre-SVGO size: {pre_size/1024:.0f}KB")

    # 6. SVGO compression
    if svgo:
        print(f"\n[4] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed: {result.stderr[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  SVGO skipped: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {total_paths} paths, "
          f"{total_subpaths} subpaths, {len(new_colors)} colors")

    return output_path, None


def _parse_polygon_d(d_str):
    """Parse an SVG polygon path d-string (M/L/Z only) into list of contours.

    Each contour is a Nx2 numpy array of (x, y) float coordinates.
    """
    import re
    contours = []
    current = []
    tokens = re.findall(r'[MLZmlhvHV]|[-]?\d+\.?\d*', d_str)
    cmd = None
    cx, cy = 0.0, 0.0  # current point
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ('M', 'L', 'Z', 'm', 'l', 'z', 'H', 'V', 'h', 'v'):
            cmd = t
            i += 1
        else:
            if cmd == 'M':
                if current:
                    contours.append(np.array(current, dtype=np.float64))
                    current = []
                cx, cy = float(tokens[i]), float(tokens[i + 1])
                current.append([cx, cy])
                i += 2
                cmd = 'L'  # implicit L after M
            elif cmd == 'L':
                cx, cy = float(tokens[i]), float(tokens[i + 1])
                current.append([cx, cy])
                i += 2
            elif cmd == 'H':
                cx = float(tokens[i])
                current.append([cx, cy])
                i += 1
            elif cmd == 'V':
                cy = float(tokens[i])
                current.append([cx, cy])
                i += 1
            elif cmd == 'h':
                cx += float(tokens[i])
                current.append([cx, cy])
                i += 1
            elif cmd == 'v':
                cy += float(tokens[i])
                current.append([cx, cy])
                i += 1
            elif cmd == 'm':
                if current:
                    contours.append(np.array(current, dtype=np.float64))
                    current = []
                cx += float(tokens[i])
                cy += float(tokens[i + 1])
                current.append([cx, cy])
                i += 2
                cmd = 'l'
            elif cmd == 'l':
                cx += float(tokens[i])
                cy += float(tokens[i + 1])
                current.append([cx, cy])
                i += 2
            else:
                i += 1
        if cmd in ('Z', 'z'):
            if current:
                contours.append(np.array(current, dtype=np.float64))
                current = []
            cmd = None
            i += 0  # Z consumed above
    if current:
        contours.append(np.array(current, dtype=np.float64))
    return contours


def _fit_contour_curves(pts, arc_tol=0.8, line_tol=0.5,
                        angle_thresh=60.0, min_fit_pts=5, window=5):
    """Convert polygon contour vertices to SVG path with L/A/Q/C curves.

    Takes integer-coordinate polygon vertices from mlp and fits the best
    geometric primitive (line/arc/bezier) to each segment between corners.

    Returns list of (cmd, params) tuples.
    """
    n = len(pts)
    if n < 3:
        cmds = [('M', [(pts[0][0], pts[0][1])])]
        for p in pts[1:]:
            cmds.append(('L', [(p[0], p[1])]))
        cmds.append(('Z', []))
        return cmds

    # Detect corners using multi-scale window
    corners = _detect_corners(pts, angle_thresh=angle_thresh,
                              min_seg_len=min_fit_pts, window=window)

    commands = [('M', [(pts[0][0], pts[0][1])])]

    for i in range(len(corners) - 1):
        seg = pts[corners[i]:corners[i + 1] + 1]
        if len(seg) < 2:
            continue
        if len(seg) >= min_fit_pts:
            cmds = _fit_segment_primitive(seg, arc_tol=arc_tol,
                                          line_tol=line_tol)
            commands.extend(cmds)
        else:
            for p in seg[1:]:
                commands.append(('L', [(p[0], p[1])]))

    # Closing segment
    last_pt = pts[corners[-1]]
    first_pt = pts[0]
    dist = math.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1])
    if dist > 0.3:
        closing = np.vstack([pts[corners[-1]:], pts[:1]])
        if len(closing) >= 2:
            if len(closing) >= min_fit_pts:
                cmds = _fit_segment_primitive(closing, arc_tol=arc_tol,
                                              line_tol=line_tol)
                commands.extend(cmds)
            else:
                for p in closing[1:]:
                    commands.append(('L', [(p[0], p[1])]))

    commands.append(('Z', []))
    return commands


def _adjust_vertex_subpixel(pts):
    """Potrace-style vertex adjustment: move polygon vertices to sub-pixel
    positions using best-fit line intersections.

    For each vertex, computes the best-fit line through the neighboring
    points on each side, then moves the vertex to the intersection of
    the two lines (constrained within 0.5 pixel of original position).

    Returns adjusted Nx2 float array.
    """
    n = len(pts)
    if n < 3:
        return pts.copy()

    adjusted = pts.astype(np.float64).copy()

    for i in range(n):
        # Points on the "incoming" side (prev direction)
        # and "outgoing" side (next direction)
        prev_pts = []
        next_pts = []
        for k in range(1, min(4, n // 2)):
            prev_pts.append(pts[(i - k) % n])
            next_pts.append(pts[(i + k) % n])

        if len(prev_pts) < 2 or len(next_pts) < 2:
            continue

        # Best-fit line through incoming points + current
        in_pts = np.array([pts[i]] + prev_pts)
        out_pts = np.array([pts[i]] + next_pts)

        # Direction vectors via SVD
        in_center = in_pts.mean(axis=0)
        in_svd = np.linalg.svd(in_pts - in_center)
        in_dir = in_svd[2][0]  # first principal component

        out_center = out_pts.mean(axis=0)
        out_svd = np.linalg.svd(out_pts - out_center)
        out_dir = out_svd[2][0]

        # Find intersection of two lines
        # Line 1: in_center + t * in_dir
        # Line 2: out_center + s * out_dir
        # Solve: in_center + t * in_dir = out_center + s * out_dir
        A_mat = np.column_stack([in_dir, -out_dir])
        b_vec = out_center - in_center

        det = A_mat[0, 0] * A_mat[1, 1] - A_mat[0, 1] * A_mat[1, 0]
        if abs(det) < 1e-6:
            continue  # Lines are parallel

        t = (b_vec[0] * A_mat[1, 1] - b_vec[1] * A_mat[0, 1]) / det
        intersection = in_center + t * in_dir

        # Constrain within 0.5 pixel of original
        dx = intersection[0] - pts[i][0]
        dy = intersection[1] - pts[i][1]
        dx = max(-0.5, min(0.5, dx))
        dy = max(-0.5, min(0.5, dy))
        adjusted[i] = pts[i][0] + dx, pts[i][1] + dy

    return adjusted


def _polygon_to_curved_path(pts, arc_radius=0.5, min_corner_angle=150.0):
    """Convert a simplified polygon to SVG path with arc-rounded corners.

    SciPrism-style: at each polygon vertex where the angle is less than
    min_corner_angle degrees, insert a small elliptical arc to smooth the
    corner. Straight segments remain as L commands.

    Args:
        pts: Nx2 float array of (sub-pixel adjusted) polygon vertices.
        arc_radius: Max radius for corner arcs (pixels).
        min_corner_angle: Angles above this are kept straight (degrees).

    Returns:
        List of (cmd, params) tuples.
    """
    n = len(pts)
    if n < 3:
        cmds = [('M', [(pts[0][0], pts[0][1])])]
        for p in pts[1:]:
            cmds.append(('L', [(p[0], p[1])]))
        cmds.append(('Z', []))
        return cmds

    # Compute angles at each vertex
    angles = []
    for i in range(n):
        v_in = pts[i] - pts[(i - 1) % n]
        v_out = pts[(i + 1) % n] - pts[i]
        len_in = np.linalg.norm(v_in)
        len_out = np.linalg.norm(v_out)
        if len_in < 1e-6 or len_out < 1e-6:
            angles.append(180.0)
            continue
        cos_a = np.clip(np.dot(v_in, v_out) / (len_in * len_out), -1, 1)
        angles.append(math.degrees(math.acos(cos_a)))

    # Compute edge lengths
    edge_lens = []
    for i in range(n):
        edge_lens.append(np.linalg.norm(pts[(i + 1) % n] - pts[i]))

    # Build path commands with arc-rounded corners
    commands = []

    # For each vertex, compute tangent points if it gets an arc
    tangent_in = [None] * n   # point on incoming edge where arc starts
    tangent_out = [None] * n  # point on outgoing edge where arc ends
    arc_params = [None] * n

    for i in range(n):
        if angles[i] >= min_corner_angle:
            continue  # Keep as straight corner

        # Compute maximum arc radius for this corner
        d_in = edge_lens[(i - 1) % n]
        d_out = edge_lens[i]
        r = min(arc_radius, d_in * 0.4, d_out * 0.4)
        if r < 0.05:
            continue

        half_angle = angles[i] / 2.0
        if half_angle < 1.0:
            continue  # Nearly straight

        # Tangent distance from corner vertex
        tan_dist = r / math.tan(math.radians(half_angle))
        if tan_dist > d_in * 0.45 or tan_dist > d_out * 0.45:
            continue

        # Direction vectors
        v_in = pts[i] - pts[(i - 1) % n]
        v_in = v_in / np.linalg.norm(v_in)
        v_out = pts[(i + 1) % n] - pts[i]
        v_out = v_out / np.linalg.norm(v_out)

        # Tangent points
        t_in = pts[i] - v_in * tan_dist
        t_out = pts[i] + v_out * tan_dist

        tangent_in[i] = t_in
        tangent_out[i] = t_out

        # Determine sweep direction using cross product
        cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
        sweep = 1 if cross > 0 else 0

        arc_params[i] = (r, r, 0.0, 0, sweep,
                         t_out[0], t_out[1])

    # Generate path
    # Find first vertex without arc to start
    start = 0
    for i in range(n):
        if arc_params[i] is None:
            start = i
            break

    # Start at the first vertex (or its tangent-out if it has arc)
    if arc_params[start] is not None:
        commands.append(('M', [(tangent_out[start][0],
                                tangent_out[start][1])]))
    else:
        commands.append(('M', [(pts[start][0], pts[start][1])]))

    for step in range(1, n + 1):
        i = (start + step) % n

        if arc_params[i] is not None:
            # Line to tangent-in, then arc to tangent-out
            commands.append(('L', [(tangent_in[i][0], tangent_in[i][1])]))
            rx, ry, rot, la, sw, ex, ey = arc_params[i]
            commands.append(('A', [(rx, ry, rot, la, sw, ex, ey)]))
        else:
            # Straight line to this vertex
            commands.append(('L', [(pts[i][0], pts[i][1])]))

    commands.append(('Z', []))
    return commands


def png_to_svg_v10(input_path, output_path=None, n_colors=32,
                   min_cc_area=10, denoise='edge', svgo=True,
                   dp_epsilon=0.7, arc_radius=0.3, min_corner_angle=130.0,
                   min_arc_verts=8):
    """PNG→SVG replicating SciPrism's strategy.

    Pipeline (reverse-engineered from SciPrism output + Potrace paper):
      1. Color quantization + CC merge
      2. multilabel-potrace raw polygon tracing (shared boundaries)
      3. Douglas-Peucker simplification (replaces potrace optimal polygon)
      4. Corner arc fitting (small arcs at staircase corners, large polygons only)
      5. Merge same-color paths + SVGO

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time
    import cv2
    import re

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v10.svg"

    # 1. Load
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Quantize
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma
    print(f"\n[1] Quantizing to {n_colors} colors (denoise={denoise})...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma, denoise=denoise)
    colors_flat = qarr.reshape(-1, 3)
    unique_colors = np.unique(colors_flat, axis=0)
    label_img = np.zeros((h, w), dtype=np.int32)
    for ci, c in enumerate(unique_colors):
        label_img[np.all(qarr == c, axis=2)] = ci
    print(f"  {len(unique_colors)} colors in {time.time()-t0:.1f}s")

    # 3. CC merge
    print(f"\n[2] Merging small components (min_area={min_cc_area})...")
    t0 = time.time()
    _merge_small_components(label_img, min_area=min_cc_area,
                           palette=unique_colors, max_color_dist=40)
    used_labels = np.unique(label_img)
    remap = np.zeros(len(unique_colors), dtype=np.int32)
    new_colors = []
    for new_idx, old_idx in enumerate(used_labels):
        remap[old_idx] = new_idx
        new_colors.append(unique_colors[old_idx])
    new_colors = np.array(new_colors, dtype=np.uint8)
    label_img_new = remap[label_img]
    print(f"  Done in {time.time()-t0:.1f}s")

    # 4. multilabel-potrace raw polygon tracing
    print(f"\n[3] multilabel-potrace raw polygon tracing...")
    t0 = time.time()
    try:
        import multilabel_potrace_svg as mlp
    except ImportError:
        return None, "multilabel-potrace not installed"

    n_labels = len(new_colors)
    comp_img = np.zeros((h, w), dtype=np.int32)
    comp_colors_list = []
    comp_id = 0
    for ci in range(n_labels):
        mask = (label_img_new == ci).astype(np.uint8)
        n_cc, cc_labels = cv2.connectedComponents(mask, connectivity=8)
        for cc in range(1, n_cc):
            comp_img[cc_labels == cc] = comp_id
            comp_colors_list.append(new_colors[ci])
            comp_id += 1
    print(f"  {comp_id} 8-connected components")

    comp_colors_arr = np.array(comp_colors_list, dtype=np.uint8)
    label_16 = np.ascontiguousarray(comp_img.astype(np.uint16))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmp_path = tmp.name

    mlp.multilabel_potrace_svg(
        label_16, tmp_path,
        straight_line_tol=0.0,  # Raw polygon (no simplification)
        smoothing=0.0,
        curve_fusion_tol=0.0,
        comp_colors=comp_colors_arr, line_width=0,
    )
    print(f"  mlp done in {time.time()-t0:.1f}s")

    # 5. Parse polygons → DP simplify → corner arcs
    print(f"\n[4] DP simplify (eps={dp_epsilon}) + corner arcs "
          f"(r={arc_radius}, angle={min_corner_angle})...")
    t0 = time.time()

    path_pattern = re.compile(r'<path d="([^"]+)" fill="([^"]+)"/>')
    with open(tmp_path) as f:
        mlp_svg = f.read()
    os.unlink(tmp_path)

    matches = path_pattern.findall(mlp_svg)

    from collections import defaultdict
    color_d_parts = defaultdict(list)
    total_subpaths = 0
    n_arcs = 0
    n_lines = 0

    for d, fill in matches:
        contours = _parse_polygon_d(d)
        for contour in contours:
            pts = contour.astype(np.float32)

            # Douglas-Peucker simplification
            if dp_epsilon > 0:
                simplified = cv2.approxPolyDP(
                    pts.reshape(-1, 1, 2), dp_epsilon, closed=True)
                pts = simplified.reshape(-1, 2).astype(np.float64)
            else:
                pts = contour

            if len(pts) >= min_arc_verts:
                # Large enough polygon: apply corner arc fitting
                cmds = _polygon_to_curved_path(
                    pts, arc_radius=arc_radius,
                    min_corner_angle=min_corner_angle)
            elif len(pts) >= 2:
                # Small polygon: keep as L-only
                cmds = [('M', [(pts[0][0], pts[0][1])])]
                for p in pts[1:]:
                    cmds.append(('L', [(p[0], p[1])]))
                cmds.append(('Z', []))
            else:
                continue

            for cmd, _ in cmds:
                if cmd == 'A':
                    n_arcs += 1
                elif cmd == 'L':
                    n_lines += 1

            d_str = _commands_to_d(cmds, precision=4)
            color_d_parts[fill].append(d_str)
            total_subpaths += 1

    print(f"  {total_subpaths} subpaths: L={n_lines}, A={n_arcs}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # 6. Build SVG
    counts = np.bincount(label_img_new.ravel(), minlength=len(new_colors))
    order = np.argsort(-counts)
    bg_idx = order[0]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
    ]

    total_paths = 0
    for color_idx in order:
        r, g, b = new_colors[color_idx]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        rgb_color = f"rgb({r},{g},{b})"

        d_parts = color_d_parts.get(hex_color, [])
        if not d_parts:
            d_parts = color_d_parts.get(rgb_color, [])
        if not d_parts:
            continue

        compound_d = " ".join(d_parts)
        if color_idx == bg_idx:
            bg_d = f"M0,0 L{w},0 L{w},{h} L0,{h} Z"
            compound_d = bg_d + " " + compound_d
        svg_lines.append(
            f'<path fill="{hex_color}" fill-rule="evenodd" '
            f'd="{compound_d}"/>')
        total_paths += 1

    svg_lines.append('</svg>')

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))

    pre_size = os.path.getsize(output_path)
    print(f"\n  Pre-SVGO size: {pre_size/1024:.0f}KB")

    # 7. SVGO
    if svgo:
        print(f"\n[5] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
        except Exception as e:
            print(f"  SVGO failed: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {total_paths} paths, "
          f"{total_subpaths} subpaths, {len(new_colors)} colors")

    return output_path, None


def _reverse_commands(cmds):
    """Reverse a list of SVG path commands (for shared boundary other side).

    Converts M-...-Z going forward into M-...-Z going backward,
    preserving the exact same geometric boundary.
    """
    if len(cmds) < 2:
        return cmds

    # Strip M and Z
    inner = [c for c in cmds if c[0] not in ('M', 'Z')]
    if not inner:
        return cmds

    # Collect all points in order (including start from M)
    points = []
    m_cmd = cmds[0]
    if m_cmd[0] == 'M':
        points.append(m_cmd[1][0])

    for cmd, params in inner:
        if cmd == 'L':
            points.append(params[0])
        elif cmd == 'A':
            rx, ry, rot, la, sw, ex, ey = params[0]
            points.append((ex, ey))
        elif cmd == 'Q':
            points.append(params[1])  # endpoint
        elif cmd == 'C':
            points.append(params[2])  # endpoint

    # Reverse: start from last point, reverse each segment
    rev = [('M', [points[-1]])]
    for i in range(len(inner) - 1, -1, -1):
        cmd, params = inner[i]
        # The endpoint of the reversed segment is the start of the forward segment
        target = points[i]  # points[i] is the start point of inner[i]
        if cmd == 'L':
            rev.append(('L', [target]))
        elif cmd == 'A':
            rx, ry, rot, la, sw, ex, ey = params[0]
            # Reverse arc: flip sweep flag
            rev.append(('A', [(rx, ry, rot, la, 1 - sw,
                              target[0], target[1])]))
        elif cmd == 'Q':
            cp = params[0]
            rev.append(('Q', [cp, target]))
        elif cmd == 'C':
            cp1, cp2, ep = params[0], params[1], params[2]
            # Reverse cubic: swap control points
            rev.append(('C', [cp2, cp1, target]))
    rev.append(('Z', []))
    return rev


def png_to_svg_v9(input_path, output_path=None, n_colors=32,
                  min_cc_area=10, denoise='edge', svgo=True,
                  arc_tol=0.8, line_tol=0.5, angle_thresh=60.0):
    """PNG→SVG with bilateral shared-boundary curve fitting.

    Combines v6's multilabel-potrace shared-boundary tracing with bilateral
    curve fitting: each shared boundary is fitted ONCE and the same curves
    are used on both sides (reversed direction), preserving exact boundary
    sharing and preventing gaps.

    Pipeline:
      1. Edge-preserving denoise → two-stage color quantization
      2. Color-guarded CC merge
      3. multilabel-potrace polygon tracing (shared boundaries)
      4. Build edge graph → identify shared boundary segments
      5. Fit curves to each shared segment → apply to both sides
      6. Merge same-color paths into compound paths
      7. SVGO compression

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time
    import cv2
    import re

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v9.svg"

    # 1. Load
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Quantize
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma
    print(f"\n[1] Quantizing to {n_colors} colors (denoise={denoise})...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma, denoise=denoise)
    colors_flat = qarr.reshape(-1, 3)
    unique_colors = np.unique(colors_flat, axis=0)
    label_img = np.zeros((h, w), dtype=np.int32)
    for ci, c in enumerate(unique_colors):
        label_img[np.all(qarr == c, axis=2)] = ci
    print(f"  {len(unique_colors)} colors in {time.time()-t0:.1f}s")

    # 3. CC merge
    print(f"\n[2] Merging small components (min_area={min_cc_area})...")
    t0 = time.time()
    _merge_small_components(label_img, min_area=min_cc_area,
                           palette=unique_colors, max_color_dist=40)
    used_labels = np.unique(label_img)
    remap = np.zeros(len(unique_colors), dtype=np.int32)
    new_colors = []
    for new_idx, old_idx in enumerate(used_labels):
        remap[old_idx] = new_idx
        new_colors.append(unique_colors[old_idx])
    new_colors = np.array(new_colors, dtype=np.uint8)
    label_img_new = remap[label_img]
    print(f"  Done in {time.time()-t0:.1f}s")

    # 4. multilabel-potrace polygon tracing
    print(f"\n[3] multilabel-potrace polygon tracing...")
    t0 = time.time()
    try:
        import multilabel_potrace_svg as mlp
    except ImportError:
        return None, "multilabel-potrace not installed"

    n_labels = len(new_colors)
    comp_img = np.zeros((h, w), dtype=np.int32)
    comp_colors_list = []
    comp_id = 0
    for ci in range(n_labels):
        mask = (label_img_new == ci).astype(np.uint8)
        n_cc, cc_labels = cv2.connectedComponents(mask, connectivity=8)
        for cc in range(1, n_cc):
            comp_img[cc_labels == cc] = comp_id
            comp_colors_list.append(new_colors[ci])
            comp_id += 1
    print(f"  {comp_id} 8-connected components")

    comp_colors_arr = np.array(comp_colors_list, dtype=np.uint8)
    label_16 = np.ascontiguousarray(comp_img.astype(np.uint16))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        tmp_path = tmp.name

    mlp.multilabel_potrace_svg(
        label_16, tmp_path,
        straight_line_tol=0.0, smoothing=0.0, curve_fusion_tol=0.0,
        comp_colors=comp_colors_arr, line_width=0,
    )
    print(f"  mlp done in {time.time()-t0:.1f}s")

    # 5. Parse polygon paths
    print(f"\n[4] Bilateral curve fitting...")
    t0 = time.time()

    path_pattern = re.compile(r'<path d="([^"]+)" fill="([^"]+)"/>')
    with open(tmp_path) as f:
        mlp_svg = f.read()
    os.unlink(tmp_path)

    matches = path_pattern.findall(mlp_svg)

    # Parse each component's polygon vertices
    components = []
    for d, fill in matches:
        coords = re.findall(r'[-]?\d+', d)
        verts = [(int(coords[i]), int(coords[i + 1]))
                 for i in range(0, len(coords), 2)]
        components.append((verts, fill))

    print(f"  {len(components)} components parsed")

    # Build edge graph
    edge_to_comp = {}
    for ci, (verts, _) in enumerate(components):
        n = len(verts)
        for i in range(n):
            e = (verts[i], verts[(i + 1) % n])
            edge_to_comp[e] = ci

    # For each component, identify shared boundary segments and fit curves
    # A segment is a run of consecutive edges shared with the same neighbor
    from collections import defaultdict

    comp_commands = []  # per-component: list of (cmd, params)
    total_arcs = 0
    total_lines = 0
    total_quads = 0
    total_cubics = 0

    # Cache: (comp_i, comp_j, start_idx) → fitted commands
    # To avoid fitting the same shared segment twice
    fitted_cache = {}

    for ci, (verts, fill) in enumerate(components):
        n = len(verts)
        if n < 3:
            # Degenerate: just M...L...Z
            cmds = [('M', [verts[0]])]
            for v in verts[1:]:
                cmds.append(('L', [v]))
            cmds.append(('Z', []))
            comp_commands.append((cmds, fill))
            continue

        # Walk edges, group into segments (shared vs non-shared)
        segments = []  # [(is_shared, neighbor, vert_indices), ...]
        i = 0
        while i < n:
            e = (verts[i], verts[(i + 1) % n])
            rev = (e[1], e[0])
            if rev in edge_to_comp and edge_to_comp[rev] != ci:
                # Shared edge
                neighbor = edge_to_comp[rev]
                seg_start = i
                while i < n:
                    e2 = (verts[i], verts[(i + 1) % n])
                    rev2 = (e2[1], e2[0])
                    if rev2 in edge_to_comp and edge_to_comp[rev2] == neighbor:
                        i += 1
                    else:
                        break
                segments.append(('shared', neighbor, seg_start, i))
            else:
                # Non-shared edge (image border)
                seg_start = i
                while i < n:
                    e2 = (verts[i], verts[(i + 1) % n])
                    rev2 = (e2[1], e2[0])
                    is_shared = (rev2 in edge_to_comp
                                 and edge_to_comp[rev2] != ci)
                    if not is_shared:
                        i += 1
                    else:
                        break
                segments.append(('border', -1, seg_start, i))

        # Build path commands from segments
        cmds = [('M', [verts[0]])]
        for seg_type, neighbor, seg_start, seg_end in segments:
            # Extract vertices for this segment
            seg_verts = []
            for j in range(seg_start, seg_end + 1):
                seg_verts.append(verts[j % n])
            if len(seg_verts) < 2:
                continue

            pts = np.array(seg_verts, dtype=np.float64)

            if seg_type == 'shared' and len(pts) >= 5:
                # Check cache
                cache_key = (min(ci, neighbor), max(ci, neighbor),
                             verts[seg_start])
                if cache_key in fitted_cache:
                    # Use cached (possibly reversed)
                    cached_cmds, cached_ci = fitted_cache[cache_key]
                    if cached_ci == ci:
                        for c in cached_cmds:
                            cmds.append(c)
                    else:
                        # Reverse the cached commands
                        rev_cmds = _reverse_commands(
                            [('M', [pts[0]])] + cached_cmds + [('Z', [])])
                        # Skip M and Z from reversed
                        for c in rev_cmds:
                            if c[0] not in ('M', 'Z'):
                                cmds.append(c)
                else:
                    # Fit curves to this segment
                    seg_cmds = _fit_contour_curves(
                        pts, arc_tol=arc_tol, line_tol=line_tol,
                        angle_thresh=angle_thresh,
                        min_fit_pts=5, window=5)
                    # Extract inner commands (skip M and Z)
                    inner = [c for c in seg_cmds if c[0] not in ('M', 'Z')]
                    fitted_cache[cache_key] = (inner, ci)
                    for c in inner:
                        cmds.append(c)
            else:
                # Non-shared or short segment: keep as L
                for p in pts[1:]:
                    cmds.append(('L', [(p[0], p[1])]))

        cmds.append(('Z', []))

        # Count commands
        for cmd, _ in cmds:
            if cmd == 'A':
                total_arcs += 1
            elif cmd == 'L':
                total_lines += 1
            elif cmd == 'Q':
                total_quads += 1
            elif cmd == 'C':
                total_cubics += 1

        comp_commands.append((cmds, fill))

    print(f"  L={total_lines}, A={total_arcs}, "
          f"Q={total_quads}, C={total_cubics}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # 6. Merge same-color paths
    counts = np.bincount(label_img_new.ravel(), minlength=len(new_colors))
    order = np.argsort(-counts)
    bg_idx = order[0]

    color_d_parts = defaultdict(list)
    for cmds, fill in comp_commands:
        d_str = _commands_to_d(cmds, precision=4)
        color_d_parts[fill].append(d_str)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
    ]

    total_paths = 0
    total_subpaths = sum(len(v) for v in color_d_parts.values())
    for color_idx in order:
        r, g, b = new_colors[color_idx]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        rgb_color = f"rgb({r},{g},{b})"

        d_parts = color_d_parts.get(hex_color, [])
        if not d_parts:
            d_parts = color_d_parts.get(rgb_color, [])
        if not d_parts:
            continue

        compound_d = " ".join(d_parts)
        if color_idx == bg_idx:
            bg_d = f"M0,0 L{w},0 L{w},{h} L0,{h} Z"
            compound_d = bg_d + " " + compound_d
        svg_lines.append(
            f'<path fill="{hex_color}" fill-rule="evenodd" '
            f'd="{compound_d}"/>')
        total_paths += 1

    svg_lines.append('</svg>')

    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))

    pre_size = os.path.getsize(output_path)
    print(f"\n  Pre-SVGO size: {pre_size/1024:.0f}KB")

    # 7. SVGO
    if svgo:
        print(f"\n[5] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
        except Exception as e:
            print(f"  SVGO failed: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {total_paths} paths, "
          f"{total_subpaths} subpaths, {len(new_colors)} colors")

    return output_path, None


def png_to_svg_v7(input_path, output_path=None, n_colors=32,
                  min_cc_area=10, denoise='edge', svgo=True,
                  arc_tol=0.5, line_tol=0.3):
    """SciPrism-style PNG→SVG: per-contour curve fitting (L/A/Q/C commands).

    Reverse-engineered from SciPrism's actual SVG output:
      - Each contour = separate path (not one compound path per color)
      - Smart curve fitting: L (straight), A (arc), Q (quadratic), C (cubic)
      - Sub-pixel precision (4 decimal places)
      - fill-rule="evenodd" with compound paths for shapes with holes
      - Background rect, shapes stacked largest-first

    Pipeline:
      1. Edge-preserving denoise → two-stage color quantization
      2. Color-guarded CC merge (removes low-contrast noise only)
      3. Per-color 2x-upscale contour tracing (RETR_CCOMP for holes)
      4. Corner detection → per-segment curve fitting (line/arc/bezier)
      5. SVGO compression

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time
    import cv2

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v7.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Edge-preserving denoise + two-stage quantization
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma
    print(f"\n[1] Quantizing to {n_colors} colors "
          f"({n_achroma} achromatic + {n_chroma} chromatic, "
          f"denoise={denoise})...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma,
                                         denoise=denoise)

    colors_flat = qarr.reshape(-1, 3)
    unique_colors = np.unique(colors_flat, axis=0)
    label_img = np.zeros((h, w), dtype=np.int32)
    for ci, c in enumerate(unique_colors):
        label_img[np.all(qarr == c, axis=2)] = ci
    print(f"  {len(unique_colors)} colors in {time.time()-t0:.1f}s")

    # 3. Color-guarded CC merge
    print(f"\n[2] Merging small components (min_area={min_cc_area})...")
    t0 = time.time()
    _merge_small_components(label_img, min_area=min_cc_area,
                           palette=unique_colors,
                           max_color_dist=40)

    used_labels = np.unique(label_img)
    remap = np.zeros(len(unique_colors), dtype=np.int32)
    new_colors = []
    for new_idx, old_idx in enumerate(used_labels):
        remap[old_idx] = new_idx
        new_colors.append(unique_colors[old_idx])
    new_colors = np.array(new_colors, dtype=np.uint8)
    label_img_new = remap[label_img]
    print(f"  Done in {time.time()-t0:.1f}s")

    # Count pixels per color (for stacking order)
    counts = np.bincount(label_img_new.ravel(), minlength=len(new_colors))
    order = np.argsort(-counts)  # largest area first

    # 4. Per-contour curve fitting (SciPrism-style)
    print(f"\n[3] Per-contour curve fitting (L/A/Q/C)...")
    t0 = time.time()

    bg_idx = order[0]
    bg_color = f"#{new_colors[bg_idx][0]:02x}" \
               f"{new_colors[bg_idx][1]:02x}{new_colors[bg_idx][2]:02x}"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{bg_color}"/>',
    ]

    total_paths = 0
    total_subpaths = 0

    for color_idx in order:
        hex_color = f"#{new_colors[color_idx][0]:02x}" \
                    f"{new_colors[color_idx][1]:02x}" \
                    f"{new_colors[color_idx][2]:02x}"
        pixel_count = counts[color_idx]
        if pixel_count < 4:
            continue
        if color_idx == bg_idx:
            continue

        # Binary mask for this color
        mask = (label_img_new == color_idx).astype(np.uint8) * 255

        # 2x upscale + 0.5px dilation for boundary coverage
        mask_2x = cv2.resize(mask, (w * 2, h * 2),
                             interpolation=cv2.INTER_NEAREST)
        mask_2x = cv2.dilate(mask_2x, np.ones((2, 2), np.uint8),
                             iterations=1)

        # Find contours with hierarchy (CCOMP: outer + holes)
        contours, hierarchy = cv2.findContours(
            mask_2x, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        # Group outer contours with their holes
        idx = 0
        while idx >= 0:
            # Outer contour
            outer_pts = contours[idx].reshape(-1, 2).astype(float) / 2.0
            if len(outer_pts) < 3:
                idx = hierarchy[idx][0]
                continue

            # Collect hole contours
            hole_pts_list = []
            child_idx = hierarchy[idx][2]
            while child_idx >= 0:
                hpts = contours[child_idx].reshape(-1, 2).astype(float) / 2.0
                if len(hpts) >= 3:
                    hole_pts_list.append(hpts)
                child_idx = hierarchy[child_idx][0]

            # Fit curves to outer contour (no DP — keep pixel precision)
            outer_cmds = _contour_to_commands(outer_pts, precision=4,
                                              arc_tol=arc_tol,
                                              line_tol=line_tol,
                                              dp_epsilon=0)
            if not outer_cmds:
                idx = hierarchy[idx][0]
                continue

            # If no holes, single path
            if not hole_pts_list:
                d = _commands_to_d(outer_cmds, precision=4)
                svg_lines.append(
                    f'<path fill="{hex_color}" fill-rule="evenodd" d="{d}"/>')
                total_paths += 1
                total_subpaths += 1
            else:
                # Compound path with holes (evenodd)
                all_cmds = list(outer_cmds)
                for hpts in hole_pts_list:
                    hole_cmds = _contour_to_commands(hpts, precision=4,
                                                     arc_tol=arc_tol,
                                                     line_tol=line_tol,
                                                     dp_epsilon=0)
                    if hole_cmds:
                        all_cmds.extend(hole_cmds)
                        total_subpaths += 1
                d = _commands_to_d(all_cmds, precision=4)
                svg_lines.append(
                    f'<path fill="{hex_color}" fill-rule="evenodd" d="{d}"/>')
                total_paths += 1
                total_subpaths += 1  # outer

            idx = hierarchy[idx][0]

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    print(f"  {total_paths} paths, {total_subpaths} subpaths "
          f"in {time.time()-t0:.1f}s")

    # 5. Write output
    with open(output_path, 'w') as f:
        f.write(svg_content)

    pre_size = os.path.getsize(output_path)
    print(f"\n  Pre-SVGO size: {pre_size/1024:.0f}KB")

    # 6. SVGO compression
    if svgo:
        print(f"\n[4] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed: {result.stderr[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  SVGO skipped: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {total_paths} paths, "
          f"{total_subpaths} subpaths, {len(new_colors)} colors")

    return output_path, None


def png_to_svg_v5(input_path, output_path=None, n_colors=32,
                  dp_epsilon=1.0, min_contour_area=4, denoise='edge',
                  svgo=True):
    """High-quality PNG→SVG via OpenCV contour tracing + curve fitting.

    Pipeline (SciPrism-inspired):
      1. Two-stage quantization (achromatic/chromatic split for color diversity)
      2. Per-color binary mask → 2x upscale → 0.5px dilation for overlap
      3. OpenCV findContours(RETR_CCOMP) for outer contours + holes
      4. Douglas-Peucker simplification + corner detection + curve fitting
         (line → arc → quadratic → cubic bezier)
      5. Stacked rendering order (largest area first) — overlap hides
         anti-aliasing gaps at color boundaries
      6. Evenodd compound paths
      7. SVGO compression

    The 2x upscale trick: OpenCV contours trace through pixel positions,
    not pixel edges.  Upscaling the mask 2x means contours trace between
    sub-pixels, which maps to pixel edges when divided by 2.  Combined
    with 0.5px dilation (2x2 kernel in 2x space), each color region
    slightly overlaps its neighbors, eliminating white/wrong-color
    anti-aliasing artifacts at boundaries.

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time
    import cv2

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v5.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Two-stage quantization
    n_achroma = max(4, n_colors * 3 // 8)
    n_chroma = n_colors - n_achroma
    print(f"\n[1] Quantizing to {n_colors} colors "
          f"({n_achroma} achromatic + {n_chroma} chromatic)...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma,
                                         denoise=denoise)

    colors_flat = qarr.reshape(-1, 3)
    unique, counts = np.unique(colors_flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    print(f"  {len(unique)} unique colors in {time.time()-t0:.1f}s "
          f"(denoise={denoise})")

    # 3. Per-color contour tracing + curve fitting
    print(f"\n[2] Contour tracing + curve fitting...")
    t0 = time.time()
    bg_color = f"#{unique[0][0]:02x}{unique[0][1]:02x}{unique[0][2]:02x}"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{bg_color}"/>',
    ]

    # 2x2 dilation kernel in 2x space = 0.5px expansion in original
    kernel_2x = np.ones((2, 2), np.uint8)

    total_subpaths = 0
    path_count = 0
    skipped_small = 0

    # Colors sorted by pixel count (largest first) — stacking order
    for i, (color, count) in enumerate(zip(unique, counts)):
        if count < 20:
            continue

        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        if hex_color == bg_color:
            continue

        # Binary mask → 2x upscale → dilate for overlap
        mask = np.all(qarr == color, axis=2).astype(np.uint8) * 255
        mask_2x = cv2.resize(mask, (w * 2, h * 2),
                             interpolation=cv2.INTER_NEAREST)
        mask_2x = cv2.dilate(mask_2x, kernel_2x, iterations=1)

        # Find contours with hierarchy (RETR_CCOMP: outer + holes)
        contours, hierarchy = cv2.findContours(
            mask_2x, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or hierarchy is None:
            continue

        hierarchy = hierarchy[0]

        # Build compound path: outer contours + their holes
        subpaths_d = []
        idx = 0
        while idx >= 0:
            cnt = contours[idx]
            area = cv2.contourArea(cnt)

            if area >= min_contour_area:
                # Outer contour (points in 2x space)
                d = _contour_pts_to_d(cnt.reshape(-1, 2).astype(np.float64),
                                      dp_epsilon=dp_epsilon)
                if d:
                    subpaths_d.append(d)

                # Process holes (children of this outer contour)
                child_idx = hierarchy[idx][2]
                while child_idx >= 0:
                    child_cnt = contours[child_idx]
                    if cv2.contourArea(child_cnt) >= min_contour_area:
                        d = _contour_pts_to_d(
                            child_cnt.reshape(-1, 2).astype(np.float64),
                            dp_epsilon=dp_epsilon)
                        if d:
                            subpaths_d.append(d)
                    else:
                        skipped_small += 1
                    child_idx = hierarchy[child_idx][0]
            else:
                skipped_small += 1

            idx = hierarchy[idx][0]

        if subpaths_d:
            d = " ".join(subpaths_d)
            svg_lines.append(
                f'<path fill="{hex_color}" fill-rule="evenodd" d="{d}"/>'
            )
            path_count += 1
            total_subpaths += len(subpaths_d)

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    print(f"  {path_count} paths, {total_subpaths} subpaths, "
          f"{skipped_small} small contours skipped "
          f"in {time.time()-t0:.1f}s")

    # 4. Write output
    with open(output_path, 'w') as f:
        f.write(svg_content)

    pre_size = os.path.getsize(output_path)
    print(f"\n  Pre-SVGO size: {pre_size/1024:.0f}KB")

    # 5. SVGO compression
    if svgo:
        print(f"\n[3] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed: {result.stderr[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  SVGO skipped: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {path_count} paths, "
          f"{total_subpaths} subpaths, {len(unique)} colors")

    return output_path, None


def png_to_svg_v4(input_path, output_path=None, n_colors=32,
                  dp_tol=0.3, arc_radius=0.5,
                  svgo=True):
    """High-quality PNG→SVG via Shapely pixel-boundary tracing with arc corners.

    Pipeline (inspired by SciPrism):
      1. Bilateral denoise (remove JPEG artifacts, preserve edges)
      2. Two-stage quantization (separate achromatic/chromatic for color
         diversity — avoids wasting palette slots on near-white variations)
      3. Per-color Shapely polygon tracing (exact pixel-edge boundaries)
      4. Douglas-Peucker simplification (collapse redundant straight runs)
      5. Arc-fitted corners (tiny arcs at 90° corners for smooth zoom)
      6. Evenodd compound paths (one <path> per color)
      7. SVGO compression

    Typical results: 0.97–0.98 SSIM, 10–15 MB (5–7 MB after SVGO,
    <1 MB gzipped as .svgz).

    Args:
        input_path:     Path to input PNG/JPEG image.
        output_path:    Output SVG path (default: <input>_v4.svg).
        n_colors:       Total number of quantized colors (default 32).
        dp_tol:         Douglas-Peucker simplification tolerance.
        arc_radius:     Corner arc radius in pixels (0.5 = half pixel).
        svgo:           Run SVGO multipass compression (default True).

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v4.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Two-stage quantization (denoise + chroma-aware palette)
    n_achroma = max(4, n_colors * 3 // 8)   # ~37% for achromatic
    n_chroma = n_colors - n_achroma          # ~63% for chromatic
    print(f"\n[1] Quantizing to {n_colors} colors "
          f"({n_achroma} achromatic + {n_chroma} chromatic)...")
    t0 = time.time()
    qarr, palette = _two_stage_quantize(arr, n_achroma=n_achroma,
                                         n_chroma=n_chroma)

    colors_flat = qarr.reshape(-1, 3)
    unique, counts = np.unique(colors_flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    print(f"  {len(unique)} unique colors in {time.time()-t0:.1f}s")

    # 3. Per-color Shapely polygon tracing with arc corners
    print(f"\n[2] Tracing pixel boundaries...")
    t0 = time.time()
    bg_color = f"#{unique[0][0]:02x}{unique[0][1]:02x}{unique[0][2]:02x}"

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="{bg_color}"/>',
    ]

    total_subpaths = 0
    path_count = 0

    for i, (color, count) in enumerate(zip(unique, counts)):
        if count < 50:
            continue

        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        if hex_color == bg_color:
            continue

        # Binary mask for this color
        mask = np.all(qarr == color, axis=2)

        # Shapely polygon from pixel boxes
        polygon = _pixels_to_polygon(mask)
        if polygon is None:
            continue

        if polygon.is_empty:
            continue

        # Convert to SVG path with DP simplification + arc corners
        d = _polygon_to_compound_path(polygon, dp_tol=dp_tol,
                                       radius=arc_radius)
        if d:
            svg_lines.append(
                f'<path fill="{hex_color}" fill-rule="evenodd" d="{d}"/>'
            )
            path_count += 1
            total_subpaths += d.count('M')

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    print(f"  {path_count} compound paths, {total_subpaths} subpaths "
          f"in {time.time()-t0:.1f}s")

    # 4. Write output
    with open(output_path, 'w') as f:
        f.write(svg_content)

    pre_size = os.path.getsize(output_path)

    # 5. SVGO compression
    if svgo:
        print(f"\n[3] SVGO compression...")
        try:
            result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_size) * 100
                print(f"  {pre_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed: {result.stderr[:200]}")
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  SVGO skipped: {e}")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {path_count} paths, "
          f"{len(unique)} colors")

    return output_path, None


def png_to_svg_v3(input_path, output_path=None, n_colors=32,
                  svgo=True):
    """Pixel-perfect PNG→SVG: quantize → scanline rects → compound paths + crispEdges.

    Achieves near-perfect SSIM (~0.99) by rendering each pixel as an exact
    rectangle with shape-rendering="crispEdges". Colors are grouped into
    compound paths for compact output.

    Pipeline:
      1. Quantize to n_colors (k-means)
      2. For each color, extract horizontal scanline runs
      3. Group into compound paths (one per color, background first)
      4. Output SVG with shape-rendering="crispEdges"
      5. Optionally compress with SVGO (~25% reduction)

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v3.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Quantize colors
    print(f"\n[1] Quantizing to {n_colors} colors...")
    t0 = time.time()
    from sklearn.cluster import MiniBatchKMeans
    pixels = arr.reshape(-1, 3).astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=42,
                             batch_size=2000, n_init=3)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.predict(pixels).reshape(h, w)
    print(f"  Quantized in {time.time()-t0:.1f}s")

    # 3. Build scanline rects per color
    print(f"\n[2] Building scanline rects...")
    t0 = time.time()

    color_counts = np.bincount(labels.ravel(), minlength=n_colors)
    color_order = np.argsort(-color_counts)  # background (most pixels) first

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'shape-rendering="crispEdges">'
    ]

    total_runs = 0
    n_colors_used = 0

    for ci in color_order:
        if color_counts[ci] < 1:
            continue

        color = centers[ci]
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

        d_parts = []
        for y in range(h):
            row = labels[y]
            x = 0
            while x < w:
                if row[x] == ci:
                    xs = x
                    while x < w and row[x] == ci:
                        x += 1
                    d_parts.append(f'M{xs} {y}h{x - xs}v1h-{x - xs}Z')
                    total_runs += 1
                else:
                    x += 1

        if d_parts:
            svg_lines.append(
                f'  <path fill="{hex_color}" d="{" ".join(d_parts)}"/>'
            )
            n_colors_used += 1

    svg_lines.append('</svg>')

    print(f"  {total_runs} runs, {n_colors_used} colors in {time.time()-t0:.1f}s")

    # 4. Write SVG
    print(f"\n[3] Writing SVG...")
    with open(output_path, 'w') as f:
        f.write('\n'.join(svg_lines))

    pre_svgo_size = os.path.getsize(output_path)
    print(f"  Pre-SVGO: {pre_svgo_size/1024:.0f}KB")

    # 5. SVGO compression
    if svgo:
        print(f"\n[4] SVGO compression...")
        try:
            svgo_result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if svgo_result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_svgo_size) * 100
                print(f"  {pre_svgo_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed, keeping uncompressed output")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  SVGO not available, keeping uncompressed output")

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {n_colors_used} colors, {total_runs} runs")

    return output_path, None


def png_to_svg_v2(input_path, output_path=None, n_colors=32,
                  precision=1, min_area=4, arc_tol=0.6, line_tol=0.5,
                  svgo=True):
    """High-quality PNG→SVG: quantize → vtracer spline → arc fitting → color merge → SVGO.

    Pipeline:
      1. Color quantize to n_colors (k-means) — reduces palette, cleaner boundaries
      2. Trace quantized image with vtracer spline mode (smooth curves for arc fitting)
      3. Parse paths, filter tiny paths
      4. Geometric primitive fitting: cubics → lines/arcs/quadratics
      5. Merge remaining similar colors
      6. Output minimal SVG, optionally compress with SVGO

    Returns:
        (output_path, None) on success, (None, error) on failure.
    """
    import time

    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_v2.svg"

    # 1. Load image
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"Image: {w}x{h}")

    # 2. Optional quantization — pre-quantize for smaller output, skip for max quality
    quant_path = None
    trace_input = input_path
    if n_colors > 0:
        print(f"\n[1] Color quantization to {n_colors} colors...")
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            quant_path = tmp.name
        quant_result, err = quantize_colors(input_path, n_colors=n_colors,
                                             output_path=quant_path)
        if err:
            return None, f"Quantization failed: {err}"
        trace_input = quant_path
        print(f"  Quantized in {time.time()-t0:.1f}s")
    else:
        print(f"\n[1] Skipping quantization (max quality mode)")

    # 3. Trace with vtracer spline mode
    #    speckle=0 critical for quality, gradient_step=1 for smooth gradients
    label = "quantized" if n_colors > 0 else "original"
    print(f"\n[2] vtracer trace (spline on {label}, cp=8)...")
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
        vt_path = tmp.name
    vt_result, err = run_vtracer(
        trace_input, output_path=vt_path,
        color_precision=8, path_precision=4,
        filter_speckle=0, segment_length=3.5,
        gradient_step=1, mode='spline',
    )
    if err:
        return None, f"vtracer failed: {err}"
    vt_size = os.path.getsize(vt_path)
    print(f"  Raw: {vt_size/1024:.0f}KB")

    # 4. Parse paths
    print(f"\n[3] Parsing paths...")
    with open(vt_path) as f:
        svg_content = f.read()
    paths, err = parse_svg_paths(svg_content)
    if err:
        return None, err
    print(f"  Parsed: {len(paths)} paths")

    # Filter tiny paths by d-string length
    min_dl = 40
    d_lens = [len(_commands_to_d(p.commands, precision=precision)) for p in paths]
    paths = [p for p, dl in zip(paths, d_lens) if dl >= min_dl]
    print(f"  Kept: {len(paths)} (after tiny filter)")

    # 5. Geometric primitive fitting — converts cubics to arcs/lines/quadratics
    print(f"\n[4] Arc fitting (post-processing)...")
    t0 = time.time()
    optimized, _ = optimize_all_paths(paths, line_tol=line_tol,
                                      arc_tol=arc_tol, quad_tol=0.4)
    print(f"  Optimized in {time.time()-t0:.1f}s")

    # Command distribution
    cmd_stats = defaultdict(int)
    for p in optimized:
        for cmd, _ in p.commands:
            cmd_stats[cmd] += 1
    total = sum(cmd_stats.values())
    print(f"  Commands ({total} total):")
    for cmd in ['M', 'L', 'A', 'Q', 'C', 'Z']:
        if cmd_stats[cmd] > 0:
            print(f"    {cmd}: {cmd_stats[cmd]:6d} ({cmd_stats[cmd]/total*100:.1f}%)")

    # 6. Merge remaining similar colors
    print(f"\n[5] Merging colors...")
    color_groups = defaultdict(list)
    for p in optimized:
        color_groups[p.fill].append(p)
    initial_colors = len(color_groups)

    # When no pre-quantization, merge to 256 to keep quality high
    merge_target = n_colors if n_colors > 0 else 256
    merge_threshold = 8 if n_colors > 0 else 4
    max_threshold = 50 if n_colors > 0 else 30
    while len(color_groups) > merge_target and merge_threshold < max_threshold:
        optimized, _ = merge_similar_colors(optimized, threshold=merge_threshold)
        color_groups = defaultdict(list)
        for p in optimized:
            color_groups[p.fill].append(p)
        merge_threshold += 2 if n_colors == 0 else 4

    unique_colors = len(color_groups)
    print(f"  {initial_colors} → {unique_colors} colors (threshold={merge_threshold-4 if n_colors > 0 else merge_threshold-2}), "
          f"{len(optimized)} paths")

    # 7. Assemble SVG
    print(f"\n[6] Assembling SVG...")

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]

    for p in optimized:
        d = _commands_to_d(p.commands, precision=precision)
        if d:
            svg_lines.append(
                f'  <path fill="{p.fill}" fill-rule="{p.fill_rule}" d="{d}"/>'
            )

    svg_lines.append('</svg>')
    svg_content = '\n'.join(svg_lines)

    with open(output_path, 'w') as f:
        f.write(svg_content)

    pre_svgo_size = os.path.getsize(output_path)

    # 8. SVGO compression (optional, ~55% size reduction with no quality loss)
    if svgo:
        print(f"\n[7] SVGO compression...")
        try:
            svgo_result = subprocess.run(
                ["npx", "svgo", output_path, "-o", output_path, "--multipass"],
                capture_output=True, text=True, timeout=120,
            )
            if svgo_result.returncode == 0:
                post_size = os.path.getsize(output_path)
                reduction = (1 - post_size / pre_svgo_size) * 100
                print(f"  {pre_svgo_size/1024:.0f}KB → {post_size/1024:.0f}KB "
                      f"({reduction:.0f}% reduction)")
            else:
                print(f"  SVGO failed, keeping uncompressed output")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  SVGO not available, keeping uncompressed output")

    # Cleanup temp files
    for tmp_file in [vt_path, quant_path]:
        if tmp_file:
            try:
                os.unlink(tmp_file)
            except OSError:
                pass

    out_size = os.path.getsize(output_path)
    print(f"\nOutput: {output_path}")
    print(f"Size: {out_size/1024:.0f}KB, {len(optimized)} paths, {unique_colors} colors")

    return output_path, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="High-precision PNG to SVG conversion")
    parser.add_argument("input", help="Input PNG image")
    parser.add_argument("-o", "--output", help="Output SVG path")
    parser.add_argument("--colors", type=int, default=32,
                        help="Number of colors for quantization (0=skip for max quality, default: 32)")
    parser.add_argument("--mode", choices=["pixel", "spline", "polygon", "semantic", "v2", "v3", "v4", "v5", "v6"],
                        default="v6",
                        help="Tracing mode: v6 (default, SciPrism-style hole-punching), "
                             "v4 (Shapely pixel boundaries + arcs), "
                             "v2 (vtracer + arc fitting), "
                             "semantic (smooth vectors + text), "
                             "v3 (pixel rects + crispEdges)")
    parser.add_argument("--color-precision", type=int, default=7,
                        choices=range(1, 9),
                        help="Color precision 1-8 bits (default: 7 for semantic, 8 for pixel)")
    parser.add_argument("--line-tol", type=float, default=0.3,
                        help="Line detection tolerance in pixels (default: 0.3)")
    parser.add_argument("--arc-tol", type=float, default=0.6,
                        help="Arc fitting tolerance in pixels (default: 0.6)")
    parser.add_argument("--quad-tol", type=float, default=0.4,
                        help="Quadratic curve tolerance in pixels (default: 0.4)")
    parser.add_argument("--precision", type=int, default=1,
                        help="Decimal precision in SVG coordinates (default: 1)")
    parser.add_argument("--speckle", type=int, default=0,
                        help="Filter speckle size in pixels (default: 0)")
    parser.add_argument("--min-path-dlen", type=int, default=60,
                        help="Min path d-string length to keep (default: 60)")
    parser.add_argument("--quality", choices=["fast", "balanced", "high", "max"],
                        default="balanced",
                        help="Quality preset for semantic mode (default: balanced). "
                             "fast: ~1MB/0.94 SSIM, balanced: ~2MB/0.96, "
                             "high: ~5MB/0.97, max: ~11MB/0.98")
    parser.add_argument("--upscale", type=float, default=None,
                        help="Upscale factor before tracing (default: from quality preset)")
    parser.add_argument("--no-text", action="store_true",
                        help="Disable text detection in semantic mode")
    parser.add_argument("--no-svgo", action="store_true",
                        help="Disable SVGO optimization in semantic mode")
    parser.add_argument("--merge-threshold", type=int, default=0,
                        help="Color merge threshold (0=off, default: 0)")
    parser.add_argument("--ssim", action="store_true",
                        help="Compute and display SSIM score after conversion (requires resvg)")
    parser.add_argument("--renderer", choices=["resvg", "chrome"],
                        default="resvg",
                        help="SVG renderer for SSIM (default: resvg)")

    args = parser.parse_args()

    if args.mode == "v6":
        result, err = png_to_svg_v6(
            args.input,
            output_path=args.output,
            n_colors=args.colors,
            svgo=not args.no_svgo,
        )
    elif args.mode == "v5":
        result, err = png_to_svg_v5(
            args.input,
            output_path=args.output,
            n_colors=args.colors,
            svgo=not args.no_svgo,
        )
    elif args.mode == "v4":
        result, err = png_to_svg_v4(
            args.input,
            output_path=args.output,
            n_colors=args.colors,
            svgo=not args.no_svgo,
        )
    elif args.mode == "v3":
        result, err = png_to_svg_v3(
            args.input,
            output_path=args.output,
            n_colors=args.colors if args.colors > 0 else 32,
            svgo=not args.no_svgo,
        )
    elif args.mode == "v2":
        result, err = png_to_svg_v2(
            args.input,
            output_path=args.output,
            n_colors=args.colors,
            precision=args.precision,
            arc_tol=args.arc_tol,
            line_tol=args.line_tol,
            svgo=not args.no_svgo,
        )
    elif args.mode == "semantic":
        result, err = png_to_svg_semantic(
            args.input,
            output_path=args.output,
            quality=args.quality,
            color_precision=args.color_precision if args.color_precision != 7 else None,
            upscale=args.upscale,
            min_path_dlen=args.min_path_dlen,
            svg_precision=args.precision if args.precision != 1 else None,
            merge_threshold=args.merge_threshold,
            include_text=not args.no_text,
            svgo=not args.no_svgo,
        )
    else:
        result, err = png_to_svg(
            args.input,
            output_path=args.output,
            n_colors=args.colors,
            line_tol=args.line_tol,
            arc_tol=args.arc_tol,
            quad_tol=args.quad_tol,
            vtracer_precision=args.precision,
            vtracer_speckle=args.speckle,
            color_precision=args.color_precision,
            mode=args.mode,
        )

    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"\nDone! Output: {result}")

    if args.ssim:
        print("\nComputing SSIM (via resvg)...")
        score, err = compute_ssim(args.input, result, renderer=args.renderer)
        if err:
            print(f"  Warning: {err}")
        else:
            print(f"  SSIM: {score:.4f}")
