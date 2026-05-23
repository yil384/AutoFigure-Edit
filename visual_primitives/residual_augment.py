"""Residual CV augmentation for pure-native draw.io reconstructions."""
from __future__ import annotations

import copy
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RESIDUAL_AUGMENT_VERSION = "residual-augment-0.1"


def augment_program_from_render_residual(
    program: dict[str, Any],
    source_image: str | Path,
    rendered_image: str | Path,
    *,
    max_paths: int = 80,
    max_shapes: int = 60,
    render_dilate: int = 4,
    min_area: int = 10,
    max_bbox_area: int = 2600,
    text_pad: float = 2.5,
    include_skeleton_paths: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add missing source foreground as native primitives.

    The source image is used only as detector evidence. The emitted program
    contains normal draw.io shapes and polylines, never raster cells.
    """
    source = _read_rgb(source_image)
    rendered = _read_rgb(rendered_image)
    h = min(source.shape[0], rendered.shape[0])
    w = min(source.shape[1], rendered.shape[1])
    source = source[:h, :w]
    rendered = rendered[:h, :w]

    source_fg = _foreground_mask(source)
    rendered_fg = _foreground_mask(rendered)
    if render_dilate > 0:
        k = int(max(1, render_dilate))
        rendered_fg = cv2.dilate(
            rendered_fg,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
    residual = cv2.bitwise_and(source_fg, cv2.bitwise_not(rendered_fg))
    _erase_program_text(residual, program, pad=text_pad)
    residual = cv2.morphologyEx(
        residual,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )

    candidates = _residual_candidates(
        source,
        residual,
        min_area=min_area,
        max_bbox_area=max_bbox_area,
        include_skeleton_paths=include_skeleton_paths,
    )
    candidates.sort(key=lambda item: (
        -float(item.get("score", 0.0)),
        item["bbox"][1],
        item["bbox"][0],
    ))

    updated = copy.deepcopy(program)
    primitives = updated.setdefault("primitives", [])
    existing_boxes = [
        _primitive_bbox(p)
        for p in primitives
        if p.get("type") in {"shape", "edge"} and _primitive_bbox(p)
    ]
    operations: list[dict[str, Any]] = []
    added_paths = 0
    added_shapes = 0

    for candidate in candidates:
        bbox = candidate["bbox"]
        if _covered_by_existing(bbox, existing_boxes):
            continue
        if candidate["kind"] == "shape" and added_shapes < max_shapes:
            primitive = {
                "id": _next_id(primitives, "res_shape"),
                "type": "shape",
                "role": "residual_symbol_or_mark",
                "shape": candidate["shape"],
                "bbox": _round_bbox(bbox),
                "style": {
                    "fill": candidate["color"],
                    "stroke": candidate["stroke"],
                },
                "source": "residual_shape_augment",
                "confidence": candidate["confidence"],
            }
            added_shapes += 1
        elif candidate["kind"] == "path" and added_paths < max_paths:
            primitive = {
                "id": _next_id(primitives, "res_edge"),
                "type": "edge",
                "role": "residual_icon_stroke",
                "bbox": _round_bbox(bbox),
                "path": _round_path(candidate["path"]),
                "style": {
                    "stroke": candidate["color"],
                    "stroke_width": candidate["stroke_width"],
                    "arrow_start": False,
                    "arrow_end": False,
                },
                "source": "residual_path_augment",
                "length": round(float(candidate["length"]), 3),
                "confidence": candidate["confidence"],
            }
            added_paths += 1
        else:
            continue
        primitives.append(primitive)
        existing_boxes.append(tuple(bbox))
        operations.append({
            "action": f"add_{candidate['kind']}",
            "primitive_id": primitive["id"],
            "bbox": primitive["bbox"],
            "color": candidate["color"],
            "score": round(float(candidate["score"]), 3),
        })
        if added_paths >= max_paths and added_shapes >= max_shapes:
            break

    _refresh_counts(updated)
    action_counts = Counter(op["action"] for op in operations)
    report = {
        "version": RESIDUAL_AUGMENT_VERSION,
        "source_image": str(source_image),
        "rendered_image": str(rendered_image),
        "config": {
            "max_paths": max_paths,
            "max_shapes": max_shapes,
            "render_dilate": render_dilate,
            "min_area": min_area,
            "max_bbox_area": max_bbox_area,
            "text_pad": text_pad,
            "include_skeleton_paths": include_skeleton_paths,
        },
        "counts": {
            "candidates": len(candidates),
            "operations": len(operations),
            **dict(sorted(action_counts.items())),
        },
        "operations": operations,
    }
    updated.setdefault("metadata", {})["residual_augment"] = {
        "version": RESIDUAL_AUGMENT_VERSION,
        "counts": report["counts"],
    }
    return updated, report


def _read_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    non_white = np.max(255.0 - arr, axis=2)
    dark = (luma < 150) & (non_white > 18)
    colored = (chroma > 28) & (luma < 242) & (non_white > 12)
    page_bg = (luma > 246) & (chroma < 9)
    pale_panel = (
        (luma > 184) &
        (chroma >= 4) &
        (chroma < 54) &
        (arr[:, :, 2] >= arr[:, :, 0] - 6)
    )
    return ((dark | colored) & ~page_bg & ~pale_panel).astype(np.uint8) * 255


def _erase_program_text(mask: np.ndarray, program: dict[str, Any], *, pad: float) -> None:
    h, w = mask.shape[:2]
    for primitive in program.get("primitives", []):
        if primitive.get("type") != "text" or not primitive.get("bbox"):
            continue
        x0, y0, x1, y1 = [float(v) for v in primitive["bbox"]]
        x0 = max(0, int(math.floor(x0 - pad)))
        y0 = max(0, int(math.floor(y0 - pad)))
        x1 = min(w, int(math.ceil(x1 + pad)))
        y1 = min(h, int(math.ceil(y1 + pad)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0


def _residual_candidates(
    source_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    min_area: int,
    max_bbox_area: int,
    include_skeleton_paths: bool,
) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict[str, Any]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = int(cv2.contourArea(contour))
        pixel_area = int(cv2.countNonZero(mask[y:y + bh, x:x + bw]))
        if pixel_area < min_area:
            continue
        bbox_area = int(max(1, bw * bh))
        if bw < 2 or bh < 2 or bbox_area > max_bbox_area:
            continue
        aspect = bw / max(1.0, float(bh))
        if aspect > 28 or aspect < 1 / 28:
            continue
        perim = float(cv2.arcLength(contour, True))
        if perim < 8 or perim > 420:
            continue
        bbox = [float(x), float(y), float(x + bw), float(y + bh)]
        color = _median_color(source_rgb, mask, x, y, bw, bh)
        fill_fraction = pixel_area / max(1.0, bbox_area)
        circularity = _circularity(contour)
        if include_skeleton_paths and _is_line_like_residual(
            bw, bh, pixel_area, bbox_area, fill_fraction
        ):
            out.extend(_skeleton_path_candidates(
                source_rgb,
                mask,
                x,
                y,
                bw,
                bh,
                color=color,
                pixel_area=pixel_area,
            ))
        if bw <= 24 and bh <= 24 and fill_fraction >= 0.28:
            shape = "ellipse" if circularity >= 0.52 else "rectangle"
            score = pixel_area + 0.3 * perim
            out.append({
                "kind": "shape",
                "shape": shape,
                "bbox": bbox,
                "color": color,
                "stroke": _stroke_for_fill(color),
                "confidence": round(min(0.92, 0.42 + fill_fraction), 3),
                "score": score,
            })
            continue
        approx = cv2.approxPolyDP(contour, 1.15, True)
        if len(approx) < 3 or len(approx) > 34:
            continue
        path = [[float(p[0][0]), float(p[0][1])] for p in approx]
        if path and path[0] != path[-1]:
            path.append(path[0])
        length = _path_length(path)
        if length < 10:
            continue
        out.append({
            "kind": "path",
            "bbox": bbox,
            "path": path,
            "length": length,
            "color": color,
            "stroke_width": 0.85 if pixel_area < 55 else 1.0,
            "confidence": round(min(0.9, 0.35 + min(0.5, pixel_area / 180.0)), 3),
            "score": pixel_area + 0.15 * length,
        })
    return _dedupe_candidates(out)


def _is_line_like_residual(
    width: int,
    height: int,
    pixel_area: int,
    bbox_area: int,
    fill_fraction: float,
) -> bool:
    if pixel_area < 10 or bbox_area < 18:
        return False
    aspect = max(width, height) / max(1.0, float(min(width, height)))
    if aspect >= 2.2 and pixel_area >= 12:
        return True
    return fill_fraction <= 0.34 and max(width, height) >= 10


def _skeleton_path_candidates(
    source_rgb: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    color: str,
    pixel_area: int,
) -> list[dict[str, Any]]:
    component = (mask[y:y + height, x:x + width] > 0).astype(np.uint8) * 255
    if cv2.countNonZero(component) < 10:
        return []
    skeleton = _morphological_skeleton(component)
    paths = _trace_skeleton_paths(skeleton, x_offset=x, y_offset=y)
    out: list[dict[str, Any]] = []
    for path in paths:
        path = _rdp_open_path(path, epsilon=1.35)
        if len(path) < 2:
            continue
        length = _path_length(path)
        if length < 9:
            continue
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        out.append({
            "kind": "path",
            "bbox": bbox,
            "path": path,
            "length": length,
            "color": color,
            "stroke_width": 0.9 if pixel_area < 80 else 1.05,
            "confidence": round(min(0.9, 0.44 + min(0.4, pixel_area / 240.0)), 3),
            "score": pixel_area + 0.62 * length + 55.0,
            "source": "residual_skeleton_path",
        })
    return out


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(work)
    skeleton = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iterations = int(max(work.shape[:2]) + 2)
    for _ in range(max_iterations):
        if cv2.countNonZero(work) <= 0:
            break
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        work = cv2.erode(work, element)
    return skeleton


def _trace_skeleton_paths(
    skeleton: np.ndarray,
    *,
    x_offset: int,
    y_offset: int,
) -> list[list[list[float]]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=8)
    paths: list[list[list[float]]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        ys, xs = np.nonzero(labels == label)
        coords = [(int(x), int(y)) for x, y in zip(xs, ys)]
        coord_set = set(coords)
        neighbor_map = {
            pt: [
                (pt[0] + dx, pt[1] + dy)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if not (dx == 0 and dy == 0)
                and (pt[0] + dx, pt[1] + dy) in coord_set
            ]
            for pt in coords
        }
        endpoints = [pt for pt, neighbors in neighbor_map.items()
                     if len(neighbors) <= 1]
        if len(endpoints) >= 2:
            start, target = _farthest_pair(endpoints)
            traced = _shortest_graph_path(neighbor_map, start, target)
            if traced:
                paths.append([
                    [float(px + x_offset), float(py + y_offset)]
                    for px, py in traced
                ])
            continue
        contour_paths = _closed_skeleton_contours(
            (labels == label).astype(np.uint8) * 255,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        paths.extend(contour_paths)
    return paths


def _farthest_pair(points: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(points) > 96:
        seed = points[0]
        first = max(points, key=lambda pt: _sq_dist(seed, pt))
        second = max(points, key=lambda pt: _sq_dist(first, pt))
        return first, second
    best = (points[0], points[-1])
    best_dist = -1
    for i, p0 in enumerate(points):
        for p1 in points[i + 1:]:
            dist = _sq_dist(p0, p1)
            if dist > best_dist:
                best = (p0, p1)
                best_dist = dist
    return best


def _sq_dist(p0: tuple[int, int], p1: tuple[int, int]) -> int:
    return (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2


def _shortest_graph_path(
    neighbor_map: dict[tuple[int, int], list[tuple[int, int]]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    queue = [start]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    for node in queue:
        if node == target:
            break
        for neighbor in neighbor_map.get(node, []):
            if neighbor in parent:
                continue
            parent[neighbor] = node
            queue.append(neighbor)
    if target not in parent:
        return []
    out = []
    node: tuple[int, int] | None = target
    while node is not None:
        out.append(node)
        node = parent[node]
    out.reverse()
    return out


def _closed_skeleton_contours(
    component: np.ndarray,
    *,
    x_offset: int,
    y_offset: int,
) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    paths: list[list[list[float]]] = []
    for contour in contours:
        if len(contour) < 5:
            continue
        path = [
            [float(point[0][0] + x_offset), float(point[0][1] + y_offset)]
            for point in contour
        ]
        if path and path[0] != path[-1]:
            path.append(path[0])
        paths.append(path)
    return paths


def _rdp_open_path(path: list[list[float]], epsilon: float) -> list[list[float]]:
    if len(path) <= 2:
        return path
    arr = np.asarray(path, dtype=np.float32).reshape((-1, 1, 2))
    closed = bool(path[0] == path[-1])
    approx = cv2.approxPolyDP(arr, epsilon, closed)
    points = [[float(p[0][0]), float(p[0][1])] for p in approx]
    if not closed:
        if points[0] != path[0]:
            points.insert(0, path[0])
        if points[-1] != path[-1]:
            points.append(path[-1])
    elif points and points[0] != points[-1]:
        points.append(points[0])
    return points


def _median_color(
    rgb: np.ndarray,
    mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> str:
    patch = rgb[y:y + h, x:x + w]
    m = mask[y:y + h, x:x + w] > 0
    pixels = patch[m]
    if len(pixels) == 0:
        return "#050505"
    med = np.median(pixels.astype(np.float32), axis=0)
    if float(np.mean(med)) > 210 and float(np.max(med) - np.min(med)) < 20:
        return "#050505"
    return "#" + "".join(f"{int(max(0, min(255, round(v)))):02x}" for v in med)


def _stroke_for_fill(color: str) -> str:
    if color.lower() in {"#050505", "#000000"}:
        return color
    try:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
    except Exception:
        return "#050505"
    r = int(max(0, r * 0.72))
    g = int(max(0, g * 0.72))
    b = int(max(0, b * 0.72))
    return f"#{r:02x}{g:02x}{b:02x}"


def _circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    perim = float(cv2.arcLength(contour, True))
    if perim <= 0:
        return 0.0
    return float(4.0 * math.pi * area / (perim * perim))


def _dedupe_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(items, key=lambda item: -float(item.get("score", 0.0)))
    kept: list[dict[str, Any]] = []
    for item in ranked:
        if any(_iou(item["bbox"], other["bbox"]) > 0.55 for other in kept):
            continue
        kept.append(item)
    return kept


def _covered_by_existing(bbox: list[float], boxes: list[tuple[float, float, float, float]]) -> bool:
    for other in boxes:
        if _iou(bbox, other) > 0.62:
            return True
        if _overlap_fraction(bbox, other) > 0.74:
            return True
    return False


def _primitive_bbox(primitive: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = primitive.get("bbox")
    if bbox:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return (x0, y0, x1, y1)
    path = primitive.get("path")
    if path:
        xs = [float(p[0]) for p in path]
        ys = [float(p[1]) for p in path]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = _area(a) + _area(b) - inter
    return inter / max(1.0, union)


def _overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return ((ix1 - ix0) * (iy1 - iy0)) / max(1.0, _area(a))


def _area(bbox) -> float:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _path_length(path: list[list[float]]) -> float:
    total = 0.0
    for p0, p1 in zip(path[:-1], path[1:]):
        total += math.hypot(float(p1[0]) - float(p0[0]),
                            float(p1[1]) - float(p0[1]))
    return total


def _next_id(primitives: list[dict[str, Any]], prefix: str) -> str:
    existing = {p.get("id") for p in primitives}
    i = 1
    while f"{prefix}_{i:04d}" in existing:
        i += 1
    return f"{prefix}_{i:04d}"


def _round_bbox(bbox) -> list[float]:
    return [round(float(v), 3) for v in bbox]


def _round_path(path) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in path]


def _refresh_counts(program: dict[str, Any]) -> None:
    primitives = program.get("primitives", [])
    program["counts"] = {
        "regions": sum(1 for p in primitives if p.get("type") == "region"),
        "texts": sum(1 for p in primitives if p.get("type") == "text"),
        "edges": sum(1 for p in primitives if p.get("type") == "edge"),
        "shapes": sum(1 for p in primitives if p.get("type") == "shape"),
        "total": len(primitives),
    }
