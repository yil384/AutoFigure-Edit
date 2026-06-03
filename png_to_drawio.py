"""Direct PNG/JPG -> drawio primitive converter.

This is intentionally independent of the PNG->SVG vectorizer.  It can create a
drawio file in either legacy overlay mode or strict pure-native mode:

  - legacy overlay mode: a locked source-image backing layer plus editable
    primitive layers;
  - pure-native mode: no source image, no raster icon crops, no foreground
    tiles, and no embedded images/base64/stencils; only draw.io-native text,
    rectangles, simple shapes, and edges are emitted;
  - a JSON primitive ledger so later passes can reason over boxes/points.

Use --pure-native for the non-cheating png->drawio framework path.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _xml_escape(value) -> str:
    s = str(value)
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;'))


def _rgb_to_hex(rgb) -> str:
    r, g, b = [int(max(0, min(255, round(v)))) for v in rgb]
    return f'#{r:02x}{g:02x}{b:02x}'


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = (value or '').strip().lstrip('#')
    if len(value) != 6:
        return (0, 0, 0)
    try:
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)


def _darken_hex(rgb, factor: float = 0.72) -> str:
    return _rgb_to_hex([float(v) * factor for v in rgb])


def _png_svg_image_style(img: Image.Image) -> str:
    """Embed a PIL image in a drawio image cell.

    draw.io's style parser splits on semicolons, so the usual
    data:image/png;base64 form is unsafe.  diagrams.net accepts
    data:image/png,{base64} without the ;base64 marker, which avoids an SVG
    wrapper and preserves the raster pixels more directly.
    """
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return (f'shape=image;html=1;imageAspect=0;aspect=fixed;'
            f'image=data:image/png,{b64};')


def _transparent_foreground_crop(crop: Image.Image) -> Image.Image:
    rgba = np.array(crop.convert('RGBA'))
    arr = rgba[:, :, :3].astype(np.float32)
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    alpha = np.where((luma < 245) | (chroma > 10), 255, 0).astype(np.uint8)
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, 'RGBA')


def _emit_image_cell(cid: int, img: Image.Image, bbox: tuple[float, float, float, float],
                     parent: str = '1', locked: bool = False) -> str:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    style = _png_svg_image_style(img)
    if locked:
        style += ('locked=1;movable=0;editable=0;deletable=0;'
                  'resizable=0;rotatable=0;connectable=0;')
    return (
        f'<mxCell id="{cid}" value="" style="{style}" '
        f'vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x0:.2f}" y="{y0:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}" as="geometry"/>'
        f'</mxCell>'
    )


def _emit_canvas_anchor_cell(cid: int, W: int, H: int,
                             parent: str = '1') -> str:
    """Invisible full-canvas cell so draw.io export keeps page bounds."""
    style = (
        'rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=none;'
        'locked=1;movable=0;editable=0;deletable=0;resizable=0;'
        'rotatable=0;connectable=0;'
    )
    return (
        f'<mxCell id="{cid}" value="" style="{style}" vertex="1" '
        f'parent="{parent}">'
        f'<mxGeometry x="0" y="0" width="{W}" height="{H}" '
        f'as="geometry"/>'
        f'</mxCell>'
    )


def _emit_rect_cell(cid: int, rect: dict, parent: str = '2') -> str:
    x0, y0, x1, y1 = rect['bbox']
    w = x1 - x0
    h = y1 - y0
    fill = rect.get('fill', '#eaf2f7')
    stroke = rect.get('stroke', '#8f9cac')
    style = (
        'rounded=1;whiteSpace=wrap;html=1;arcSize=8;'
        f'fillColor={fill};strokeColor={stroke};strokeWidth=0.8;'
    )
    return (
        f'<mxCell id="{cid}" value="" style="{style}" '
        f'vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x0:.1f}" y="{y0:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>'
    )


def _emit_native_shape_cell(cid: int, shape: dict, parent: str = '7') -> str:
    x0, y0, x1, y1 = shape['bbox']
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    fill = shape.get('fill', '#d8e6ef')
    stroke = shape.get('stroke', '#6f8190')
    shape_name = shape.get('shape')
    if shape_name == 'ellipse':
        base = 'ellipse;'
    elif shape_name == 'triangle':
        base = 'triangle;'
    elif shape_name == 'rhombus':
        base = 'rhombus;'
    elif shape_name == 'hexagon':
        base = 'hexagon;'
    else:
        base = 'rounded=0;'
    if shape.get('direction'):
        base += f'direction={shape["direction"]};'
    style = (
        f'{base}whiteSpace=wrap;html=1;'
        f'fillColor={fill};strokeColor={stroke};strokeWidth=0.7;'
    )
    return (
        f'<mxCell id="{cid}" value="" style="{style}" '
        f'vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x0:.1f}" y="{y0:.1f}" '
        f'width="{w:.1f}" height="{h:.1f}" as="geometry"/>'
        f'</mxCell>'
    )


def _emit_text_cell(cid: int, item: dict, parent: str = '4',
                    font_family: str = 'Arial') -> str:
    x0, y0, x1, y1 = item['bbox']
    w = max(2.0, x1 - x0)
    h = max(2.0, y1 - y0)
    fs = item.get('font_size', 9)
    text = _xml_escape(item.get('text', ''))
    if '\n' in text:
        text = text.replace('\n', '&lt;br&gt;')
    align = item.get('align', 'center')
    style_parts = [
        'text', 'html=1', 'strokeColor=none', 'fillColor=none',
        f'align={align}', 'verticalAlign=middle', 'whiteSpace=wrap',
        'rounded=0', f'fontFamily={font_family}', f'fontSize={fs}',
        'fontColor=#050505',
    ]
    if item.get('rotation') is not None:
        style_parts.append(f'rotation={float(item["rotation"]):.1f}')
    if item.get('bold'):
        style_parts.append('fontStyle=1')
    style = ';'.join(style_parts) + ';'
    return (
        f'<mxCell id="{cid}" value="{text}" style="{style}" '
        f'vertex="1" parent="{parent}">'
        f'<mxGeometry x="{x0 - 1:.1f}" y="{y0 - 1:.1f}" '
        f'width="{w + 2:.1f}" height="{h + 2:.1f}" as="geometry"/>'
        f'</mxCell>'
    )


def _line_path(line: dict) -> list[tuple[float, float]]:
    if 'path' in line:
        return [(float(x), float(y)) for x, y in line['path']]
    x0, y0, x1, y1 = line['points']
    return [(float(x0), float(y0)), (float(x1), float(y1))]


def _emit_edge_cell(cid: int, line: dict, parent: str = '3') -> str:
    pts = _line_path(line)
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    is_freeform = line.get('orient') == 'P' or len(pts) > 2
    parts = [
        'rounded=0', 'html=1',
        f'strokeColor={line.get("stroke", "#050505")}',
        f'strokeWidth={line.get("width", 1.2):.2f}',
    ]
    if line.get('dashed'):
        parts.extend(['dashed=1', 'dashPattern=4 3'])
    if is_freeform:
        parts.extend(['edgeStyle=none', 'orthogonalLoop=0'])
    else:
        parts.extend(['edgeStyle=orthogonalEdgeStyle', 'orthogonalLoop=1',
                      'jettySize=auto'])
    parts.append('startArrow=classic;startFill=1'
                 if line.get('arrow_start') else 'startArrow=none')
    parts.append('endArrow=classic;endFill=1'
                 if line.get('arrow_end') else 'endArrow=none')
    style = ';'.join(parts) + ';'
    waypoints = ''
    if len(pts) > 2:
        inner = ''.join(
            f'<mxPoint x="{x:.1f}" y="{y:.1f}"/>'
            for x, y in pts[1:-1])
        waypoints = f'<Array as="points">{inner}</Array>'
    return (
        f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="{parent}">'
        f'<mxGeometry relative="1" as="geometry">'
        f'<mxPoint x="{x0:.1f}" y="{y0:.1f}" as="sourcePoint"/>'
        f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="targetPoint"/>'
        f'{waypoints}'
        f'</mxGeometry></mxCell>'
    )


def _emit_crop_cell(cid: int, crop: dict, parent: str = '5') -> str:
    return _emit_image_cell(cid, crop['image'], crop['bbox'], parent=parent,
                            locked=False)


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = (ax1 - ax0) * (ay1 - ay0)
    ba = (bx1 - bx0) * (by1 - by0)
    return inter / max(1.0, aa + ba - inter)


def _bbox_overlap_fraction(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    return inter / aa


def _bbox_intersection_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def _path_bbox(path: list[tuple[float, float]], pad: float = 0.0):
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _path_length(path: list[tuple[float, float]]) -> float:
    return float(sum(
        math.hypot(path[i + 1][0] - path[i][0],
                   path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)))


def _rdp_path(path: list[tuple[float, float]],
              epsilon: float = 1.8) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    arr = np.asarray(path, dtype=np.float32).reshape((-1, 1, 2))
    approx = cv2.approxPolyDP(arr, epsilon, False)
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    if pts[0] != path[0]:
        pts.insert(0, path[0])
    if pts[-1] != path[-1]:
        pts.append(path[-1])
    return pts


def _detect_ocr_text_single(img: Image.Image, conf_threshold: float = 45.0,
                            scale: float = 2.0, psm: int = 11) -> list[dict]:
    try:
        import pytesseract
    except Exception:
        return []

    if scale != 1.0:
        up = img.resize((int(img.width * scale), int(img.height * scale)),
                        Image.Resampling.BICUBIC)
    else:
        up = img
    data = pytesseract.image_to_data(
        up, output_type=pytesseract.Output.DICT, config=f'--psm {psm}')

    words = []
    for i, raw in enumerate(data.get('text', [])):
        text = (raw or '').strip()
        if not text:
            continue
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1.0
        if conf < conf_threshold:
            continue
        if len(text) == 1 and text in '|=+_-/\\(){}[]:;.,':
            continue
        x = data['left'][i] / scale
        y = data['top'][i] / scale
        w = data['width'][i] / scale
        h = data['height'][i] / scale
        if w < 1 or h < 2:
            continue
        words.append({
            'text': text, 'bbox': (x, y, x + w, y + h),
            'conf': conf, 'height': h,
        })

    if not words:
        return []

    words.sort(key=lambda t: ((t['bbox'][1] + t['bbox'][3]) / 2, t['bbox'][0]))
    lines: list[list[dict]] = []
    for word in words:
        cy = (word['bbox'][1] + word['bbox'][3]) / 2
        h = word['height']
        placed = False
        for line in lines:
            lcy = np.mean([(w['bbox'][1] + w['bbox'][3]) / 2 for w in line])
            lh = np.median([w['height'] for w in line])
            if abs(cy - lcy) <= max(4.0, 0.65 * max(h, lh)):
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    blocks = []
    for line in lines:
        line.sort(key=lambda t: t['bbox'][0])
        heights = [w['height'] for w in line]
        med_h = float(np.median(heights))
        gap_threshold = max(7.0, med_h * 1.08)
        group = [line[0]]
        groups = []
        for word in line[1:]:
            prev = group[-1]
            gap = word['bbox'][0] - prev['bbox'][2]
            if gap > gap_threshold:
                groups.append(group)
                group = [word]
            else:
                group.append(word)
        groups.append(group)

        for group in groups:
            text = ' '.join(w['text'] for w in group)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                continue
            x0 = min(w['bbox'][0] for w in group)
            y0 = min(w['bbox'][1] for w in group)
            x1 = max(w['bbox'][2] for w in group)
            y1 = max(w['bbox'][3] for w in group)
            word_h = float(np.median([w['height'] for w in group]))
            block_h = float(y1 - y0)
            if text.startswith('Figure '):
                fs = max(11, min(18, int(round(word_h * 0.82))))
                align = 'left'
            else:
                fs = max(5, min(12, int(round(word_h * 0.76))))
                if block_h > word_h * 1.75:
                    fs = min(fs, max(8, int(round(word_h * 0.68))))
                align = 'center'
            bold = (
                text.startswith('Figure ') or
                (fs >= 9 and len(text) > 2) or
                (fs >= 8 and any(ch.isupper() for ch in text) and len(text) > 2)
            )
            blocks.append({
                'kind': 'text', 'text': text, 'bbox': (x0, y0, x1, y1),
                'font_size': fs, 'conf': float(np.mean([w['conf'] for w in group])),
                'bold': bold, 'align': align,
            })
    return blocks


def _is_plausible_ocr_block(item: dict) -> bool:
    text = item.get('text', '').strip()
    if not text:
        return False
    x0, y0, x1, y1 = item['bbox']
    w = x1 - x0
    h = y1 - y0
    cleaned = re.sub(r'\s+', '', text)
    if not cleaned:
        return False
    if any(ch in text for ch in '\\‘’`~<>'):
        return False
    if len(cleaned) <= 1 and not cleaned.isdigit():
        return False
    if w > 760 or h > 90:
        return False
    if w > 260 and not (
        text.startswith('Figure ') or
        'Ranked Candidates' in text or
        'Application-Aware' in text or
        'Continuous-Variable' in text
    ):
        return False
    alnum = sum(ch.isalnum() for ch in cleaned)
    if alnum / max(1, len(cleaned)) < 0.38:
        return False
    return True


def _merge_ocr_blocks(candidates: list[dict]) -> list[dict]:
    candidates = [
        c for c in candidates
        if _is_plausible_ocr_block(c)
    ]
    candidates.sort(key=lambda c: (
        -float(c.get('conf', 0.0)),
        (c['bbox'][2] - c['bbox'][0]) * (c['bbox'][3] - c['bbox'][1]),
    ))
    merged: list[dict] = []
    for cand in candidates:
        cb = cand['bbox']
        replacement = None
        skip = False
        for i, old in enumerate(merged):
            ob = old['bbox']
            if _bbox_iou(cb, ob) > 0.48:
                replacement = i
                break
            if (_bbox_overlap_fraction(cb, ob) > 0.72 or
                    _bbox_overlap_fraction(ob, cb) > 0.72):
                if float(old.get('conf', 0.0)) >= float(cand.get('conf', 0.0)):
                    skip = True
                else:
                    replacement = i
                break
        if skip:
            continue
        if replacement is not None:
            merged[replacement] = cand
        else:
            merged.append(cand)
    merged.sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
    return merged


def detect_ocr_text(img: Image.Image, conf_threshold: float = 45.0,
                    scale: float = 2.0, psm: int = 11,
                    multipass: bool = False) -> list[dict]:
    if not multipass:
        return _detect_ocr_text_single(
            img, conf_threshold=conf_threshold, scale=scale, psm=psm)
    passes = [
        (psm, scale),
        (11, max(scale, 2.0)),
        (12, max(scale, 2.0)),
        (6, max(scale, 2.0)),
        (11, max(scale, 3.0)),
    ]
    seen = set()
    candidates = []
    for pass_psm, pass_scale in passes:
        key = (pass_psm, float(pass_scale))
        if key in seen:
            continue
        seen.add(key)
        candidates.extend(_detect_ocr_text_single(
            img, conf_threshold=conf_threshold,
            scale=pass_scale, psm=pass_psm))
    return _merge_ocr_blocks(candidates)


def _is_trusted_text(item: dict, min_conf: float = 70.0) -> bool:
    text = item.get('text', '').strip()
    if not text:
        return False
    x0, y0, x1, y1 = item['bbox']
    w = x1 - x0
    h = y1 - y0
    conf = float(item.get('conf', 0.0))
    if text.startswith('Figure ') and conf >= 88:
        return True
    if conf < min_conf:
        return False
    cleaned = re.sub(r'\s+', '', text)
    alnum = sum(ch.isalnum() for ch in cleaned)
    if alnum == 0:
        return False
    if alnum / max(1, len(cleaned)) < 0.45:
        return False
    if any(ch in text for ch in '\\‘’`~<>'):
        return False
    if text in {'Bit', 'alll'}:
        return False
    if h > 30:
        return False
    if h > 22 and conf < 86:
        return False
    if len(cleaned) <= 2 and not re.fullmatch(r'\d+(?:-\d+)?', cleaned):
        return False
    if len(cleaned) <= 4 and h > 24 and conf < 90:
        return False
    if w > 170 and not (text.startswith('Figure ') or '&' in text or '-' in text):
        return False
    return True


def _is_geometry_mask_text(item: dict, min_conf: float = 70.0) -> bool:
    """Select OCR boxes that are reliable enough to erase before geometry detection."""
    text = item.get('text', '').strip()
    if not text:
        return False
    x0, y0, x1, y1 = item['bbox']
    h = y1 - y0
    conf = float(item.get('conf', 0.0))
    if text.startswith('Figure ') and conf >= 88:
        return True
    if conf < min_conf:
        return False
    if h > 30:
        return False
    cleaned = re.sub(r'\s+', '', text)
    if len(cleaned) <= 1 and not cleaned.isdigit():
        return False
    if any(ch in text for ch in '\\‘’`~<>'):
        return False
    return True


def detect_pale_rectangles(rgb: np.ndarray,
                           max_area_ratio: float = 0.38,
                           min_fill_ratio: float = 0.58,
                           min_contour_fill_ratio: float = 0.50) -> list[dict]:
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    # Pale tinted fills used by academic diagrams.  The blue-ish constraint
    # avoids selecting ordinary black text anti-aliasing.
    mask = ((luma > 175) & (chroma >= 4) & (chroma <= 80) &
            (arr[:, :, 2] >= arr[:, :, 0] - 8))
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 18 or bh < 14 or area < 450:
            continue
        if area > max_area_ratio * w * h:
            continue
        contour_fill = cv2.contourArea(cnt) / max(1.0, float(area))
        if contour_fill < min_contour_fill_ratio:
            continue
        roi_mask = mask[y:y + bh, x:x + bw] > 0
        if roi_mask.mean() < min_fill_ratio:
            continue
        roi = rgb[y:y + bh, x:x + bw]
        samples = roi[roi_mask]
        if len(samples) == 0:
            continue
        fill = _rgb_to_hex(np.median(samples, axis=0))
        rects.append({
            'kind': 'rect',
            'bbox': (float(x), float(y), float(x + bw), float(y + bh)),
            'fill': fill,
            'stroke': '#8f9cac',
            'area': float(area),
        })

    rects.sort(key=lambda r: r['area'], reverse=True)
    deduped = []
    for rect in rects:
        if any(_bbox_iou(rect['bbox'], old['bbox']) > 0.88 for old in deduped):
            continue
        deduped.append(rect)
    return deduped


def detect_separated_pale_rectangles(rgb: np.ndarray,
                                     max_area_ratio: float = 0.38) -> list[dict]:
    """Recover pale panels that were over-merged by close morphology."""
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    mask = ((luma > 175) & (chroma >= 4) & (chroma <= 80) &
            (arr[:, :, 2] >= arr[:, :, 0] - 8))
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 24 or bh < 18 or area < 2500:
            continue
        if area > max_area_ratio * w * h:
            continue
        contour_fill = cv2.contourArea(cnt) / max(1.0, float(area))
        if contour_fill < 0.62:
            continue
        roi_mask = mask[y:y + bh, x:x + bw] > 0
        if roi_mask.mean() < 0.68:
            continue
        roi = rgb[y:y + bh, x:x + bw]
        samples = roi[roi_mask]
        if len(samples) == 0:
            continue
        rects.append({
            'kind': 'rect',
            'bbox': (float(x), float(y), float(x + bw), float(y + bh)),
            'fill': _rgb_to_hex(np.median(samples, axis=0)),
            'stroke': '#8f9cac',
            'area': float(area),
            'source': 'split_fill',
        })

    rects.sort(key=lambda r: r['area'], reverse=True)
    deduped = []
    for rect in rects:
        if any(_bbox_iou(rect['bbox'], old['bbox']) > 0.84 for old in deduped):
            continue
        deduped.append(rect)
    return deduped


def detect_border_rectangles(rgb: np.ndarray,
                             max_area_ratio: float = 0.14) -> list[dict]:
    """Detect rectangular containers from their border geometry.

    This complements fill-color detection. Scientific diagrams often use very
    pale panel fills whose chroma is close to the page background, so relying
    only on filled masks misses large rounded containers.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 42, 135)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 24 or bh < 18 or area < 500:
            continue
        if area > max_area_ratio * w * h:
            continue
        ratio = bw / max(1.0, bh)
        if ratio < 0.22 or ratio > 7.0:
            continue
        perim = cv2.arcLength(cnt, True)
        if perim <= 0:
            continue
        approx = cv2.approxPolyDP(cnt, 0.025 * perim, True)
        if len(approx) > 18:
            continue
        contour_area = cv2.contourArea(cnt)
        rectangularity = contour_area / max(1.0, float(area))
        if rectangularity < 0.58:
            continue
        roi_edges = edges[y:y + bh, x:x + bw] > 0
        band = max(2, min(4, int(round(min(bw, bh) * 0.08))))
        perimeter_samples = np.concatenate([
            roi_edges[:band, :].reshape(-1),
            roi_edges[max(0, bh - band):, :].reshape(-1),
            roi_edges[:, :band].reshape(-1),
            roi_edges[:, max(0, bw - band):].reshape(-1),
        ])
        perimeter_coverage = float(perimeter_samples.mean()) if len(perimeter_samples) else 0.0
        if perimeter_coverage < 0.10:
            continue
        roi = rgb[y:y + bh, x:x + bw].astype(np.float32)
        if roi.size == 0:
            continue
        inset = max(2, min(8, int(min(bw, bh) * 0.08)))
        inner = roi[inset:max(inset + 1, bh - inset),
                    inset:max(inset + 1, bw - inset)]
        if inner.size == 0:
            inner = roi
        luma = (0.299 * inner[:, :, 0] + 0.587 * inner[:, :, 1] +
                0.114 * inner[:, :, 2])
        chroma = inner.max(axis=2) - inner.min(axis=2)
        fill_pixels = inner[(luma > 145) & (chroma < 95)]
        if len(fill_pixels) < max(25, 0.12 * inner.shape[0] * inner.shape[1]):
            continue
        fill = _rgb_to_hex(np.median(fill_pixels, axis=0))
        rects.append({
            'kind': 'rect',
            'bbox': (float(x), float(y), float(x + bw), float(y + bh)),
            'fill': fill,
            'stroke': '#8f9cac',
            'area': float(area),
            'source': 'border',
            'perimeter_coverage': perimeter_coverage,
        })

    rects.sort(key=lambda r: r['area'], reverse=True)
    deduped = []
    for rect in rects:
        if any(_bbox_iou(rect['bbox'], old['bbox']) > 0.82 for old in deduped):
            continue
        deduped.append(rect)
    return deduped


def merge_rectangles(*groups: list[dict]) -> list[dict]:
    rects = [r for group in groups for r in group]
    rects.sort(key=lambda r: r.get('area', 0.0), reverse=True)
    deduped = []
    for rect in rects:
        if any(_bbox_iou(rect['bbox'], old['bbox']) > 0.78 for old in deduped):
            continue
        deduped.append(rect)
    deduped.sort(key=lambda r: (r['bbox'][1], r['bbox'][0],
                                -(r['bbox'][2] - r['bbox'][0]) *
                                (r['bbox'][3] - r['bbox'][1])))
    return deduped


def filter_fill_rectangles(fill_rects: list[dict],
                           border_rects: list[dict]) -> list[dict]:
    filtered = []
    for rect in fill_rects:
        x0, y0, x1, y1 = rect['bbox']
        bw = x1 - x0
        bh = y1 - y0
        area = max(1.0, bw * bh)
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        contained_borders = [
            b for b in border_rects
            if _bbox_overlap_fraction(b['bbox'], rect['bbox']) >= 0.88 and
            _bbox_overlap_fraction(rect['bbox'], b['bbox']) <= 0.82
        ]
        contained_area = sum(
            (b['bbox'][2] - b['bbox'][0]) * (b['bbox'][3] - b['bbox'][1])
            for b in contained_borders
        )
        if len(contained_borders) >= 2 and contained_area >= 0.16 * area:
            continue
        if area >= 25000 and aspect >= 2.25 and bh > bw:
            continue
        filtered.append(rect)
    return filtered


def filter_border_rectangles(rects: list[dict],
                             text_blocks: list[dict]) -> list[dict]:
    filtered = []
    for rect in rects:
        x0, y0, x1, y1 = rect['bbox']
        area = max(1.0, (x1 - x0) * (y1 - y0))
        coverage = float(rect.get('perimeter_coverage', 1.0))
        text_area = sum(
            _bbox_intersection_area(rect['bbox'], text['bbox'])
            for text in text_blocks
        )
        text_frac = text_area / area
        if area < 2500 and text_frac > 0.16:
            continue
        if area < 2500 and coverage < 0.48:
            continue
        if area < 8000 and coverage < 0.32:
            continue
        filtered.append(rect)
    return filtered


def _erase_bboxes(mask: np.ndarray, items: list[dict], pad: int = 2) -> None:
    h, w = mask.shape[:2]
    for item in items:
        x0, y0, x1, y1 = item['bbox']
        x0 = max(0, int(math.floor(x0)) - pad)
        y0 = max(0, int(math.floor(y0)) - pad)
        x1 = min(w, int(math.ceil(x1)) + pad)
        y1 = min(h, int(math.ceil(y1)) + pad)
        mask[y0:y1, x0:x1] = 0


def _bbox_touches_text(bbox, text_blocks: list[dict],
                       threshold: float = 0.35) -> bool:
    return any(_bbox_overlap_fraction(bbox, t['bbox']) > threshold
               for t in text_blocks)


def _endpoint_arrow_score(dark: np.ndarray,
                          tip: tuple[float, float],
                          inside: tuple[float, float],
                          radius: int = 9) -> float:
    tx, ty = tip
    ix, iy = inside
    ux = ix - tx
    uy = iy - ty
    norm = math.hypot(ux, uy)
    if norm < 3.0:
        return 0.0
    ux /= norm
    uy /= norm
    vx = -uy
    vy = ux

    h, w = dark.shape[:2]
    cx = int(round(tx))
    cy = int(round(ty))
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    ys, xs = np.nonzero(dark[y0:y1, x0:x1])
    if len(xs) < 8:
        return 0.0
    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0
    rx = xs - tx
    ry = ys - ty
    along = rx * ux + ry * uy
    lateral = rx * vx + ry * vy

    inward = ((along >= -0.75) & (along <= radius + 0.75) &
              (np.abs(lateral) <= radius))
    outward = ((along < -0.75) & (along >= -radius) &
               (np.abs(lateral) <= radius))
    if int(inward.sum()) < 8:
        return 0.0
    if int(outward.sum()) > max(5, int(inward.sum()) // 2):
        return 0.0

    near = inward & (along <= 2.8)
    body = inward & (along > 1.5) & (along <= radius)
    side = body & (np.abs(lateral) >= 1.35)
    side_pos = side & (lateral > 1.35)
    side_neg = side & (lateral < -1.35)
    side_count = int(side.sum())
    if side_count < 6 or int(side_pos.sum()) < 2 or int(side_neg.sum()) < 2:
        return 0.0

    body_lat = lateral[body]
    near_lat = lateral[near]
    if len(body_lat) < 6 or len(near_lat) < 2:
        return 0.0
    body_width = float(np.percentile(body_lat, 90) -
                       np.percentile(body_lat, 10))
    near_width = float(np.percentile(near_lat, 90) -
                       np.percentile(near_lat, 10))
    if body_width < 4.0 or body_width < near_width + 1.1:
        return 0.0

    side_along = along[side]
    mean_along = float(side_along.mean()) if len(side_along) else 0.0
    if mean_along < 1.6 or mean_along > radius - 0.4:
        return 0.0

    balance = min(int(side_pos.sum()), int(side_neg.sum())) / max(1, side_count)
    expansion = min(1.0, (body_width - near_width) / 5.0)
    density = min(1.0, side_count / 18.0)
    return float(0.45 * expansion + 0.35 * density + 0.20 * balance)


def _apply_directional_arrowheads(rgb: np.ndarray, text_blocks: list[dict],
                                  lines: list[dict],
                                  threshold: float = 0.70,
                                  min_length: float = 24.0) -> None:
    """Infer arrowheads only when the endpoint shape widens like a triangle."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = ((gray < 130).astype(np.uint8) * 255)
    _erase_bboxes(dark, text_blocks, pad=3)
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    dark_bool = dark > 0
    for line in lines:
        pts = _line_path(line)
        line['arrow_start'] = False
        line['arrow_end'] = False
        if len(pts) < 2:
            continue
        if _path_length(pts) < min_length:
            continue
        start_score = _endpoint_arrow_score(dark_bool, pts[0], pts[1])
        end_score = _endpoint_arrow_score(dark_bool, pts[-1], pts[-2])
        line['arrow_start'] = start_score >= threshold
        line['arrow_end'] = end_score >= threshold
        if start_score or end_score:
            line['arrow_scores'] = {
                'start': round(start_score, 3),
                'end': round(end_score, 3),
            }


def _estimate_line_stroke(rgb: np.ndarray,
                          path: list[tuple[float, float]],
                          sample_radius: int = 2) -> str:
    if len(path) < 2:
        return '#050505'
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for p0, p1 in zip(path[:-1], path[1:]):
        x0, y0 = p0
        x1, y1 = p1
        cv2.line(mask,
                 (int(round(max(0, min(w - 1, x0)))),
                  int(round(max(0, min(h - 1, y0))))),
                 (int(round(max(0, min(w - 1, x1)))),
                  int(round(max(0, min(h - 1, y1))))),
                 255, max(1, sample_radius * 2 + 1))
    pixels = rgb[mask > 0].astype(np.float32)
    if len(pixels) == 0:
        return '#050505'
    luma = (0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] +
            0.114 * pixels[:, 2])
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    ink = pixels[(luma < 230) | (chroma > 22)]
    if len(ink) < 4:
        return '#050505'

    # Ignore very pale antialias halos; preserve saturated connector colors.
    luma = (0.299 * ink[:, 0] + 0.587 * ink[:, 1] +
            0.114 * ink[:, 2])
    chroma = ink.max(axis=1) - ink.min(axis=1)
    if float(np.percentile(chroma, 75)) < 18:
        dark = ink[luma <= np.percentile(luma, 55)]
        color = np.median(dark if len(dark) else ink, axis=0)
    else:
        colored = ink[chroma >= np.percentile(chroma, 50)]
        color = np.median(colored if len(colored) else ink, axis=0)
    color_luma = (0.299 * color[0] + 0.587 * color[1] +
                  0.114 * color[2])
    color_chroma = float(np.max(color) - np.min(color))
    if ((color_luma < 105 and color_chroma < 60) or
            (color_luma < 190 and color_chroma < 30)):
        return '#050505'
    return _rgb_to_hex(color)


def _merge_axis_lines(raw: list[dict], gap: float = 6.0,
                      cross_tol: float = 3.0) -> list[dict]:
    merged = []
    for orient in ('H', 'V'):
        lines = [l for l in raw if l['orient'] == orient]
        if orient == 'H':
            lines.sort(key=lambda l: (round(l['points'][1] / cross_tol), l['points'][0]))
        else:
            lines.sort(key=lambda l: (round(l['points'][0] / cross_tol), l['points'][1]))
        groups: list[list[dict]] = []
        for line in lines:
            placed = False
            for group in groups:
                g0 = group[-1]
                if orient == 'H':
                    cy = line['points'][1]
                    gcy = np.mean([g['points'][1] for g in group])
                    starts_after = line['points'][0] <= max(g['points'][2] for g in group) + gap
                    if abs(cy - gcy) <= cross_tol and starts_after:
                        group.append(line)
                        placed = True
                        break
                else:
                    cx = line['points'][0]
                    gcx = np.mean([g['points'][0] for g in group])
                    starts_after = line['points'][1] <= max(g['points'][3] for g in group) + gap
                    if abs(cx - gcx) <= cross_tol and starts_after:
                        group.append(line)
                        placed = True
                        break
            if not placed:
                groups.append([line])
        for group in groups:
            if orient == 'H':
                x0 = min(g['points'][0] for g in group)
                x1 = max(g['points'][2] for g in group)
                y = float(np.mean([g['points'][1] for g in group]))
                if x1 - x0 >= 10:
                    merged.append({'kind': 'line', 'orient': 'H',
                                   'points': (x0, y, x1, y), 'width': 1.2})
            else:
                x = float(np.mean([g['points'][0] for g in group]))
                y0 = min(g['points'][1] for g in group)
                y1 = max(g['points'][3] for g in group)
                if y1 - y0 >= 10:
                    merged.append({'kind': 'line', 'orient': 'V',
                                   'points': (x, y0, x, y1), 'width': 1.2})
    return merged


def detect_lines(rgb: np.ndarray, text_blocks: list[dict],
                 detect_arrows: bool = False) -> list[dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 95).astype(np.uint8) * 255
    h, w = dark.shape[:2]
    _erase_bboxes(dark, text_blocks, pad=3)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 13))
    hmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
    vmask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
    hmask = cv2.morphologyEx(
        hmask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    vmask = cv2.morphologyEx(
        vmask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))

    raw = []

    def collect(mask: np.ndarray, orient: str) -> None:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue
            pixels = int(cv2.countNonZero(mask[y:y + bh, x:x + bw]))
            density = pixels / max(1.0, float(bw * bh))
            if density < 0.12:
                continue
            if orient == 'H':
                if bw < 12 or bh > 7 or bw / max(1.0, bh) < 3.2:
                    continue
                bbox = (x, max(0, y - 2), x + bw, min(h, y + bh + 2))
                if _bbox_touches_text(bbox, text_blocks, threshold=0.45):
                    continue
                cy = y + bh / 2.0
                raw.append({'orient': 'H',
                            'points': (float(x), float(cy),
                                       float(x + bw), float(cy))})
            else:
                if bh < 12 or bw > 7 or bh / max(1.0, bw) < 3.2:
                    continue
                bbox = (max(0, x - 2), y, min(w, x + bw + 2), y + bh)
                if _bbox_touches_text(bbox, text_blocks, threshold=0.45):
                    continue
                cx = x + bw / 2.0
                raw.append({'orient': 'V',
                            'points': (float(cx), float(y),
                                       float(cx), float(y + bh))})

    collect(hmask, 'H')
    collect(vmask, 'V')
    lines = _merge_axis_lines(raw, gap=4.0, cross_tol=2.5)
    for line in lines:
        line['arrow_start'] = False
        line['arrow_end'] = False
    return lines


def _neighbors8(pt: tuple[int, int], skel: np.ndarray):
    y, x = pt
    h, w = skel.shape
    for yy in range(max(0, y - 1), min(h, y + 2)):
        for xx in range(max(0, x - 1), min(w, x + 2)):
            if yy == y and xx == x:
                continue
            if skel[yy, xx]:
                yield yy, xx


def _is_near_axis_line(path: list[tuple[float, float]],
                       axis_lines: list[dict]) -> bool:
    bbox = _path_bbox(path, pad=3)
    for line in axis_lines:
        lb = _path_bbox(_line_path(line), pad=4)
        if _bbox_iou(bbox, lb) > 0.35:
            return True
    return False


def _erase_lines(mask: np.ndarray, lines: list[dict], thickness: int = 3) -> None:
    h, w = mask.shape[:2]
    for line in lines:
        pts = _line_path(line)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            x0, y0 = p0
            x1, y1 = p1
            cv2.line(mask, (int(round(max(0, min(w - 1, x0)))),
                            int(round(max(0, min(h - 1, y0))))),
                     (int(round(max(0, min(w - 1, x1)))),
                     int(round(max(0, min(h - 1, y1))))),
                     0, thickness)


def _triangle_direction(approx: np.ndarray) -> str | None:
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) != 3:
        return None
    xs = pts[:, 0]
    ys = pts[:, 1]
    x_span = float(max(1.0, xs.max() - xs.min()))
    y_span = float(max(1.0, ys.max() - ys.min()))
    y_tol = max(1.2, y_span * 0.22)
    x_tol = max(1.2, x_span * 0.22)
    if int(np.sum(ys <= ys.min() + y_tol)) == 1:
        return 'north'
    if int(np.sum(ys >= ys.max() - y_tol)) == 1:
        return 'south'
    if int(np.sum(xs <= xs.min() + x_tol)) == 1:
        return 'west'
    if int(np.sum(xs >= xs.max() - x_tol)) == 1:
        return 'east'
    return None


def _is_rhombus_like(approx: np.ndarray, bw: int, bh: int) -> bool:
    pts = approx.reshape(-1, 2).astype(np.float32)
    if len(pts) != 4 or bw <= 0 or bh <= 0:
        return False
    xs = pts[:, 0] / float(max(1, bw))
    ys = pts[:, 1] / float(max(1, bh))
    top = np.any((ys < 0.30) & (xs > 0.22) & (xs < 0.78))
    bottom = np.any((ys > 0.70) & (xs > 0.22) & (xs < 0.78))
    left = np.any((xs < 0.30) & (ys > 0.22) & (ys < 0.78))
    right = np.any((xs > 0.70) & (ys > 0.22) & (ys < 0.78))
    return int(top) + int(bottom) + int(left) + int(right) >= 3


def _classify_native_contour(cnt: np.ndarray, bw: int, bh: int,
                             pix_area: int, fill_ratio: float,
                             circularity: float, perim: float,
                             color_std: float) -> tuple[str | None,
                                                        str | None, float]:
    if perim <= 0:
        return None, None, 0.0
    ratio = bw / max(1.0, bh)
    approx = cv2.approxPolyDP(cnt, 0.045 * perim, True)
    vertices = len(approx)

    if 0.62 <= ratio <= 1.62 and circularity >= 0.56 and fill_ratio >= 0.40:
        return 'ellipse', None, float(min(1.0, circularity))

    aspect = max(bw, bh) / max(1.0, min(bw, bh))
    if vertices == 3 and fill_ratio >= 0.32 and pix_area >= 24 and aspect <= 3.0:
        direction = _triangle_direction(approx)
        return 'triangle', direction, float(min(1.0, fill_ratio + 0.18))

    if (vertices == 4 and 0.45 <= ratio <= 2.20 and
            0.32 <= fill_ratio <= 0.72 and _is_rhombus_like(approx, bw, bh)):
        return 'rhombus', None, float(min(1.0, fill_ratio + 0.24))

    if fill_ratio >= 0.48 and vertices <= 8:
        if color_std > 34 and pix_area > 85:
            return None, None, 0.0
        return 'rectangle', None, float(min(1.0, fill_ratio))

    if fill_ratio >= 0.62 and pix_area >= 28:
        if color_std > 34 and pix_area > 85:
            return None, None, 0.0
        return 'rectangle', None, float(min(1.0, fill_ratio))

    return None, None, 0.0


def detect_skeleton_connectors(rgb: np.ndarray, text_blocks: list[dict],
                               axis_lines: list[dict],
                               detect_arrows: bool = False,
                               max_connectors: int = 140) -> list[dict]:
    """Trace non-axis connectors using thinned foreground skeletons."""
    try:
        from skimage.morphology import skeletonize
    except Exception:
        return []

    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    luma = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] +
            0.114 * arr[:, :, 2])
    ink = ((gray < 120) | ((luma < 190) & (chroma > 35))).astype(np.uint8)
    _erase_bboxes(ink, text_blocks, pad=4)
    ink = cv2.morphologyEx(
        ink * 255, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    ink = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    skel = skeletonize(ink > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, 8)
    connectors = []
    for li in range(1, n):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        bw = int(stats[li, cv2.CC_STAT_WIDTH])
        bh = int(stats[li, cv2.CC_STAT_HEIGHT])
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < 12 or max(bw, bh) < 16:
            continue
        if bw * bh > 0.16 * w * h:
            continue
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.22):
            continue

        comp = labels[y:y + bh, x:x + bw] == li
        comp_skel = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
        comp_skel[1:bh + 1, 1:bw + 1] = comp.astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        neigh = cv2.filter2D(comp_skel, cv2.CV_16S, kernel) - comp_skel
        node_mask = (comp_skel > 0) & ((neigh == 1) | (neigh >= 3))
        nodes = {(int(yy), int(xx)) for yy, xx in np.argwhere(node_mask)}
        if len(nodes) < 2 or len(nodes) > 36:
            continue

        visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for node in list(nodes):
            for nb in _neighbors8(node, comp_skel):
                edge = tuple(sorted((node, nb)))
                if edge in visited_edges:
                    continue
                path_pix = [node, nb]
                visited_edges.add(edge)
                prev = node
                cur = nb
                steps = 0
                while cur not in nodes and steps < 2000:
                    nxts = [p for p in _neighbors8(cur, comp_skel)
                            if p != prev]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    visited_edges.add(tuple(sorted((cur, nxt))))
                    path_pix.append(nxt)
                    prev, cur = cur, nxt
                    steps += 1

                if len(path_pix) < 10:
                    continue
                path = [(float(px + x - 1), float(py + y - 1))
                        for py, px in path_pix]
                length = _path_length(path)
                if length < 18:
                    continue
                if _is_near_axis_line(path, axis_lines):
                    continue
                pb = _path_bbox(path, pad=2)
                if _bbox_touches_text(pb, text_blocks, threshold=0.28):
                    continue
                simp = _rdp_path(path, epsilon=1.7)
                if len(simp) < 2 or _path_length(simp) < 16:
                    continue
                dx = abs(simp[-1][0] - simp[0][0])
                dy = abs(simp[-1][1] - simp[0][1])
                if len(simp) <= 3 and (dx < 4 or dy < 4):
                    continue
                connectors.append({
                    'kind': 'line',
                    'orient': 'P',
                    'path': simp[:16],
                    'length': length,
                    'width': 1.1,
                    'arrow_start': False,
                    'arrow_end': False,
                })

    connectors.sort(key=lambda c: c['length'], reverse=True)
    deduped = []
    for conn in connectors:
        bbox = _path_bbox(_line_path(conn), pad=4)
        if any(_bbox_iou(bbox, _path_bbox(_line_path(old), pad=4)) > 0.55
               for old in deduped):
            continue
        deduped.append(conn)
        if len(deduped) >= max_connectors:
            break

    deduped.sort(key=lambda c: (_path_bbox(_line_path(c))[1],
                                _path_bbox(_line_path(c))[0]))
    return deduped


def detect_native_shapes(rgb: np.ndarray, text_blocks: list[dict],
                         lines: list[dict], max_shapes: int = 280) -> list[dict]:
    """Recognize simple filled symbols as native draw.io shapes."""
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    pale_bg = ((luma > 175) & (chroma >= 4) & (chroma <= 80) &
               (arr[:, :, 2] >= arr[:, :, 0] - 8))
    colored = (chroma > 18) & (luma < 248)
    dark_filled = (luma < 88) & (chroma < 42)
    pale_symbol = ((luma > 120) & (luma < 247) & (chroma >= 5) &
                   (chroma <= 88) & (arr[:, :, 2] >= arr[:, :, 0] - 16))
    page_bg = (luma > 246) & (chroma < 9)
    mask = ((colored | dark_filled | pale_symbol) & ~page_bg).astype(np.uint8) * 255
    _erase_bboxes(mask, text_blocks, pad=2)
    _erase_lines(mask, lines, thickness=3)
    candidate = mask > 0
    q = (rgb.astype(np.uint16) // 32).clip(0, 7)
    qlabels = q[:, :, 0] * 64 + q[:, :, 1] * 8 + q[:, :, 2]
    shapes = []
    vals, counts = np.unique(qlabels[candidate], return_counts=True)
    color_bins = [int(v) for v, c in zip(vals, counts) if int(c) >= 14]
    for color_bin in color_bins:
        bin_mask = ((qlabels == color_bin) & candidate).astype(np.uint8) * 255
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        bin_mask = cv2.morphologyEx(
            bin_mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, 8)
        for li in range(1, n):
            x = int(stats[li, cv2.CC_STAT_LEFT])
            y = int(stats[li, cv2.CC_STAT_TOP])
            bw = int(stats[li, cv2.CC_STAT_WIDTH])
            bh = int(stats[li, cv2.CC_STAT_HEIGHT])
            pix_area = int(stats[li, cv2.CC_STAT_AREA])
            if pix_area < 18 or bw < 4 or bh < 4:
                continue
            if max(bw, bh) > 280 or bw * bh > 35000:
                continue
            aspect = max(bw, bh) / max(1.0, min(bw, bh))
            if aspect > 14 or (aspect > 8 and min(bw, bh) < 7):
                continue
            bbox = (float(x), float(y), float(x + bw), float(y + bh))
            if _bbox_touches_text(bbox, text_blocks, threshold=0.18):
                continue

            comp = (labels[y:y + bh, x:x + bw] == li).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(cnt))
            bbox_area = float(max(1, bw * bh))
            fill_ratio = contour_area / bbox_area
            if fill_ratio < 0.28:
                continue
            perim = float(cv2.arcLength(cnt, True))
            circularity = (4.0 * math.pi * contour_area / (perim * perim)
                           if perim > 0 else 0.0)
            ratio = bw / max(1.0, bh)

            samples = rgb[y:y + bh, x:x + bw][comp > 0]
            if len(samples) == 0:
                continue
            med = np.median(samples, axis=0)
            color_std = float(np.mean(np.std(samples.astype(np.float32),
                                             axis=0)))
            fill = _rgb_to_hex(med)
            stroke = _darken_hex(med)

            shape_kind, direction, confidence = _classify_native_contour(
                cnt, bw, bh, pix_area, fill_ratio, circularity, perim,
                color_std)
            if shape_kind is None:
                continue
            pad = 0.4 if shape_kind == 'ellipse' else 0.2
            shapes.append({
                'kind': 'native_shape',
                'shape': shape_kind,
                'direction': direction,
                'bbox': (max(0.0, float(x) - pad), max(0.0, float(y) - pad),
                         min(float(w), float(x + bw) + pad),
                         min(float(h), float(y + bh) + pad)),
                'fill': fill,
                'stroke': stroke,
                'area': float(pix_area),
                'confidence': float(confidence),
            })
    saturated_mask = ((chroma > 35) & (luma < 248) & ~pale_bg).astype(np.uint8) * 255
    _erase_bboxes(saturated_mask, text_blocks, pad=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(saturated_mask, 8)
    for li in range(1, n):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        bw = int(stats[li, cv2.CC_STAT_WIDTH])
        bh = int(stats[li, cv2.CC_STAT_HEIGHT])
        pix_area = int(stats[li, cv2.CC_STAT_AREA])
        if pix_area < 45 or bw < 7 or bh < 7 or max(bw, bh) > 30:
            continue
        ratio = bw / max(1.0, bh)
        if not (0.55 <= ratio <= 1.8):
            continue
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.18):
            continue
        comp = (labels[y:y + bh, x:x + bw] == li).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(cnt))
        fill_ratio = contour_area / float(max(1, bw * bh))
        perim = float(cv2.arcLength(cnt, True))
        circularity = (4.0 * math.pi * contour_area / (perim * perim)
                       if perim > 0 else 0.0)
        samples = rgb[y:y + bh, x:x + bw][comp > 0]
        if len(samples) == 0:
            continue
        med = np.median(samples, axis=0)
        color_std = float(np.mean(np.std(samples.astype(np.float32),
                                         axis=0)))
        shape_kind, direction, confidence = _classify_native_contour(
            cnt, bw, bh, pix_area, fill_ratio, circularity, perim, color_std)
        if shape_kind is None:
            continue
        pad = 0.5
        shapes.append({
            'kind': 'native_shape',
            'shape': shape_kind,
            'direction': direction,
            'bbox': (max(0.0, float(x) - pad), max(0.0, float(y) - pad),
                     min(float(w), float(x + bw) + pad),
                     min(float(h), float(y + bh) + pad)),
            'fill': _rgb_to_hex(med),
            'stroke': _darken_hex(med),
            'area': float(pix_area),
            'confidence': float(confidence),
        })
    if not shapes:
        return []

    shapes.sort(key=lambda s: (-s['confidence'], -s['area']))
    deduped = []
    for shape in shapes:
        if any(_bbox_iou(shape['bbox'], old['bbox']) > 0.72
               for old in deduped):
            continue
        deduped.append(shape)
        if len(deduped) >= max_shapes:
            break
    deduped.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
    return deduped


def detect_short_strokes(rgb: np.ndarray, text_blocks: list[dict],
                         lines: list[dict],
                         native_shapes: list[dict] | None = None,
                         max_strokes: int = 140) -> list[dict]:
    """Recognize small non-text strokes as editable native line cells."""
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    pale_bg = ((luma > 175) & (chroma >= 4) & (chroma <= 80) &
               (arr[:, :, 2] >= arr[:, :, 0] - 8))
    colored_stroke = (chroma > 42) & (luma < 225)
    dark_stroke = (luma < 150) & (chroma < 65)
    muted_stroke = (luma < 205) & (chroma >= 10) & (chroma <= 55)
    mask = ((colored_stroke | dark_stroke | muted_stroke) & ~pale_bg).astype(np.uint8) * 255
    _erase_bboxes(mask, text_blocks, pad=2)
    _erase_lines(mask, lines, thickness=4)
    if native_shapes:
        shape_boxes = [{'bbox': s['bbox']} for s in native_shapes]
        _erase_bboxes(mask, shape_boxes, pad=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    strokes = []
    for li in range(1, n):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        bw = int(stats[li, cv2.CC_STAT_WIDTH])
        bh = int(stats[li, cv2.CC_STAT_HEIGHT])
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < 7 or area > 190 or bw < 3 or bh < 3:
            continue
        if max(bw, bh) > 78 or bw * bh > 900:
            continue
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.10):
            continue

        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == li)
        if len(xs) < 6:
            continue
        coords = np.column_stack([
            xs.astype(np.float32) + float(x),
            ys.astype(np.float32) + float(y),
        ])
        center = coords.mean(axis=0)
        centered = coords - center
        cov = np.cov(centered.T)
        if not np.all(np.isfinite(cov)):
            continue
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vec = vecs[:, order[0]]
        if vals[0] <= 0:
            continue
        proj = centered @ vec
        length = float(proj.max() - proj.min())
        width_est = float(max(1.0, 2.0 * math.sqrt(max(vals[1], 0.0))))
        bbox_aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if length < 12.0:
            continue
        if length / max(1.0, width_est) < 3.0 and bbox_aspect < 3.2:
            continue
        if width_est > 5.0 and min(bw, bh) > 5:
            continue

        p0 = center + vec * proj.min()
        p1 = center + vec * proj.max()
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        if dx < 1.0 and dy < 1.0:
            continue
        if dx >= dy * 3.0:
            orient = 'H'
            cy = (y0 + y1) / 2.0
            points = (min(x0, x1), cy, max(x0, x1), cy)
            path = [(points[0], points[1]), (points[2], points[3])]
        elif dy >= dx * 3.0:
            orient = 'V'
            cx = (x0 + x1) / 2.0
            points = (cx, min(y0, y1), cx, max(y0, y1))
            path = [(points[0], points[1]), (points[2], points[3])]
        else:
            orient = 'P'
            points = (x0, y0, x1, y1)
            path = [(x0, y0), (x1, y1)]

        stroke = {
            'kind': 'line',
            'orient': orient,
            'points': tuple(float(v) for v in points),
            'width': float(max(0.8, min(1.8, width_est))),
            'arrow_start': False,
            'arrow_end': False,
            'stroke': _estimate_line_stroke(rgb, path),
            'source': 'short_stroke',
            'length': length,
            'area': float(area),
        }
        strokes.append(stroke)

    strokes.sort(key=lambda s: (-s['length'], -s['area']))
    deduped = []
    for stroke in strokes:
        bbox = _path_bbox(_line_path(stroke), pad=3)
        if any(_bbox_iou(bbox, _path_bbox(_line_path(old), pad=3)) > 0.55
               for old in deduped):
            continue
        deduped.append(stroke)
        if len(deduped) >= max_strokes:
            break
    deduped.sort(key=lambda s: (_path_bbox(_line_path(s))[1],
                                _path_bbox(_line_path(s))[0]))
    return deduped


def detect_contour_paths(rgb: np.ndarray, text_blocks: list[dict],
                         lines: list[dict],
                         native_shapes: list[dict] | None = None,
                         max_paths: int = 140) -> list[dict]:
    """Approximate small icon outlines as editable draw.io polylines."""
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    pale_bg = ((luma > 175) & (chroma >= 4) & (chroma <= 80) &
               (arr[:, :, 2] >= arr[:, :, 0] - 8))
    page_bg = (luma > 246) & (chroma < 9)
    dark_outline = (luma < 132) & (chroma < 95)
    colored_outline = (chroma > 32) & (luma < 230)
    mask = ((dark_outline | colored_outline) & ~pale_bg & ~page_bg).astype(np.uint8) * 255
    _erase_bboxes(mask, text_blocks, pad=2)
    _erase_lines(mask, lines, thickness=4)
    if native_shapes:
        shape_boxes = [{'bbox': s['bbox']} for s in native_shapes]
        _erase_bboxes(mask, shape_boxes, pad=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 5 or bh < 5:
            continue
        if bw > 92 or bh > 92 or bw * bh > 3600:
            continue
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.12):
            continue
        if native_shapes and any(
            _bbox_overlap_fraction(bbox, s['bbox']) > 0.48
            for s in native_shapes
        ):
            continue
        perim = float(cv2.arcLength(cnt, True))
        if perim < 18 or perim > 320:
            continue
        contour_area = float(cv2.contourArea(cnt))
        if contour_area / max(1.0, bw * bh) > 0.72:
            continue
        approx = cv2.approxPolyDP(cnt, 1.25, True)
        if len(approx) < 3 or len(approx) > 24:
            continue
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        if cv2.isContourConvex(approx) and len(pts) <= 4:
            continue
        pts.append(pts[0])
        if _path_length(pts) < 18:
            continue
        paths.append({
            'kind': 'line',
            'orient': 'P',
            'path': pts,
            'length': _path_length(pts),
            'width': 0.9,
            'arrow_start': False,
            'arrow_end': False,
            'stroke': _estimate_line_stroke(rgb, pts),
            'source': 'contour_path',
        })

    paths.sort(key=lambda p: (-p['length'], _path_bbox(_line_path(p))[1],
                              _path_bbox(_line_path(p))[0]))
    deduped = []
    for path in paths:
        bbox = _path_bbox(_line_path(path), pad=3)
        if any(_bbox_iou(bbox, _path_bbox(_line_path(old), pad=3)) > 0.55
               for old in deduped):
            continue
        deduped.append(path)
        if len(deduped) >= max_paths:
            break
    deduped.sort(key=lambda p: (_path_bbox(_line_path(p))[1],
                                _path_bbox(_line_path(p))[0]))
    return deduped


def detect_tiny_native_marks(rgb: np.ndarray, text_blocks: list[dict],
                             lines: list[dict],
                             native_shapes: list[dict] | None = None,
                             max_marks: int = 90) -> list[dict]:
    """Convert residual non-text specks into native dots/rectangles.

    This recovers icon pins, small graph nodes, chart dots, and chip details
    without falling back to raster crops or stencils.
    """
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    page_bg = (luma > 246) & (chroma < 9)
    mask = ((((luma < 112) & (chroma < 80)) |
             ((chroma > 20) & (luma < 238))) &
            ~page_bg).astype(np.uint8) * 255
    _erase_bboxes(mask, text_blocks, pad=2)
    _erase_lines(mask, lines, thickness=3)
    if native_shapes:
        _erase_bboxes(mask, native_shapes, pad=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    marks = []
    for li in range(1, n):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        bw = int(stats[li, cv2.CC_STAT_WIDTH])
        bh = int(stats[li, cv2.CC_STAT_HEIGHT])
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < 5 or area > 130 or bw < 2 or bh < 2:
            continue
        if bw > 24 or bh > 24:
            continue
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        if aspect > 5.5:
            continue
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.08):
            continue
        comp = labels[y:y + bh, x:x + bw] == li
        samples = rgb[y:y + bh, x:x + bw][comp]
        if len(samples) == 0:
            continue
        med = np.median(samples, axis=0)
        med_luma = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
        med_chroma = float(np.max(med) - np.min(med))
        if med_chroma < 12 and med_luma > 118:
            continue
        comp_u8 = comp.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(cnt))
            perim = float(cv2.arcLength(cnt, True))
            circularity = (4.0 * math.pi * contour_area / (perim * perim)
                           if perim > 0 else 0.0)
        else:
            circularity = 0.0
        shape_kind = 'ellipse' if circularity > 0.46 and aspect < 2.0 else 'rectangle'
        pad = 0.3 if shape_kind == 'ellipse' else 0.0
        marks.append({
            'kind': 'native_shape',
            'shape': shape_kind,
            'direction': None,
            'bbox': (max(0.0, float(x) - pad), max(0.0, float(y) - pad),
                     min(float(w), float(x + bw) + pad),
                     min(float(h), float(y + bh) + pad)),
            'fill': _rgb_to_hex(med),
            'stroke': _darken_hex(med),
            'area': float(area),
            'confidence': 0.45,
            'source': 'tiny_mark',
            'chroma': med_chroma,
        })

    centers = [
        ((m['bbox'][0] + m['bbox'][2]) / 2.0,
         (m['bbox'][1] + m['bbox'][3]) / 2.0)
        for m in marks
    ]
    clustered_marks = []
    for i, mark in enumerate(marks):
        cx, cy = centers[i]
        has_neighbor = any(
            j != i and math.hypot(cx - ox, cy - oy) <= 32.0
            for j, (ox, oy) in enumerate(centers)
        )
        if mark.get('area', 0.0) >= 24 or mark.get('chroma', 0.0) >= 30 or has_neighbor:
            clustered_marks.append(mark)
    marks = clustered_marks

    marks.sort(key=lambda s: (-s['area'], s['bbox'][1], s['bbox'][0]))
    deduped = []
    for mark in marks:
        if any(_bbox_iou(mark['bbox'], old['bbox']) > 0.55
               for old in deduped):
            continue
        deduped.append(mark)
        if len(deduped) >= max_marks:
            break
    deduped.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
    return deduped


def filter_background_native_shapes(shapes: list[dict],
                                    rects: list[dict]) -> list[dict]:
    """Remove pale container/background regions misclassified as symbols."""
    filtered = []
    for shape in shapes:
        if shape.get('source') == 'tiny_mark':
            filtered.append(shape)
            continue
        x0, y0, x1, y1 = shape['bbox']
        bw = x1 - x0
        bh = y1 - y0
        area = float(shape.get('area', max(1.0, bw * bh)))
        r, g, b = _hex_to_rgb(shape.get('fill', '#000000'))
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        chroma = max(r, g, b) - min(r, g, b)
        pale_large = (
            area >= 650 and luma >= 218 and chroma <= 32 and
            max(bw, bh) >= 34
        )
        pale_background_blob = (
            area >= 900 and luma >= 226 and chroma <= 22 and
            max(bw, bh) >= 42
        )
        inside_rect = any(
            _bbox_overlap_fraction(shape['bbox'], rect['bbox']) >= 0.62
            for rect in rects
        )
        if (pale_large and inside_rect) or pale_background_blob:
            continue
        filtered.append(shape)
    return filtered


def detect_bar_charts(rgb: np.ndarray, text_blocks: list[dict],
                      max_rects: int = 150) -> list[dict]:
    """Detect bar chart columns as native colored rectangles.

    Finds narrow vertical colored blobs (height > 1.4 * width, height ≥ 12px,
    width ≤ 55px) that cluster horizontally — the hallmark of a bar chart.
    Groups with ≥ 2 bars are emitted as individual filled rectangles.
    """
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    page_bg = (luma > 246) & (chroma < 9)

    bar_mask = ((chroma > 18) & (luma > 65) & (luma < 238) &
                ~page_bg).astype(np.uint8) * 255
    _erase_bboxes(bar_mask, text_blocks, pad=2)
    bar_mask = cv2.morphologyEx(
        bar_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bar_mask, 8)
    candidates = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 55 or bw < 6 or bh < 12: continue
        if bw > 55 or bh > 220: continue
        if bh < bw * 1.4: continue  # must be taller than wide
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.15): continue
        comp = (labels[y:y + bh, x:x + bw] == i)
        fill_ratio = area / max(1, bw * bh)
        if fill_ratio < 0.35: continue
        samples = rgb[y:y + bh, x:x + bw][comp]
        if len(samples) == 0: continue
        med = np.median(samples, axis=0)
        candidates.append({
            'kind': 'rect',
            'bbox': bbox,
            'fill': _rgb_to_hex(med),
            'stroke': _darken_hex(med),
            'area': float(area),
            '_cx': float(x + bw / 2),
        })

    if not candidates:
        return []

    # Group by horizontal proximity (≤ 60px gap between bar centers)
    candidates.sort(key=lambda c: c['_cx'])
    groups: list[list[dict]] = []
    cur: list[dict] = [candidates[0]]
    for c in candidates[1:]:
        if c['_cx'] - cur[-1]['_cx'] <= 60:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)

    result = []
    for grp in groups:
        if len(grp) < 2:
            continue
        # Merge bars that are horizontally overlapping or touching (gap < 4px)
        # to prevent adjacent thin bars from being misread as letters by OCR.
        merged: list[dict] = []
        grp_sorted = sorted(grp, key=lambda r: r['bbox'][0])
        cur_r = dict(grp_sorted[0])
        del cur_r['_cx']
        for nxt in grp_sorted[1:]:
            cx0, cy0, cx1, cy1 = cur_r['bbox']
            nx0, ny0, nx1, ny1 = nxt['bbox']
            gap = nx0 - cx1
            if gap < 4:
                # Merge: expand bbox, average fill by area
                ca = (cx1 - cx0) * (cy1 - cy0)
                na = (nx1 - nx0) * (ny1 - ny0)
                cur_r['bbox'] = (cx0, min(cy0, ny0),
                                  nx1, max(cy1, ny1))
                # blend fill proportionally
                cr, cg, cb = _hex_to_rgb(cur_r['fill'])
                nr, ng, nb = _hex_to_rgb(nxt['fill'])
                t = na / max(1, ca + na)
                blended = (int(cr + t*(nr-cr)), int(cg + t*(ng-cg)), int(cb + t*(nb-cb)))
                cur_r['fill'] = _rgb_to_hex(blended)
                cur_r['stroke'] = _darken_hex(blended)
            else:
                merged.append(cur_r)
                cur_r = dict(nxt)
                if '_cx' in cur_r: del cur_r['_cx']
        if '_cx' in cur_r: del cur_r['_cx']
        merged.append(cur_r)
        result.extend(merged)
    return result[:max_rects]


def detect_icon_crops(img: Image.Image, rgb: np.ndarray,
                      text_blocks: list[dict], lines: list[dict],
                      native_shapes: list[dict] | None = None,
                      max_dim: int = 150,
                      max_crops: int = 260) -> list[dict]:
    """Find small non-text visual islands and preserve them as editable crops."""
    arr = rgb.astype(np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    pale_bg = ((luma > 175) & (chroma >= 4) & (chroma <= 80) &
               (arr[:, :, 2] >= arr[:, :, 0] - 8))
    fg = (((luma < 218) | (chroma > 28)) & ~pale_bg).astype(np.uint8) * 255
    _erase_bboxes(fg, text_blocks, pad=2)
    for line in lines:
        pts = _line_path(line)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            x0, y0 = p0
            x1, y1 = p1
            cv2.line(fg, (int(round(x0)), int(round(y0))),
                     (int(round(x1)), int(round(y1))), 0, 5)

    fg = cv2.morphologyEx(
        fg, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    fg = cv2.dilate(
        fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    crops = []
    for li in range(1, n):
        x = int(stats[li, cv2.CC_STAT_LEFT])
        y = int(stats[li, cv2.CC_STAT_TOP])
        bw = int(stats[li, cv2.CC_STAT_WIDTH])
        bh = int(stats[li, cv2.CC_STAT_HEIGHT])
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < 20 or bw < 5 or bh < 5:
            continue
        if max(bw, bh) > max_dim:
            continue
        if max(bw, bh) / max(1.0, min(bw, bh)) > 12:
            continue
        bbox = (float(x), float(y), float(x + bw), float(y + bh))
        if _bbox_touches_text(bbox, text_blocks, threshold=0.18):
            continue
        component_mask = labels[y:y + bh, x:x + bw] == li
        component_chroma = chroma[y:y + bh, x:x + bw][component_mask]
        crop_chroma = (float(np.percentile(component_chroma, 75))
                       if component_chroma.size else 0.0)
        component_rgb = arr[y:y + bh, x:x + bw][component_mask]
        mean_rgb = ([float(v) for v in component_rgb.mean(axis=0)]
                    if component_rgb.size else [0.0, 0.0, 0.0])
        colored_crop = crop_chroma > 34
        line_cover = sum(
            _bbox_overlap_fraction(bbox, _path_bbox(_line_path(line), pad=4))
            for line in lines)
        if (not colored_crop and
                ((area <= 160 and line_cover > 0.65) or
                 (area <= 360 and line_cover > 0.90))):
            continue
        if native_shapes:
            shape_cover = sum(_bbox_overlap_fraction(bbox, shape['bbox'])
                              for shape in native_shapes)
            if (shape_cover > 0.55 or
                    (area <= 450 and shape_cover > 0.35) or
                    any(_bbox_overlap_fraction(bbox, shape['bbox']) > 0.68
                        for shape in native_shapes)):
                continue
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        crop_img = _transparent_foreground_crop(img.crop((x0, y0, x1, y1)))
        crops.append({
            'kind': 'icon_crop',
            'bbox': (float(x0), float(y0), float(x1), float(y1)),
            'area': float(area),
            'chroma': crop_chroma,
            'mean_rgb': mean_rgb,
            'image': crop_img,
        })

    if crops:
        parent = list(range(len(crops)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        def merge_candidate(crop: dict) -> bool:
            x0, y0, x1, y1 = crop['bbox']
            return (crop.get('chroma', 0.0) > 34 and
                    crop.get('area', 0.0) <= 260 and
                    (y1 - y0) <= 32 and (x1 - x0) <= 52)

        for i, a in enumerate(crops):
            if not merge_candidate(a):
                continue
            ax0, ay0, ax1, ay1 = a['bbox']
            ah = max(1.0, ay1 - ay0)
            for j in range(i + 1, len(crops)):
                b = crops[j]
                if not merge_candidate(b):
                    continue
                bx0, by0, bx1, by1 = b['bbox']
                bh = max(1.0, by1 - by0)
                y_overlap = max(0.0, min(ay1, by1) - max(ay0, by0))
                if y_overlap / min(ah, bh) < 0.35:
                    continue
                gap = max(0.0, max(ax0, bx0) - min(ax1, bx1))
                combo_w = max(ax1, bx1) - min(ax0, bx0)
                combo_h = max(ay1, by1) - min(ay0, by0)
                color_dist = float(np.linalg.norm(
                    np.array(a.get('mean_rgb', [0, 0, 0]), dtype=float) -
                    np.array(b.get('mean_rgb', [0, 0, 0]), dtype=float)))
                if gap <= 10 and combo_w <= 92 and combo_h <= 34 and color_dist <= 75:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i, crop in enumerate(crops):
            if merge_candidate(crop):
                groups.setdefault(find(i), []).append(i)

        merged_indices: set[int] = set()
        merged_crops = []
        for group in groups.values():
            if len(group) < 2:
                continue
            xs0, ys0, xs1, ys1 = zip(*(crops[i]['bbox'] for i in group))
            mx0 = max(0, int(np.floor(min(xs0))) - 1)
            my0 = max(0, int(np.floor(min(ys0))) - 1)
            mx1 = min(w, int(np.ceil(max(xs1))) + 1)
            my1 = min(h, int(np.ceil(max(ys1))) + 1)
            if mx1 - mx0 > 110 or my1 - my0 > 38:
                continue
            crop_img = _transparent_foreground_crop(
                img.crop((mx0, my0, mx1, my1)))
            mean_rgb = np.mean([
                np.array(crops[i].get('mean_rgb', [0, 0, 0]), dtype=float)
                for i in group
            ], axis=0)
            merged_crops.append({
                'kind': 'icon_crop',
                'bbox': (float(mx0), float(my0), float(mx1), float(my1)),
                'area': float(sum(crops[i].get('area', 0.0) for i in group)),
                'chroma': float(max(crops[i].get('chroma', 0.0)
                                    for i in group)),
                'mean_rgb': [float(v) for v in mean_rgb],
                'merged_components': len(group),
                'image': crop_img,
            })
            merged_indices.update(group)

        if merged_crops:
            crops = [crop for i, crop in enumerate(crops)
                     if i not in merged_indices] + merged_crops

    crops.sort(key=lambda c: (c['bbox'][1], c['bbox'][0], -c['area']))
    deduped = []
    for crop in crops:
        if any(_bbox_iou(crop['bbox'], old['bbox']) > 0.65
               for old in deduped):
            continue
        deduped.append(crop)
        if len(deduped) >= max_crops:
            break
    return deduped


def detect_foreground_tiles(img: Image.Image, tile_size: int = 320,
                            min_pixels: int = 40) -> list[dict]:
    """Transparent foreground tiles for exact-looking no-background rebuilds."""
    rgb = np.array(img.convert('RGB'))
    h, w = rgb.shape[:2]
    tiles = []
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            crop_rgb = rgb[y0:y1, x0:x1].astype(np.float32)
            luma = (0.299 * crop_rgb[:, :, 0] +
                    0.587 * crop_rgb[:, :, 1] +
                    0.114 * crop_rgb[:, :, 2])
            chroma = crop_rgb.max(axis=2) - crop_rgb.min(axis=2)
            mask = ((luma < 245) | (chroma > 10)).astype(np.uint8)
            if int(mask.sum()) < min_pixels:
                continue
            ys, xs = np.nonzero(mask)
            rx0 = max(0, int(xs.min()) - 1)
            ry0 = max(0, int(ys.min()) - 1)
            rx1 = min(x1 - x0, int(xs.max()) + 2)
            ry1 = min(y1 - y0, int(ys.max()) + 2)
            crop = img.crop((x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1))
            tiles.append({
                'kind': 'foreground_tile',
                'bbox': (float(x0 + rx0), float(y0 + ry0),
                         float(x0 + rx1), float(y0 + ry1)),
                'pixels': int(mask[ry0:ry1, rx0:rx1].sum()),
                'image': _transparent_foreground_crop(crop),
            })
    return tiles


def _wrap_drawio(cells: list[str], W: int, H: int,
                 overlay_visible: bool = False,
                 visual_foreground_visible: bool = False,
                 native_overlay_visible: bool = True,
                 include_source_layer: bool = True,
                 include_raster_layers: bool = True) -> str:
    native_visible = overlay_visible and native_overlay_visible
    primitive_visible_attr = '' if native_visible else ' visible="0"'
    visual_visible_attr = '' if visual_foreground_visible else ' visible="0"'
    source_layer = (
        '        <mxCell id="1" value="source image" parent="0"/>\n'
        if include_source_layer else
        '        <mxCell id="1" value="pure native canvas" parent="0"/>\n'
    )
    raster_layers = ''
    if include_raster_layers:
        raster_layers = (
            f'        <mxCell id="5" value="editable icon crops" parent="0"{primitive_visible_attr}/>\n'
            f'        <mxCell id="6" value="visual foreground tiles" parent="0"{visual_visible_attr}/>\n'
        )
    body = '\n        '.join(cells)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" modified="2026-05-19T00:00:00.000Z" '
        'agent="png_to_drawio" version="24.7.0" type="device">\n'
        '  <diagram id="png_direct" name="png-direct">\n'
        f'    <mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" '
        f'pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" '
        f'shadow="0" background="none">\n'
        '      <root>\n'
        '        <mxCell id="0"/>\n'
        f'{source_layer}'
        f'        <mxCell id="2" value="editable rectangles" parent="0"{primitive_visible_attr}/>\n'
        f'        <mxCell id="3" value="editable lines" parent="0"{primitive_visible_attr}/>\n'
        f'        <mxCell id="4" value="editable text" parent="0"{primitive_visible_attr}/>\n'
        f'{raster_layers}'
        f'        <mxCell id="7" value="editable native symbols" parent="0"{primitive_visible_attr}/>\n'
        f'        {body}\n'
        '      </root>\n'
        '    </mxGraphModel>\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )


def assert_pure_native_drawio(drawio: str) -> None:
    forbidden = [
        'shape=image',
        'data:image',
        '<image',
        'base64',
        'shape=stencil',
        'stencil(',
    ]
    hits = [token for token in forbidden if token in drawio]
    if hits:
        raise ValueError(
            'pure-native drawio contains forbidden raster/stencil tokens: '
            + ', '.join(hits))


def convert_png_to_drawio(input_path: str, output_path: str,
                          show_overlay: bool = False,
                          include_background: bool = True,
                          font_family: str = 'Arial',
                          ocr_conf: float = 45.0,
                          trusted_text_conf: float = 70.0,
                          ocr_scale: float = 2.0,
                          ocr_psm: int = 11,
                          ocr_multipass: bool = False,
                          detect_arrows: bool = False,
                          emit_skeleton_connectors: bool = True,
                          emit_native_shapes: bool = True,
                          emit_contour_paths: bool = False,
                          emit_icon_crops: bool = True,
                          emit_visual_foreground: bool = True,
                          show_visual_foreground: bool = False,
                          native_overlay_visible: bool = True,
                          pure_native: bool = False,
                          ledger_path: str | None = None) -> dict:
    if pure_native:
        show_overlay = True
        include_background = False
        emit_icon_crops = False
        emit_visual_foreground = False
        show_visual_foreground = False
        native_overlay_visible = True

    img = Image.open(input_path).convert('RGB')
    rgb = np.array(img)
    W, H = img.size

    text_blocks = detect_ocr_text(
        img, conf_threshold=ocr_conf, scale=ocr_scale, psm=ocr_psm,
        multipass=ocr_multipass)
    trusted_text_blocks = [
        t for t in text_blocks
        if _is_trusted_text(t, min_conf=trusted_text_conf)
    ]
    geometry_text_conf = max(70.0, trusted_text_conf)
    geometry_text_blocks = [
        t for t in text_blocks
        if _is_geometry_mask_text(t, min_conf=geometry_text_conf)
    ]
    detection_text_blocks = (
        geometry_text_blocks if pure_native else text_blocks
    )
    raw_fill_rects = (
        detect_pale_rectangles(rgb) +
        detect_separated_pale_rectangles(rgb)
    )
    border_rects = filter_border_rectangles(
        detect_border_rectangles(rgb), detection_text_blocks)
    fill_rects = filter_fill_rectangles(raw_fill_rects, border_rects)
    bar_chart_rects = (detect_bar_charts(rgb, detection_text_blocks)
                       if emit_native_shapes else [])
    rects = merge_rectangles(fill_rects, border_rects)
    # Bar chart rects are added after dedup so they are not swallowed by
    # large background panels.
    rects = rects + [r for r in bar_chart_rects
                     if not any(_bbox_iou(r['bbox'], old['bbox']) > 0.60
                                for old in rects)]
    axis_lines = detect_lines(rgb, detection_text_blocks,
                              detect_arrows=detect_arrows)
    connectors = (detect_skeleton_connectors(
        rgb, detection_text_blocks, axis_lines, detect_arrows=detect_arrows)
        if emit_skeleton_connectors else [])
    lines = axis_lines + connectors
    if detect_arrows:
        _apply_directional_arrowheads(rgb, detection_text_blocks, lines)
    for line in lines:
        line['stroke'] = _estimate_line_stroke(rgb, _line_path(line))
    native_shapes = (detect_native_shapes(rgb, detection_text_blocks, lines)
                     if emit_native_shapes else [])
    native_shapes = filter_background_native_shapes(native_shapes, rects)
    contour_paths = (detect_contour_paths(
        rgb, text_blocks, lines, native_shapes=native_shapes)
        if emit_native_shapes and emit_contour_paths else [])
    lines = lines + contour_paths
    short_strokes = (detect_short_strokes(
        rgb, detection_text_blocks, lines, native_shapes=native_shapes)
        if emit_native_shapes else [])
    lines = lines + short_strokes
    tiny_marks = (detect_tiny_native_marks(
        rgb, detection_text_blocks, lines, native_shapes=native_shapes)
        if emit_native_shapes else [])
    native_shapes = native_shapes + tiny_marks
    icon_crops = (detect_icon_crops(
        img, rgb, trusted_text_blocks, lines, native_shapes=native_shapes)
                  if emit_icon_crops else [])
    foreground_tiles = (detect_foreground_tiles(img)
                        if emit_visual_foreground else [])

    cells = []
    cid = 100
    cells.append(_emit_canvas_anchor_cell(cid, W, H, parent='1'))
    cid += 1
    if include_background:
        cells.append(_emit_image_cell(cid, img, (0, 0, W, H), parent='1',
                                      locked=True))
        cid += 1

    for tile in foreground_tiles:
        cells.append(_emit_crop_cell(cid, tile, parent='6'))
        cid += 1
    for rect in rects:
        cells.append(_emit_rect_cell(cid, rect, parent='2'))
        cid += 1
    for crop in icon_crops:
        cells.append(_emit_crop_cell(cid, crop, parent='5'))
        cid += 1
    for shape in native_shapes:
        cells.append(_emit_native_shape_cell(cid, shape, parent='7'))
        cid += 1
    for line in lines:
        cells.append(_emit_edge_cell(cid, line, parent='3'))
        cid += 1
    for text in trusted_text_blocks:
        cells.append(_emit_text_cell(cid, text, parent='4',
                                     font_family=font_family))
        cid += 1

    drawio = _wrap_drawio(
        cells, W, H, overlay_visible=show_overlay,
        visual_foreground_visible=show_visual_foreground,
        native_overlay_visible=native_overlay_visible,
        include_source_layer=include_background,
        include_raster_layers=(emit_icon_crops or emit_visual_foreground))
    if pure_native:
        assert_pure_native_drawio(drawio)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(drawio)

    ledger = {
        'source': os.path.abspath(input_path),
        'width': W,
        'height': H,
        'layers': {
            'source_image': {'id': '1', 'visible': include_background},
            'editable_rectangles': {
                'id': '2', 'visible': show_overlay and native_overlay_visible},
            'editable_lines': {
                'id': '3', 'visible': show_overlay and native_overlay_visible},
            'editable_text': {
                'id': '4', 'visible': show_overlay and native_overlay_visible},
            'editable_icon_crops': {
                'id': '5', 'visible': (
                    emit_icon_crops and show_overlay and native_overlay_visible)},
            'visual_foreground_tiles': {
                'id': '6', 'visible': (
                    emit_visual_foreground and show_visual_foreground)},
            'editable_native_symbols': {
                'id': '7', 'visible': show_overlay and native_overlay_visible},
        },
        'primitives': {
            'rectangles': rects,
            'native_shapes': native_shapes,
            'tiny_marks': tiny_marks,
            'contour_paths': contour_paths,
            'short_strokes': short_strokes,
            'foreground_tiles': [
                {k: v for k, v in tile.items() if k != 'image'}
                for tile in foreground_tiles
            ],
            'icon_crops': [
                {k: v for k, v in crop.items() if k != 'image'}
                for crop in icon_crops
            ],
            'lines': lines,
            'text': trusted_text_blocks,
            'ocr_all_text': text_blocks,
            'geometry_text': geometry_text_blocks,
        },
        'counts': {
            'rectangles': len(rects),
            'bar_chart_rects': len(bar_chart_rects),
            'native_shapes': len(native_shapes),
            'tiny_marks': len(tiny_marks),
            'foreground_tiles': len(foreground_tiles),
            'icon_crops': len(icon_crops),
            'axis_lines': len(axis_lines),
            'skeleton_connectors': len(connectors),
            'short_strokes': len(short_strokes),
            'contour_paths': len(contour_paths),
            'lines': len(lines),
            'text': len(trusted_text_blocks),
            'ocr_all_text': len(text_blocks),
            'geometry_text': len(geometry_text_blocks),
            'cells': len(cells),
            'pure_native': bool(pure_native),
        },
    }
    if ledger_path is None:
        ledger_path = str(out.with_suffix('.primitive_ledger.json'))
    Path(ledger_path).write_text(json.dumps(ledger, indent=2))
    return ledger['counts'] | {'output': str(out), 'ledger': ledger_path}


def main():
    ap = argparse.ArgumentParser(
        description='Direct PNG/JPG to drawio primitive-overlay converter')
    ap.add_argument('input', help='input PNG/JPG/WebP/TIFF')
    ap.add_argument('-o', '--output', default=None,
                    help='output .drawio path')
    ap.add_argument('--show-overlay', action='store_true',
                    help='make editable primitive layer visible')
    ap.add_argument('--pure-native', action='store_true',
                    help='strict framework mode: no source image, no raster '
                         'icon crops, no foreground tiles, no base64, no '
                         'stencils; output only native draw.io primitives')
    ap.add_argument('--no-background', action='store_true',
                    help='do not include locked source PNG backing')
    ap.add_argument('--font-family', default='Arial')
    ap.add_argument('--ocr-conf', type=float, default=45.0)
    ap.add_argument('--trusted-text-conf', type=float, default=70.0)
    ap.add_argument('--ocr-scale', type=float, default=2.0)
    ap.add_argument('--ocr-psm', type=int, default=11)
    ap.add_argument('--ocr-multipass', action='store_true',
                    help='run several Tesseract PSM/scale passes and merge '
                         'deduplicated text boxes')
    ap.add_argument('--detect-arrows', action='store_true',
                    help='experimental: infer arrowheads for native lines')
    ap.add_argument('--no-skeleton-connectors', action='store_true',
                    help='do not trace freeform connectors from skeletons')
    ap.add_argument('--no-native-shapes', action='store_true',
                    help='do not emit native ellipse/rectangle symbols')
    ap.add_argument('--contour-paths', action='store_true',
                    help='experimental: emit small icon outlines as native '
                         'polyline paths; disabled by default because noisy '
                         'contours can make diagrams look dirty')
    ap.add_argument('--no-icon-crops', action='store_true',
                    help='do not emit editable raster crops for small icons')
    ap.add_argument('--no-visual-foreground', action='store_true',
                    help='do not include transparent foreground tile layer')
    ap.add_argument('--show-visual-foreground', action='store_true',
                    help='make transparent foreground tile layer visible')
    ap.add_argument('--hide-native-overlay', action='store_true',
                    help='with --show-overlay, keep native primitive layers hidden')
    ap.add_argument('--ledger', default=None,
                    help='primitive ledger JSON path')
    args = ap.parse_args()

    inp = Path(args.input)
    output = args.output or str(inp.with_suffix('.drawio'))
    stats = convert_png_to_drawio(
        str(inp), output,
        show_overlay=args.show_overlay,
        include_background=not args.no_background,
        font_family=args.font_family,
        ocr_conf=args.ocr_conf,
        trusted_text_conf=args.trusted_text_conf,
        ocr_scale=args.ocr_scale,
        ocr_psm=args.ocr_psm,
        ocr_multipass=args.ocr_multipass,
        detect_arrows=args.detect_arrows,
        emit_skeleton_connectors=not args.no_skeleton_connectors,
        emit_native_shapes=not args.no_native_shapes,
        emit_contour_paths=args.contour_paths,
        emit_icon_crops=not args.no_icon_crops,
        emit_visual_foreground=not args.no_visual_foreground,
        show_visual_foreground=args.show_visual_foreground,
        native_overlay_visible=not args.hide_native_overlay,
        pure_native=args.pure_native,
        ledger_path=args.ledger,
    )
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
