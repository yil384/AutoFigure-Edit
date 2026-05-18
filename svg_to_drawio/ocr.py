"""OCR pipeline + multi-label split.

`ocr_clusters` (from v5): runs full-image OCR + crop fallback + Claude verify.
`split_multilabel_clusters`: when one cluster absorbed glyphs from two labels
on the same baseline (e.g. 'Quantum Chemistry' + 'Hubbard' both detected as
one cluster), splits off the spillover into a new cluster claiming the
unclaimed OCR text.
"""
import difflib

from svg_to_drawio_v5 import ocr_clusters
from svg_to_drawio.auth import make_anthropic_client

__all__ = [
    'ocr_clusters', 'split_multilabel_clusters', 'clean_suspect_vertical_ocr',
    'reject_hallucinated_text', 'recover_vertical_with_crop',
    'recover_missing_chars', 'recover_inter_cluster_chars',
    'merge_hyphen_continuation', 'cross_reference_typos',
    'recover_split_pending', 'merge_stacked_text_lines',
]


def merge_stacked_text_lines(clusters: list, max_lines: int = 4) -> int:
    """Merge short text clusters that are vertically stacked into one
    multi-line cluster — e.g. 'Verilog SFT' on top of 'corpus (87K)' inside
    one box becomes a single 'Verilog SFT\\ncorpus (87K)' label. Without
    this, the semantic-shape pass picks one of the two fragments and
    drops the other.

    Stricter than merge_hyphen_continuation:
      - Both lines must be horizontal text with non-empty text.
      - Same x-center (≤ 8 px difference), similar widths (within 50%).
      - y gap between them ≤ left_height × 1.4 (just below).
      - Combined still ≤ `max_lines` rows.
    """
    n = 0
    # Iterate repeatedly because a 3-line label collapses pair-by-pair.
    for _ in range(max_lines):
        merged_any = False
        for li, lc in enumerate(clusters):
            if lc.get('vertical') or not (lc.get('text') or '').strip():
                continue
            lx0, ly0, lx1, ly1 = lc['bbox']
            lh = ly1 - ly0; lw = lx1 - lx0
            lcx = (lx0 + lx1) / 2
            lines_in_l = (lc.get('text') or '').count('\n') + 1
            if lines_in_l >= max_lines:
                continue
            for ri, rc in enumerate(clusters):
                if li == ri or rc.get('vertical'):
                    continue
                rt = (rc.get('text') or '').strip()
                if not rt:
                    continue
                rx0, ry0, rx1, ry1 = rc['bbox']
                rh = ry1 - ry0; rw = rx1 - rx0
                rcx = (rx0 + rx1) / 2
                gap = ry0 - ly1
                if gap < -2 or gap > lh * 1.4:
                    continue
                # Same column / same x-center
                if abs(rcx - lcx) > max(8, lh * 0.5):
                    continue
                # Similar widths
                if max(rw, lw) > min(rw, lw) * 2.0:
                    continue
                # Similar font heights
                if max(rh, lh) > min(rh, lh) * 1.5:
                    continue
                # Merge
                ltext = (lc.get('text') or '').strip()
                lc['text'] = f'{ltext}\n{rt}'
                lc['bbox'] = [min(lx0, rx0), ly0, max(lx1, rx1), ry1]
                lc['_multi_line'] = True
                lp = set(lc.get('glyph_path_ids', []))
                rp = set(rc.get('glyph_path_ids', []))
                lc['glyph_path_ids'] = lp | rp
                rc['text'] = ''
                rc['glyph_path_ids'] = set()
                n += 1
                merged_any = True
                break
            if merged_any:
                break
        if not merged_any:
            break
    return n


def recover_split_pending(clusters: list, png_path: str) -> int:
    """Re-OCR each cluster whose source == 'split_pending' (created by
    split_spatially_joined_clusters when the parent's text didn't have enough
    words to distribute). Sends the bbox crop to Claude and accepts any
    non-empty reading. Returns count recovered.
    """
    try:
        import cv2
    except ImportError:
        return 0
    img = cv2.imread(png_path)
    if img is None:
        return 0
    H, W = img.shape[:2]
    n = 0
    for c in clusters:
        if c.get('ocr_source') != 'split_pending':
            continue
        x0, y0, x1, y1 = c['bbox']
        px = max(6, int((x1 - x0) * 0.05)); py = max(3, int((y1 - y0) * 0.2))
        cx0 = max(0, int(x0) - px); cy0 = max(0, int(y0) - py)
        cx1 = min(W, int(x1) + px); cy1 = min(H, int(y1) + py)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        if crop.shape[0] < 60:
            sf = 60 / max(1, crop.shape[0])
            crop = cv2.resize(crop, (int(crop.shape[1] * sf), int(crop.shape[0] * sf)),
                              interpolation=cv2.INTER_CUBIC)
        fixed = _claude_fix_text(crop, '')
        if fixed and fixed.strip():
            c['text'] = fixed.strip()
            c['ocr_source'] = 'split_recovered'
            n += 1
    return n


def _is_subsequence(short: str, longer: str) -> bool:
    """Return True if every character of `short` appears in `longer` in order
    (case-insensitive). 'Ohead' is a subsequence of 'Overhead'."""
    s = short.lower(); l = longer.lower()
    i = 0
    for ch in l:
        if i < len(s) and s[i] == ch:
            i += 1
    return i == len(s)


def cross_reference_typos(clusters: list,
                          max_distance_px: float = 400.0,
                          min_word_len: int = 4) -> int:
    """Detect short-word typos by cross-referencing against longer same-panel
    sibling words. When a short alphabetic token (≥ `min_word_len` chars) is
    a strict subsequence of a longer word that appears in a nearby cluster
    (within `max_distance_px` pixel center-to-center), and the longer word
    is a real word (extra letters added consecutively, not interleaved
    randomly), replace the short with the longer.

    Targets cases like 'Physical-Qubit Ohead' → 'Physical-Qubit Overhead'
    where 'Overhead' appears as the chart-axis label of the same panel.

    Returns count of clusters whose text was modified.
    """
    # Index alphabetic tokens by lowercased form to their (cluster_idx, text, bbox).
    word_index: dict[str, list[tuple[int, str, list]]] = {}
    for ci, c in enumerate(clusters):
        text = (c.get('text') or '').strip()
        if not text or c.get('vertical'):
            continue
        for tok in text.replace('-', ' ').replace(',', ' ').split():
            cleaned = ''.join(ch for ch in tok if ch.isalpha())
            if len(cleaned) < min_word_len:
                continue
            word_index.setdefault(cleaned.lower(), []).append((ci, cleaned, c['bbox']))

    n = 0
    for ci, c in enumerate(clusters):
        text = (c.get('text') or '').strip()
        if not text or c.get('vertical'):
            continue
        cb = c['bbox']
        ccx = (cb[0] + cb[2]) / 2; ccy = (cb[1] + cb[3]) / 2
        new_text = text
        replaced_any = False
        # Walk each alphabetic token of this cluster
        # We split conservatively so punctuation stays attached to tokens
        # and we replace whole-word matches only.
        import re as _re
        for m in _re.finditer(r'[A-Za-z]+', text):
            tok = m.group()
            if len(tok) < min_word_len:
                continue
            tok_low = tok.lower()
            best = None
            for cand_low, entries in word_index.items():
                if cand_low == tok_low:
                    continue
                if len(cand_low) <= len(tok):
                    continue
                if not _is_subsequence(tok_low, cand_low):
                    continue
                # Limit edit distance: extra chars must be ≤ half the length
                if len(cand_low) - len(tok) > max(2, len(tok) // 2):
                    continue
                # Require at least one nearby occurrence
                for ei, eword, ebb in entries:
                    if ei == ci:
                        continue
                    ecx = (ebb[0] + ebb[2]) / 2; ecy = (ebb[1] + ebb[3]) / 2
                    dist = ((ecx - ccx) ** 2 + (ecy - ccy) ** 2) ** 0.5
                    if dist <= max_distance_px:
                        if best is None or len(eword) > len(best):
                            best = eword
                        break
            if best:
                new_text = new_text.replace(tok, best)
                replaced_any = True
        if replaced_any and new_text != text:
            c['text'] = new_text
            c['source'] = (c.get('source') or '') + '+xref'
            n += 1
    return n


def merge_hyphen_continuation(clusters: list, png_path: str | None = None) -> int:
    """Merge a horizontal cluster whose text ends in '-' with a cluster on the
    next line directly below in the same x-range, e.g. 'Bloch- Interfero-' +
    'Messiah' → 'Bloch- Interfero-\\nMessiah'. Same for 'Non-' + 'Gaussian'.

    Why: vertically-stacked label fragments render fine as separate cells but
    aren't editable as a single label, and the hyphen looks dangling. Merging
    keeps both lines visible (drawio renders the '\\n' with whiteSpace=wrap)
    while making the text editable as one piece.

    When `png_path` is provided AND a Claude credential is available, the
    merged region is re-OCR'd with Claude on a high-res crop. Why: previous
    line-by-line OCR can misread short words ('meters' → 'Messiah'); the
    wider merged crop usually disambiguates. The Claude reading replaces the
    old text only when its alphanumeric form is similar to the concatenation
    (substring match on either part) — guards against hallucination.

    Heuristics for pairing:
      - left ends with '-' (after strip)
      - right is directly below: 0 ≤ right.y0 - left.y1 ≤ left_height * 1.5
      - x-overlap ≥ 50% of right's width (or left's width, whichever smaller)
      - both horizontal clusters with non-empty text

    Returns count merged. Right cluster's text is cleared so it's not double-
    rendered.
    """
    img = None
    if png_path:
        try:
            import cv2
            img = cv2.imread(png_path)
        except ImportError:
            img = None
    n = 0
    for li, lc in enumerate(clusters):
        if lc.get('vertical'): continue
        if lc.get('_multi_line'): continue
        ltext = (lc.get('text') or '').strip()
        if not ltext or not ltext.rstrip().endswith('-'):
            continue
        # If the left text already contains a space after stripping the
        # trailing hyphen, it's two hyphen-ending words on one line (e.g.
        # 'Bloch- Interfero-' which is actually 'Bloch-' + gap + 'Interfero-'
        # belonging to two SEPARATE labels). Pairing just one bottom-line
        # cluster with both gives a wrong merge — skip.
        if ' ' in ltext.rstrip('-').rstrip():
            continue
        lx0, ly0, lx1, ly1 = lc['bbox']
        lh = ly1 - ly0
        lw = lx1 - lx0
        best = None
        for ri, rc in enumerate(clusters):
            if ri == li or rc.get('vertical'): continue
            rtext = (rc.get('text') or '').strip()
            if not rtext: continue
            rx0, ry0, rx1, ry1 = rc['bbox']
            gap = ry0 - ly1
            if gap < -2 or gap > lh * 1.5:
                continue
            ox0, ox1 = max(lx0, rx0), min(lx1, rx1)
            if ox1 <= ox0:
                continue
            overlap = ox1 - ox0
            rw = rx1 - rx0
            if overlap < 0.5 * min(lw, rw):
                continue
            score = abs(gap) + abs(((rx0 + rx1) / 2) - ((lx0 + lx1) / 2))
            if best is None or score < best[0]:
                best = (score, ri, rc, rx0, ry0, rx1, ry1)
        if not best:
            continue
        _, ri, rc, rx0, ry0, rx1, ry1 = best
        rtext = (rc.get('text') or '').strip()
        merged_text = f'{ltext}\n{rtext}'
        merged_bbox = [min(lx0, rx0), ly0, max(lx1, rx1), ry1]
        # Optional Claude re-verify: re-read the merged region at high zoom
        # so a line-by-line misread (e.g. 'meters' → 'Messiah') gets corrected.
        if img is not None:
            try:
                import cv2 as _cv2
                H, W = img.shape[:2]
                bx0, by0, bx1, by1 = merged_bbox
                px = max(8, int((bx1 - bx0) * 0.05))
                py = max(4, int((by1 - by0) * 0.1))
                cx0 = max(0, int(bx0) - px); cy0 = max(0, int(by0) - py)
                cx1 = min(W, int(bx1) + px); cy1 = min(H, int(by1) + py)
                crop = img[cy0:cy1, cx0:cx1]
                if crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 12:
                    if crop.shape[0] < 60:
                        sf = 60 / crop.shape[0]
                        crop = _cv2.resize(crop,
                                           (int(crop.shape[1] * sf), int(crop.shape[0] * sf)),
                                           interpolation=_cv2.INTER_CUBIC)
                    fixed = _claude_fix_text(crop, merged_text.replace('\n', ' '))
                    if fixed:
                        f_alnum = ''.join(ch for ch in fixed.lower() if ch.isalnum())
                        l_alnum = ''.join(ch for ch in ltext.lower() if ch.isalnum())
                        # Accept if the merged-Claude reading shares the left
                        # prefix (which is usually the well-OCR'd part) — that
                        # guards against Claude swapping in unrelated words.
                        if l_alnum and l_alnum[:max(3, len(l_alnum)//2)] in f_alnum:
                            # Reflow the Claude reading back into 2 lines so
                            # the multi-line render still wraps at the hyphen.
                            if '-' in fixed and '\n' not in fixed:
                                # break after the LAST hyphen (e.g.
                                # 'Bloch-Interferometers' → keep prefix on one
                                # line, suffix on the next)
                                idx = fixed.rfind('-')
                                if 0 < idx < len(fixed) - 1:
                                    fixed = fixed[:idx + 1] + '\n' + fixed[idx + 1:]
                            merged_text = fixed
            except Exception:
                pass
        lc['text'] = merged_text
        lc['bbox'] = merged_bbox
        lp = set(lc.get('glyph_path_ids', []))
        rp = set(rc.get('glyph_path_ids', []))
        lc['glyph_path_ids'] = lp | rp
        lc['_multi_line'] = True
        rc['text'] = ''
        n += 1
    return n


def recover_inter_cluster_chars(clusters: list, png_path: str,
                                max_gap: float = 30.0) -> int:
    """Find pairs of horizontal clusters on the same y-row separated by a
    small x-gap, and check if a connector character (like '&', ':', '-')
    sits between them that EasyOCR missed. If a dark glyph candidate sits
    in the gap, ask Claude to read the merged crop and replace the left
    cluster's text with the corrected merged text. The right cluster is
    then cleared so it doesn't double-render.

    Returns the number of pairs successfully merged.
    """
    try:
        import cv2
    except ImportError:
        return 0
    img = cv2.imread(png_path)
    if img is None:
        return 0
    H, W = img.shape[:2]

    horizontals = []
    for i, c in enumerate(clusters):
        if c.get('vertical'):
            continue
        text = (c.get('text') or '').strip()
        if not text:
            continue
        bb = c['bbox']
        horizontals.append((i, c, bb, text))

    n = 0
    used = set()
    for li, lc, lbb, ltext in horizontals:
        if li in used:
            continue
        lx0, ly0, lx1, ly1 = lbb
        lyc = (ly0 + ly1) / 2
        lh = ly1 - ly0
        # Find a right neighbor on the same baseline
        for ri, rc, rbb, rtext in horizontals:
            if ri == li or ri in used:
                continue
            rx0, ry0, rx1, ry1 = rbb
            ryc = (ry0 + ry1) / 2
            if abs(ryc - lyc) > max(3, lh * 0.4):
                continue
            if rx0 <= lx1:
                continue
            gap = rx0 - lx1
            if gap > max_gap or gap < 2:
                continue
            # Same baseline, small gap → could be a missing connector
            x0 = max(0, int(lx0) - 4); y0 = max(0, int(min(ly0, ry0)) - 4)
            x1 = min(W, int(rx1) + 4); y1 = min(H, int(max(ly1, ry1)) + 4)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if crop.shape[0] < 28:
                sf = 28 / max(1, crop.shape[0])
                crop = cv2.resize(crop,
                                  (int(crop.shape[1]*sf), int(crop.shape[0]*sf)),
                                  interpolation=cv2.INTER_CUBIC)
            candidate = f'{ltext} {rtext}'
            fixed = _claude_fix_text(crop, candidate)
            if not fixed:
                continue
            # Only accept if Claude looks like it actually filled in a connector:
            # the merged text must STARTSWITH ltext, ENDSWITH rtext, with 1–5
            # chars between (e.g. ' & ', ': ', ' - '). This rejects Claude
            # hallucinations that swap the right-hand label for a different
            # word (e.g. 'Natural-Language Frontend' → 'Natural-Language
            # Processing') because the result wouldn't end with rtext.
            if not (fixed.startswith(ltext) and fixed.endswith(rtext)):
                continue
            between = fixed[len(ltext):len(fixed) - len(rtext)]
            # Require Claude to have ADDED a real punctuation connector. Whitelist
            # the chars we expect ('&', ':', '-', ',', ';', '/', '+', '*'); anything
            # else (Unicode emoji, letters, etc.) is rejected. This stops Claude
            # from hallucinating a one-char filler like '☺' between two labels
            # that share a baseline by coincidence.
            allowed = set('&:-,;/+*')
            between_clean = between.strip()
            if not (1 <= len(between_clean) <= 3):
                continue
            if not all(ch in allowed for ch in between_clean):
                continue
            lc['text'] = fixed
            lc['source'] = 'merged_inter_cluster'
            # Expand left cluster bbox to cover the merged region
            lc['bbox'] = [min(lx0, rx0), min(ly0, ry0), max(lx1, rx1), max(ly1, ry1)]
            rc['text'] = ''
            used.add(li); used.add(ri)
            n += 1
            break
    return n


def recover_missing_chars(clusters: list, png_path: str,
                          letter_ratio: float = 1.25) -> int:
    """For horizontal clusters where the SVG glyph count exceeds the OCR text
    letter count by more than `letter_ratio`, re-read the cluster's bbox crop
    with Claude. Fixes cases where EasyOCR splits a label into two boxes and
    drops a small connector character (e.g. '&' between 'Ranked Candidates'
    and 'Surgery Scores').

    Returns number of clusters whose text was replaced.
    """
    try:
        import cv2
    except ImportError:
        return 0
    img = cv2.imread(png_path)
    if img is None:
        return 0
    H, W = img.shape[:2]
    n = 0
    for c in clusters:
        if c.get('vertical'):
            continue
        text = (c.get('text') or '').strip()
        if not text:
            continue
        letters = sum(1 for ch in text if ch.isalnum())
        n_glyphs = len(c.get('glyph_path_ids', []))
        if n_glyphs == 0 or letters == 0:
            continue
        if n_glyphs <= letters * letter_ratio:
            continue
        x0, y0, x1, y1 = c['bbox']
        h_extra = max(8, int((y1 - y0) * 0.3))
        w_extra = max(8, int((x1 - x0) * 0.05))
        cx0 = max(0, int(x0) - w_extra); cy0 = max(0, int(y0) - h_extra)
        cx1 = min(W, int(x1) + w_extra); cy1 = min(H, int(y1) + h_extra)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        if crop.shape[0] < 32:
            sf = 32 / max(1, crop.shape[0])
            crop = cv2.resize(crop, (int(crop.shape[1]*sf), int(crop.shape[0]*sf)),
                              interpolation=cv2.INTER_CUBIC)
        fixed = _claude_fix_text(crop, text)
        if fixed and fixed != text and len(fixed) >= len(text):
            c['text'] = fixed
            c['source'] = 'recovered_missing_chars'
            n += 1
    return n


def _claude_fix_text(crop_bgr, candidate: str, api_key: str | None = None) -> str | None:
    """Send a rotated crop + an EasyOCR candidate to Claude and ask for the
    corrected reading. Returns None on any failure (so callers can fall back
    to the EasyOCR text). Skips silently when no credential is available.
    """
    try:
        import io, base64
        from PIL import Image
    except ImportError:
        return None
    client = make_anthropic_client()
    if client is None:
        return None
    try:
        pil = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR→RGB
        if pil.height < 60:
            sf = 60 / pil.height
            pil = pil.resize((int(pil.width * sf), int(pil.height * sf)),
                             Image.LANCZOS)
        buf = io.BytesIO(); pil.save(buf, format='PNG')
        b64 = base64.standard_b64encode(buf.getvalue()).decode('ascii')
        prompt = (
            'The image shows a small piece of text from a scientific figure. '
            f'A previous OCR pass read it as: "{candidate}". '
            'That reading may have typos. Reply with the correct reading, '
            'EXACTLY as it appears, no quotes, no explanation. If unreadable, '
            'reply with the previous reading unchanged.'
        )
        msg = client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=80,
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': 'image/png', 'data': b64}},
                {'type': 'text', 'text': prompt},
            ]}],
        )
        text = msg.content[0].text.strip().strip('"\'')
        return text or None
    except Exception:
        return None


def recover_vertical_with_crop(clusters: list, png_path: str,
                               min_glyphs: int = 4,
                               use_claude_fix: bool = True) -> int:
    """For vertical clusters with no OCR text but enough glyphs to plausibly
    contain a real word, run EasyOCR on a rotated crop of the cluster region.
    EasyOCR's full-image pass usually misses rotated text; running it on a
    tightly-cropped, pre-rotated image gives a second chance.

    If ANTHROPIC_API_KEY is set and use_claude_fix=True, the EasyOCR result
    is sent to Claude with the crop image to correct typos (e.g. 'Phpsical
    Qub' → 'Physical-Qubit').

    Returns the number of clusters that recovered a non-empty text.
    """
    targets = [c for c in clusters
               if c.get('vertical')
               and not (c.get('text') or '').strip()
               and len(c.get('glyph_path_ids', [])) >= min_glyphs]
    if not targets:
        return 0
    try:
        import cv2
        import easyocr  # noqa
    except ImportError:
        return 0
    img = cv2.imread(png_path)
    if img is None:
        return 0
    H, W = img.shape[:2]
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    n = 0
    for c in targets:
        x0, y0, x1, y1 = c['bbox']
        # Use generous padding so we don't miss letters that the cluster
        # bbox under-shot. Asymmetric: more in the long-axis direction
        # (vertical text reads bottom-to-top, so extend in y) so we capture
        # truncated head/tail letters.
        h_extra = max(20, int((y1 - y0) * 0.4))
        w_extra = max(6, int((x1 - x0) * 0.5))
        cx0 = max(0, int(x0) - w_extra); cy0 = max(0, int(y0) - h_extra)
        cx1 = min(W, int(x1) + w_extra); cy1 = min(H, int(y1) + h_extra)
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            continue
        # Rotate 90° clockwise so the rotated text reads horizontally.
        crop_r = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        # Upscale tiny crops for better OCR
        if crop_r.shape[0] < 32:
            sf = 32 / max(1, crop_r.shape[0])
            crop_r = cv2.resize(crop_r,
                                (int(crop_r.shape[1] * sf), int(crop_r.shape[0] * sf)),
                                interpolation=cv2.INTER_CUBIC)
        try:
            raw = reader.readtext(crop_r, detail=1, paragraph=False,
                                  decoder='beamsearch', beamWidth=8)
        except Exception:
            continue
        if not raw:
            continue
        # Pick the highest-confidence reading; require ≥3 chars
        best = max(raw, key=lambda r: float(r[2]))
        text = best[1].strip()
        conf = float(best[2])
        if len(text) >= 3 and conf >= 0.3:
            if use_claude_fix:
                fixed = _claude_fix_text(crop_r, text)
                if fixed and fixed != text:
                    text = fixed
            c['text'] = text
            c['conf'] = conf
            c['source'] = 'recovered_vertical_crop'
            n += 1
    return n


def reject_hallucinated_text(clusters: list) -> int:
    """Clear OCR text when the cluster has too few glyphs to plausibly contain
    that many letters. Claude's verifier sometimes invents words from icon
    decorations (e.g. an 'AI' chip with 3 glyphs gets read as 'meters').

    Heuristic: if a horizontal cluster has ≤3 glyphs but the OCR text has
    ≥5 visible characters, the text was almost certainly hallucinated.
    Returns the number of cleared clusters.
    """
    n = 0
    for c in clusters:
        if c.get('vertical'):
            continue
        text = (c.get('text') or '').strip()
        if not text:
            continue
        n_glyphs = len(c.get('glyph_path_ids', []))
        # Count visible letters (ignore spaces/punct in length budget)
        letters = sum(1 for ch in text if ch.isalnum())
        if n_glyphs <= 3 and letters >= 5:
            c['text'] = ''
            n += 1
    return n


def clean_suspect_vertical_ocr(clusters: list) -> int:
    """Drop OCR text from vertical clusters when the text looks like a
    single-glyph false positive — EasyOCR often returns 'p' / '1' / '!' for
    rotated multi-letter labels it couldn't actually read. Better to render
    nothing than the wrong letter (which the user sees as 'text disappeared').

    Returns the number of cleared clusters.
    """
    n = 0
    for c in clusters:
        if not c.get('vertical'):
            continue
        text = (c.get('text') or '').strip()
        if not text:
            continue
        n_glyphs = len(c.get('glyph_path_ids', []))
        # Cluster has 4+ glyphs (at least 4 letters worth of ink) but OCR only
        # gave 1-2 chars — almost always a misread of rotated text.
        if n_glyphs >= 4 and len(text) <= 2:
            c['text'] = ''
            n += 1
    return n


def _text_similar(a: str, b: str, ratio: float = 0.7) -> bool:
    """Two texts are 'similar' when their alphanumeric forms match closely.
    Used to skip OCR-noise variants (e.g. 'Hubbard' vs 'hubbard.', 'AI' vs 'Al')
    that aren't really separate labels.
    """
    an = ''.join(ch for ch in a.lower() if ch.isalnum())
    bn = ''.join(ch for ch in b.lower() if ch.isalnum())
    if not an or not bn:
        return False
    if an == bn or an in bn or bn in an:
        return True
    return difflib.SequenceMatcher(None, an, bn).ratio() > ratio


def split_multilabel_clusters(clusters, full_ocr, path_by_idx):
    """Detect clusters whose glyphs span multiple labels on one baseline and
    split each spillover region into its own cluster, claiming an unclaimed
    OCR box.

    Heuristic for a true multi-label split:
      - There's an unclaimed OCR box (not text-similar to the cluster's own
        text) on the same y baseline within the cluster's x range.
      - At least 2 of the cluster's glyphs fall inside the OCR box's x range.

    Returns (clusters_extended, n_split). Modifies the input clusters in
    place (removes split-off glyphs and shrinks bbox).
    """
    if not full_ocr:
        return clusters, 0

    # Map each cluster's text to its source OCR box (claim it so we don't
    # consider it a candidate for split). Skip text-similar variants too.
    used_ocr_idxs = set()
    for c in clusters:
        ctext = c.get('text', '')
        if not ctext:
            continue
        cbb = c['bbox']
        best_oi, best_ov = None, 0
        for oi, ob in enumerate(full_ocr):
            if oi in used_ocr_idxs:
                continue
            if ob.get('text', '').strip() != ctext:
                continue
            ox0, oy0, ox1, oy1 = ob['x1'], ob['y1'], ob['x2'], ob['y2']
            ix0 = max(cbb[0], ox0); iy0 = max(cbb[1], oy0)
            ix1 = min(cbb[2], ox1); iy1 = min(cbb[3], oy1)
            if ix1 > ix0 and iy1 > iy0:
                ov = (ix1 - ix0) * (iy1 - iy0)
                if ov > best_ov:
                    best_ov, best_oi = ov, oi
        if best_oi is not None:
            used_ocr_idxs.add(best_oi)

    new_clusters = []
    for c in clusters:
        ctext = c.get('text', '')
        if not ctext:
            continue
        cbb = c['bbox']
        cx0, cy0, cx1, cy1 = cbb
        cyc = (cy0 + cy1) / 2
        ch = cy1 - cy0
        for oi, ob in enumerate(full_ocr):
            if oi in used_ocr_idxs:
                continue
            otext = ob.get('text', '').strip()
            if not otext:
                continue
            if _text_similar(ctext, otext):
                continue
            ox0, oy0, ox1, oy1 = ob['x1'], ob['y1'], ob['x2'], ob['y2']
            oyc = (oy0 + oy1) / 2
            if abs(cyc - oyc) > max(4, ch * 0.7):
                continue
            if ox0 < cx0 - 5 or ox1 > cx1 + 5:
                continue
            gids = list(c.get('glyph_path_ids', []))
            glyphs_in_ocr = []
            for gid in gids:
                p = path_by_idx.get(gid)
                if not p:
                    continue
                bb = p['bbox']
                gxc = (bb[0] + bb[2]) / 2
                gyc = (bb[1] + bb[3]) / 2
                if abs(gyc - oyc) > ch and abs(gyc - cyc) > ch:
                    continue
                if ox0 - 3 <= gxc <= ox1 + 3:
                    glyphs_in_ocr.append((gid, bb))
            if len(glyphs_in_ocr) < 2:
                continue

            new_glyph_ids = [g[0] for g in glyphs_in_ocr]
            xs = [g[1][0] for g in glyphs_in_ocr] + [g[1][2] for g in glyphs_in_ocr]
            ys = [g[1][1] for g in glyphs_in_ocr] + [g[1][3] for g in glyphs_in_ocr]
            was_set = isinstance(c.get('glyph_path_ids'), set)
            new_clusters.append({
                'bbox': [min(xs), min(ys), max(xs), max(ys)],
                'font_size': c.get('font_size', 9),
                'bold': c.get('bold', False),
                'ink_ratio': c.get('ink_ratio', 0.5),
                'glyph_path_ids': set(new_glyph_ids) if was_set else new_glyph_ids,
                'num_glyphs': len(new_glyph_ids),
                'vertical': False,
                'text': otext,
                'conf': ob.get('conf', 0.95),
                'source': 'split_from_multilabel',
            })
            used_ocr_idxs.add(oi)

            kept = [gid for gid in gids if gid not in set(new_glyph_ids)]
            c['glyph_path_ids'] = set(kept) if was_set else kept
            c['num_glyphs'] = len(kept)
            if kept:
                xs0, ys0_, xs1, ys1_ = [], [], [], []
                for gid in kept:
                    p = path_by_idx.get(gid)
                    if not p:
                        continue
                    xs0.append(p['bbox'][0])
                    ys0_.append(p['bbox'][1])
                    xs1.append(p['bbox'][2])
                    ys1_.append(p['bbox'][3])
                if xs0:
                    c['bbox'] = [min(xs0), min(ys0_), max(xs1), max(ys1_)]
    if new_clusters:
        clusters.extend(new_clusters)
    return clusters, len(new_clusters)
