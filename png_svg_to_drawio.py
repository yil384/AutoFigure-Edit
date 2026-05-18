"""png_svg_to_drawio.py — General PNG+SVG → drawio converter.

Single entry point that takes (input.png, input.svg) and produces a clean
drawio file with:
  - Native drawio text cells (font: DejaVu Sans default, configurable)
  - Atomic icons as single SVG image cells (indivisible in drawio)
  - Solid panel backgrounds (no holes, in correct z-order)
  - Transparent canvas background (locked invisible bounding rect)
  - All structural lines/arrows preserved as singleton stencils

Pipeline:
  1. Parse SVG paths
  2. Drop full-canvas white background
  3. Identify panel backgrounds → drop holes → emit as bottom layer
  4. Cluster glyph-shaped paths into text lines (SVG geometry → pixel-precise bbox)
  5. OCR for content:
     - EasyOCR full-image pass → mutex assignment (each OCR box → 1 cluster)
     - Claude Vision verify pass with confidence guards (fixes V/M, I/l, symbols)
  6. Cluster non-text paths into icons (spatial proximity, max 120×120)
  7. Absorb singleton paths near icons
  8. Color-quantize each icon (merge similar-color paths)
  9. Emit drawio:
       z=0: locked invisible canvas rect (sets page dimensions)
       z=1: solid panel backgrounds
       z=2: singleton stencils (long arrows, lines, dividers)
       z=3: atomic SVG icon cells
       z=4: drawio text cells

Usage:
  python png_svg_to_drawio.py input.svg input.png [-o output.drawio]
    [--cache-dir <dir>]           # caches OCR + cluster results
    [--font-family "DejaVu Sans"]
    [--no-claude]                 # skip Claude verify pass (uses EasyOCR only)
    [--quantize-threshold 30]
    [--max-icon-dim 120]
    [--anthropic-key <key>]       # or env ANTHROPIC_API_KEY
"""
from __future__ import annotations

import os, sys, argparse, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_drawio import convert_pair


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('svg', help='input SVG (vectorized version of the PNG)')
    ap.add_argument('png', help='input PNG/JPG (rasterized source for OCR/colors)')
    ap.add_argument('-o', '--output', default=None,
                    help='output drawio path (default: <svg>.drawio)')
    ap.add_argument('--cache-dir', default=None,
                    help='cache directory for OCR + cluster JSONs (speeds up reruns)')
    ap.add_argument('--font-family', default='DejaVu Sans',
                    help='fontFamily for drawio text cells (default: DejaVu Sans)')
    ap.add_argument('--quantize-threshold', type=float, default=30.0,
                    help='RGB ΔE76 distance for icon color merging (higher = simpler)')
    ap.add_argument('--max-icon-dim', type=float, default=120,
                    help='max bbox dimension (px) for icon clustering')
    ap.add_argument('--no-transparent-bg', action='store_true',
                    help='keep white canvas background')
    args = ap.parse_args()

    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.svg).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.svg).stem

    cluster_cache = cache_dir / f'{base_name}.clusters.json'
    ocr_cache = cache_dir / f'{base_name}.ocr.json'

    print(f'svg:    {args.svg}')
    print(f'png:    {args.png}')
    print(f'output: {out}')
    print(f'cache:  cluster={cluster_cache}  ocr={ocr_cache}')

    stats = convert_pair(
        svg_path=args.svg,
        png_path=args.png,
        drawio_path=out,
        cluster_cache=str(cluster_cache),
        ocr_cache=str(ocr_cache),
        font_family=args.font_family,
        max_icon_dim=args.max_icon_dim,
        quantize_threshold=args.quantize_threshold,
        transparent_bg=not args.no_transparent_bg,
    )
    print(f'\nstats: {json.dumps(stats, indent=2)}')


if __name__ == '__main__':
    main()
