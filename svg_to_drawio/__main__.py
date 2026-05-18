"""CLI: python -m svg_to_drawio <svg> <png> -o out.drawio [--cluster-cache ...]"""
import argparse
import sys
from pathlib import Path

# Make legacy v* modules importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from svg_to_drawio.convert import convert_pair


def main():
    ap = argparse.ArgumentParser(prog='svg_to_drawio')
    ap.add_argument('svg')
    ap.add_argument('png')
    ap.add_argument('-o', '--output', default=None,
                    help='output .drawio path (default: <svg>.drawio)')
    ap.add_argument('--cluster-cache', default=None,
                    help='reuse cached glyph clusters JSON (skips clustering)')
    ap.add_argument('--ocr-cache', default=None,
                    help='reuse cached verified OCR JSON (skips OCR)')
    ap.add_argument('--font-family', default='DejaVu Sans')
    ap.add_argument('--max-icon-dim', type=float, default=120)
    ap.add_argument('--quantize', type=float, default=15.0,
                    help='RGB color-quantization threshold (ΔE76); lower = more color fidelity')
    args = ap.parse_args()
    out = args.output or str(Path(args.svg).with_suffix('.drawio'))
    convert_pair(args.svg, args.png, out,
                 cluster_cache=args.cluster_cache,
                 ocr_cache=args.ocr_cache,
                 font_family=args.font_family,
                 max_icon_dim=args.max_icon_dim,
                 quantize_threshold=args.quantize)


if __name__ == '__main__':
    main()
