"""svg_to_drawio_v17.py — DEPRECATED entry point.

Real implementation lives in the `svg_to_drawio/` package. This shim is kept
so that legacy scripts (and shell history) calling `python svg_to_drawio_v17.py`
keep working.

Use the new entry point:
    python -m svg_to_drawio <svg> <png> -o out.drawio [opts]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from svg_to_drawio.convert import convert_pair as convert  # re-export

__all__ = ['convert']


if __name__ == '__main__':
    from svg_to_drawio.__main__ import main
    main()
