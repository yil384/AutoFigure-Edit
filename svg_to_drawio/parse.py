"""SVG path parsing and stencil encoding.

Re-exports from the legacy v2/v4/v5 modules (kept for now to avoid double code).
"""
from svg_to_drawio_v2 import (
    parse_path, expand_path, path_bbox, segments_to_stencil, encode_stencil,
    _strip_ns,
)
from svg_to_drawio_v4 import split_subpaths, subpath_bbox
from svg_to_drawio_v5 import parse_svg_paths

__all__ = [
    'parse_path', 'expand_path', 'path_bbox', 'segments_to_stencil',
    'encode_stencil', '_strip_ns',
    'split_subpaths', 'subpath_bbox',
    'parse_svg_paths',
]
