"""svg_to_drawio — convert (PNG, SVG) → drawio with native edges, atomic icons,
editable text cells, and solid panels.

Entry point: `svg_to_drawio.convert.convert_pair(svg, png, out, **opts)`.
"""
from svg_to_drawio.convert import convert_pair

__all__ = ['convert_pair']
