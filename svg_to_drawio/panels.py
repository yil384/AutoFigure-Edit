"""Panel background detection and outer-only subpath conversion."""
from svg_to_drawio_v12 import is_full_canvas_background
from svg_to_drawio_v14 import is_panel_background, keep_outer_subpath_only

__all__ = [
    'is_full_canvas_background', 'is_panel_background', 'keep_outer_subpath_only',
]
