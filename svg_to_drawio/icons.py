"""Icon clustering and atomic SVG image cells."""
from svg_to_drawio_v6 import bbox_overlap_or_near, bbox_union, cluster_icons
from svg_to_drawio_v7 import segments_to_svg_d, svg_to_image_cell_style
from svg_to_drawio_v9 import quantized_icon_svg
from svg_to_drawio_v11 import absorb_singletons

__all__ = [
    'bbox_overlap_or_near', 'bbox_union', 'cluster_icons',
    'segments_to_svg_d', 'svg_to_image_cell_style',
    'quantized_icon_svg', 'absorb_singletons',
]
