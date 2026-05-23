"""Visual primitive program utilities for editable diagram reconstruction."""

from .schema import load_ledger, save_program, to_visual_primitive_program
from .emit_drawio import compile_program_to_drawio
from .patches import apply_patch_plan, load_patch
from .cv_tools import draw_cv_overlay, extract_cv_primitives, save_cv_evidence
from .cv_snap import snap_program_to_cv_evidence
from .cv_augment import augment_program_from_cv_evidence

__all__ = [
    "augment_program_from_cv_evidence",
    "apply_patch_plan",
    "compile_program_to_drawio",
    "draw_cv_overlay",
    "extract_cv_primitives",
    "load_ledger",
    "load_patch",
    "save_cv_evidence",
    "save_program",
    "snap_program_to_cv_evidence",
    "to_visual_primitive_program",
]
