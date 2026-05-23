#!/usr/bin/env python3
"""Generate, render, and rank CV-constrained pure-native draw.io variants."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from png_to_drawio import convert_png_to_drawio  # noqa: E402
from visual_primitives.cv_augment import augment_program_from_cv_evidence  # noqa: E402
from visual_primitives.cv_snap import snap_program_to_cv_evidence  # noqa: E402
from visual_primitives.cv_tools import (  # noqa: E402
    draw_cv_overlay,
    extract_cv_primitives,
    save_cv_evidence,
)
from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import (  # noqa: E402
    load_ledger,
    save_program,
    to_visual_primitive_program,
)
from visual_primitives.text_cleanup import (  # noqa: E402
    remove_contained_duplicate_text,
    remove_noise_text_fragments,
)
from visual_primitives.text_render_tune import tune_text_rendering  # noqa: E402
from visual_primitives.unsupported_prune import prune_unsupported_edges  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
    tile_winners,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run an automatic CV variant search for pure-native draw.io")
    ap.add_argument("image", help="input PNG/JPG/WebP/TIFF")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives",
                    help="directory for generated artifacts")
    ap.add_argument("--name", default=None,
                    help="artifact stem; defaults to input stem")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--ocr-conf", type=float, default=35.0)
    ap.add_argument("--trusted-text-conf", type=float, default=55.0)
    ap.add_argument("--ocr-scale", type=float, default=2.0)
    ap.add_argument("--ocr-psm", type=int, default=11)
    ap.add_argument("--ocr-multipass", action="store_true")
    ap.add_argument("--cv-max-lines", type=int, default=900)
    ap.add_argument("--cv-max-lines-values", default=None,
                    help="comma-separated sweep values; overrides --cv-max-lines")
    ap.add_argument("--cv-max-regions", type=int, default=240)
    ap.add_argument("--max-augment-lines", type=int, default=70)
    ap.add_argument("--augment-lines-values", default=None,
                    help="comma-separated line augment sweep values; 0 means disabled")
    ap.add_argument("--include-text-augment", action="store_true",
                    help="include a risky text-augment candidate in the search")
    ap.add_argument("--include-text-only-augment", action="store_true",
                    help="include text augmentation without adding extra CV lines")
    ap.add_argument("--augment-text-min-conf", type=float, default=0.72,
                    help="minimum OCR confidence for text augmentation")
    ap.add_argument("--include-text-render-sweep", action="store_true",
                    help="add generic text rendering variants for text-augment candidates")
    ap.add_argument("--clean-noise-text", action="store_true",
                    help="remove tiny standalone OCR noise fragments before snapping")
    ap.add_argument("--text-render-font-scales", default="1.12",
                    help="comma-separated font scale values for text rendering sweep")
    ap.add_argument("--text-render-bbox-pads", default="1",
                    help="comma-separated bbox pad values for text rendering sweep")
    ap.add_argument("--text-render-bold-modes", default="none",
                    help="comma-separated bold modes: none,headers,all")
    ap.add_argument("--text-render-font-families", default="Helvetica",
                    help="comma-separated font families for text rendering sweep")
    ap.add_argument("--include-contour-paths", action="store_true",
                    help="include pure-native icon contour path candidates")
    ap.add_argument("--include-unsupported-prune", action="store_true",
                    help="include source-supported edge pruning candidates")
    ap.add_argument("--unsupported-prune-thresholds", default="0.70",
                    help="comma-separated edge support thresholds")
    ap.add_argument("--unsupported-prune-modes", default="all_cv",
                    help="comma-separated prune modes: augment,all_cv,all_edges")
    ap.add_argument("--unsupported-prune-mask-dilate", type=int, default=5)
    ap.add_argument("--unsupported-prune-stroke-pad", type=float, default=2.0)
    ap.add_argument("--unsupported-prune-min-length", type=float, default=12.0)
    ap.add_argument("--tiles", action="store_true",
                    help="compute quadrant-level metric winners")
    ap.add_argument("--generate-only", action="store_true",
                    help="generate variants but do not export/rank them")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    image = Path(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or image.stem
    base = out_dir / stem

    native_drawio = Path(f"{base}.native_raw.drawio")
    ledger_path = Path(f"{base}.primitive_ledger.json")

    counts = convert_png_to_drawio(
        input_path=str(image),
        output_path=str(native_drawio),
        show_overlay=True,
        include_background=False,
        font_family="Arial",
        ocr_conf=args.ocr_conf,
        trusted_text_conf=args.trusted_text_conf,
        ocr_scale=args.ocr_scale,
        ocr_psm=args.ocr_psm,
        ocr_multipass=args.ocr_multipass,
        detect_arrows=True,
        emit_skeleton_connectors=True,
        emit_native_shapes=True,
        emit_contour_paths=False,
        emit_icon_crops=False,
        emit_visual_foreground=False,
        show_visual_foreground=False,
        native_overlay_visible=True,
        pure_native=True,
        ledger_path=str(ledger_path),
    )

    ledger = load_ledger(ledger_path)
    base_program = to_visual_primitive_program(ledger)
    base_program, text_cleanup_report = remove_contained_duplicate_text(base_program)
    if args.clean_noise_text:
        base_program, noise_cleanup_report = remove_noise_text_fragments(base_program)
    else:
        noise_cleanup_report = _skipped_report("noise text cleanup disabled")
    save_program(base_program, Path(f"{base}.native_raw.vp_program.json"))

    program_specs: list[dict[str, Any]] = [{
        "prefix": "",
        "raw_name": "native_raw",
        "program": base_program,
        "report": {
            "source": "raw_native_program",
            "text_cleanup": text_cleanup_report,
            "noise_text_cleanup": noise_cleanup_report,
        },
    }]
    if args.include_contour_paths:
        contour_drawio = Path(f"{base}.contour_seed.drawio")
        contour_ledger_path = Path(f"{base}.contour_primitive_ledger.json")
        contour_counts = convert_png_to_drawio(
            input_path=str(image),
            output_path=str(contour_drawio),
            show_overlay=True,
            include_background=False,
            font_family="Arial",
            ocr_conf=args.ocr_conf,
            trusted_text_conf=args.trusted_text_conf,
            ocr_scale=args.ocr_scale,
            ocr_psm=args.ocr_psm,
            ocr_multipass=args.ocr_multipass,
            detect_arrows=True,
            emit_skeleton_connectors=True,
            emit_native_shapes=True,
            emit_contour_paths=True,
            emit_icon_crops=False,
            emit_visual_foreground=False,
            show_visual_foreground=False,
            native_overlay_visible=True,
            pure_native=True,
            ledger_path=str(contour_ledger_path),
        )
        contour_program = to_visual_primitive_program(load_ledger(contour_ledger_path))
        contour_program, contour_cleanup_report = remove_contained_duplicate_text(contour_program)
        if args.clean_noise_text:
            contour_program, contour_noise_cleanup_report = remove_noise_text_fragments(contour_program)
        else:
            contour_noise_cleanup_report = _skipped_report("noise text cleanup disabled")
        save_program(contour_program, Path(f"{base}.contour_raw.vp_program.json"))
        program_specs.append({
            "prefix": "contour_",
            "raw_name": "contour_raw",
            "program": contour_program,
            "report": {
                "source": "raw_native_program_with_contour_paths",
                "native_counts": contour_counts,
                "text_cleanup": contour_cleanup_report,
                "noise_text_cleanup": contour_noise_cleanup_report,
            },
        })

    variants: list[dict[str, Any]] = []
    for spec in program_specs:
        variants.append(_write_variant(
            name=spec["raw_name"],
            stem=base,
            program=spec["program"],
            report=spec["report"],
        ))

    cv_line_values = _parse_int_values(
        args.cv_max_lines_values,
        fallback=[700, args.cv_max_lines, 1200],
    )
    augment_line_values = _parse_int_values(
        args.augment_lines_values,
        fallback=[0, 40, args.max_augment_lines],
    )
    evidence_sets = []
    primary_cv_path = None
    primary_cv_overlay = None
    for cv_max_lines in cv_line_values:
        tag = f"cv_l{cv_max_lines:04d}"
        cv_path = Path(f"{base}.{tag}.cv_primitives.json")
        cv_overlay = Path(f"{base}.{tag}.cv_overlay.png")
        evidence = extract_cv_primitives(
            image,
            max_lines=cv_max_lines,
            max_regions=args.cv_max_regions,
        )
        save_cv_evidence(evidence, cv_path)
        draw_cv_overlay(image, evidence, cv_overlay)
        if primary_cv_path is None:
            primary_cv_path = cv_path
            primary_cv_overlay = cv_overlay
        evidence_sets.append({
            "tag": tag,
            "cv_max_lines": cv_max_lines,
            "cv_evidence": str(cv_path),
            "cv_overlay": str(cv_overlay),
            "cv_counts": evidence.get("counts", {}),
        })

        for spec in program_specs:
            prefix = spec["prefix"]
            snap_program, snap_report = snap_program_to_cv_evidence(
                spec["program"], evidence)
            variants.append(_write_variant(
                name=f"{prefix}{tag}_snap",
                stem=base,
                program=snap_program,
                report={
                    "program_source": spec["raw_name"],
                    "cv_config": {"cv_max_lines": cv_max_lines},
                    "snap": snap_report,
                },
            ))

            for max_aug_lines in augment_line_values:
                if max_aug_lines <= 0:
                    continue
                line_aug_program, line_aug_report = augment_program_from_cv_evidence(
                    snap_program,
                    evidence,
                    add_text=False,
                    max_lines=max_aug_lines,
                )
                variants.append(_write_variant(
                    name=f"{prefix}{tag}_lineaug{max_aug_lines:03d}",
                    stem=base,
                    program=line_aug_program,
                    report={
                        "program_source": spec["raw_name"],
                        "cv_config": {"cv_max_lines": cv_max_lines},
                        "snap": snap_report,
                        "augment": line_aug_report,
                    },
                ))

            if args.include_text_augment:
                text_aug_program, text_aug_report = augment_program_from_cv_evidence(
                    snap_program,
                    evidence,
                    add_text=True,
                    max_lines=args.max_augment_lines,
                    min_text_confidence=args.augment_text_min_conf,
                )
                variants.append(_write_variant(
                    name=f"{prefix}{tag}_textaugment",
                    stem=base,
                    program=text_aug_program,
                    report={
                        "program_source": spec["raw_name"],
                        "cv_config": {"cv_max_lines": cv_max_lines},
                        "snap": snap_report,
                        "augment": text_aug_report,
                    },
                ))
            if args.include_text_only_augment:
                text_only_aug_program, text_only_aug_report = augment_program_from_cv_evidence(
                    snap_program,
                    evidence,
                    add_text=True,
                    add_lines=False,
                    min_text_confidence=args.augment_text_min_conf,
                )
                variants.append(_write_variant(
                    name=f"{prefix}{tag}_textonlyaugment",
                    stem=base,
                    program=text_only_aug_program,
                    report={
                        "program_source": spec["raw_name"],
                        "cv_config": {"cv_max_lines": cv_max_lines},
                        "snap": snap_report,
                        "augment": text_only_aug_report,
                    },
                ))

    if args.include_text_render_sweep:
        variants = _append_text_render_variants(
            variants,
            stem=base,
            font_scales=_parse_float_values(args.text_render_font_scales),
            bbox_pads=_parse_float_values(args.text_render_bbox_pads),
            bold_modes=_parse_str_values(args.text_render_bold_modes),
            font_families=_parse_str_values(args.text_render_font_families),
        )
    if args.include_unsupported_prune:
        variants = _append_unsupported_prune_variants(
            variants,
            stem=base,
            source_image=image,
            thresholds=_parse_float_values(args.unsupported_prune_thresholds),
            modes=_parse_str_values(args.unsupported_prune_modes),
            mask_dilate=args.unsupported_prune_mask_dilate,
            stroke_pad=args.unsupported_prune_stroke_pad,
            min_length=args.unsupported_prune_min_length,
        )

    for variant in variants:
        compile_program_to_drawio(
            variant["program"],
            variant["drawio"],
            font_family=variant.get("font_family", "Arial"),
        )
        save_program(variant["program"], variant["program_path"])
        variant["report_path"].write_text(json.dumps(
            variant["report"], indent=2, ensure_ascii=True))

    manifest_path = Path(f"{base}.variant_manifest.json")
    manifest = {
        "source_image": str(image),
        "native_counts": counts,
        "cv_sweep": {
            "cv_max_lines_values": cv_line_values,
            "augment_lines_values": augment_line_values,
            "cv_max_regions": args.cv_max_regions,
            "include_contour_paths": args.include_contour_paths,
            "augment_text_min_conf": args.augment_text_min_conf,
            "include_text_only_augment": args.include_text_only_augment,
            "include_text_render_sweep": args.include_text_render_sweep,
            "clean_noise_text": args.clean_noise_text,
            "include_unsupported_prune": args.include_unsupported_prune,
            "text_render_font_scales": _parse_float_values(args.text_render_font_scales),
            "text_render_bbox_pads": _parse_float_values(args.text_render_bbox_pads),
            "text_render_bold_modes": _parse_str_values(args.text_render_bold_modes),
            "text_render_font_families": _parse_str_values(args.text_render_font_families),
            "unsupported_prune_thresholds": _parse_float_values(args.unsupported_prune_thresholds),
            "unsupported_prune_modes": _parse_str_values(args.unsupported_prune_modes),
            "unsupported_prune_mask_dilate": args.unsupported_prune_mask_dilate,
            "unsupported_prune_stroke_pad": args.unsupported_prune_stroke_pad,
            "unsupported_prune_min_length": args.unsupported_prune_min_length,
        },
        "cv_evidence": str(primary_cv_path) if primary_cv_path else None,
        "cv_overlay": str(primary_cv_overlay) if primary_cv_overlay else None,
        "evidence_sets": evidence_sets,
        "variants": [
            {
                "name": variant["name"],
                "drawio": str(variant["drawio"]),
                "program": str(variant["program_path"]),
                "report": str(variant["report_path"]),
                "font_family": variant.get("font_family", "Arial"),
            }
            for variant in variants
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    if args.generate_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "cv_evidence": str(primary_cv_path) if primary_cv_path else None,
            "cv_overlay": str(primary_cv_overlay) if primary_cv_overlay else None,
            "variants": [str(variant["drawio"]) for variant in variants],
        }, indent=2))
        return

    rows = evaluate_drawio_variants(
        image,
        [variant["drawio"] for variant in variants],
        drawio_cli=args.drawio_cli,
        export=True,
        include_tiles=args.tiles,
    )
    ranking_path = Path(f"{base}.variant_ranking.json")
    ranking = {
        "source_image": str(image),
        "native_counts": counts,
        "cv_sweep": {
            "cv_max_lines_values": cv_line_values,
            "augment_lines_values": augment_line_values,
            "cv_max_regions": args.cv_max_regions,
            "include_contour_paths": args.include_contour_paths,
            "augment_text_min_conf": args.augment_text_min_conf,
            "include_text_only_augment": args.include_text_only_augment,
            "include_text_render_sweep": args.include_text_render_sweep,
            "clean_noise_text": args.clean_noise_text,
            "include_unsupported_prune": args.include_unsupported_prune,
            "text_render_font_scales": _parse_float_values(args.text_render_font_scales),
            "text_render_bbox_pads": _parse_float_values(args.text_render_bbox_pads),
            "text_render_bold_modes": _parse_str_values(args.text_render_bold_modes),
            "text_render_font_families": _parse_str_values(args.text_render_font_families),
            "unsupported_prune_thresholds": _parse_float_values(args.unsupported_prune_thresholds),
            "unsupported_prune_modes": _parse_str_values(args.unsupported_prune_modes),
            "unsupported_prune_mask_dilate": args.unsupported_prune_mask_dilate,
            "unsupported_prune_stroke_pad": args.unsupported_prune_stroke_pad,
            "unsupported_prune_min_length": args.unsupported_prune_min_length,
        },
        "evidence_sets": evidence_sets,
        "winner": rows[0]["drawio"] if rows else None,
        "tile_winners": tile_winners(rows) if args.tiles else {},
        "variants": rows,
    }
    ranking_path.write_text(json.dumps(ranking, indent=2, ensure_ascii=True))

    best_drawio = Path(f"{base}.best.drawio")
    best_png = Path(str(best_drawio) + ".png")
    best_compare = best_drawio.with_suffix(best_drawio.suffix + ".compare.png")
    if rows:
        shutil.copyfile(rows[0]["drawio"], best_drawio)
        if rows[0].get("rendered_png"):
            shutil.copyfile(rows[0]["rendered_png"], best_png)
        if rows[0].get("compare_png"):
            shutil.copyfile(rows[0]["compare_png"], best_compare)

    print(json.dumps({
        "best_drawio": str(best_drawio) if rows else None,
        "best_png": str(best_png) if best_png.exists() else None,
        "best_compare": str(best_compare) if best_compare.exists() else None,
        "ranking": str(ranking_path),
        "manifest": str(manifest_path),
        "cv_evidence": str(primary_cv_path) if primary_cv_path else None,
        "cv_overlay": str(primary_cv_overlay) if primary_cv_overlay else None,
        "scores": [compact_score_row(row) for row in rows],
    }, indent=2))


def _write_variant(
    *,
    name: str,
    stem: Path,
    program: dict[str, Any],
    report: dict[str, Any],
    font_family: str = "Arial",
) -> dict[str, Any]:
    return {
        "name": name,
        "drawio": Path(f"{stem}.{name}.drawio"),
        "program_path": Path(f"{stem}.{name}.vp_program.json"),
        "report_path": Path(f"{stem}.{name}.report.json"),
        "program": copy.deepcopy(program),
        "report": report,
        "font_family": font_family,
    }


def _append_text_render_variants(
    variants: list[dict[str, Any]],
    *,
    stem: Path,
    font_scales: list[float],
    bbox_pads: list[float],
    bold_modes: list[str],
    font_families: list[str],
) -> list[dict[str, Any]]:
    out = list(variants)
    targets = [
        variant for variant in variants
        if "textaugment" in variant["name"] or "textonlyaugment" in variant["name"]
    ]
    for variant in targets:
        for font_family in font_families:
            safe_family = _safe_name(font_family)
            for bold_mode in bold_modes:
                for font_scale in font_scales:
                    for bbox_pad in bbox_pads:
                        tuned_program, tune_report = tune_text_rendering(
                            variant["program"],
                            font_scale=font_scale,
                            bbox_pad=bbox_pad,
                            bold_mode=bold_mode,
                        )
                        suffix = (
                            f"font_{safe_family}_b{bold_mode}"
                            f"_s{font_scale:.2f}_p{bbox_pad:g}"
                        )
                        out.append(_write_variant(
                            name=f"{variant['name']}_{suffix}",
                            stem=stem,
                            program=tuned_program,
                            report={
                                "program_source": variant["name"],
                                "font_family": font_family,
                                "text_render_tune": tune_report,
                            },
                            font_family=font_family,
                        ))
    return out


def _append_unsupported_prune_variants(
    variants: list[dict[str, Any]],
    *,
    stem: Path,
    source_image: Path,
    thresholds: list[float],
    modes: list[str],
    mask_dilate: int,
    stroke_pad: float,
    min_length: float,
) -> list[dict[str, Any]]:
    out = list(variants)
    targets = [
        variant for variant in variants
        if (
            "textaugment_font_" in variant["name"] or
            "textonlyaugment_font_" in variant["name"]
        )
    ]
    if not targets:
        targets = list(variants)
    for variant in targets:
        for threshold in thresholds:
            for mode in modes:
                pruned_program, prune_report = prune_unsupported_edges(
                    variant["program"],
                    source_image,
                    support_threshold=threshold,
                    mode=mode,
                    mask_dilate=mask_dilate,
                    stroke_pad=stroke_pad,
                    min_length=min_length,
                )
                suffix = f"prune_{mode}_t{int(round(threshold * 100)):02d}"
                out.append(_write_variant(
                    name=f"{variant['name']}_{suffix}",
                    stem=stem,
                    program=pruned_program,
                    report={
                        "program_source": variant["name"],
                        "unsupported_prune": prune_report,
                    },
                    font_family=variant.get("font_family", "Arial"),
                ))
    return out


def _parse_int_values(raw: str | None, *, fallback: list[int]) -> list[int]:
    values = fallback if raw is None else [
        int(part.strip())
        for part in raw.split(",")
        if part.strip()
    ]
    out: list[int] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _parse_float_values(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in out:
            out.append(value)
    return out


def _parse_str_values(raw: str) -> list[str]:
    out: list[str] = []
    for part in raw.split(","):
        value = part.strip()
        if value and value not in out:
            out.append(value)
    return out


def _safe_name(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in value
    ).strip("_") or "value"


def _skipped_report(reason: str) -> dict[str, Any]:
    return {
        "version": "text-cleanup-0.1",
        "skipped": True,
        "reason": reason,
        "counts": {"operations": 0},
        "operations": [],
    }


if __name__ == "__main__":
    main()
