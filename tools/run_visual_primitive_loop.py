#!/usr/bin/env python3
"""Run one render-grounded visual primitive reconstruction iteration.

This is the product/paper-facing loop:
PNG -> primitive ledger -> pure-native draw.io -> rendered PNG -> QA report.

The VLM is intentionally represented at the program boundary: the generated
QA report includes a strict JSON refinement prompt that points at primitive
ids and image coordinates. The current runner does not call a remote VLM by
default; it creates the artifacts needed for a VLM or human-in-the-loop agent
to produce a patch without violating native-only constraints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from png_to_drawio import convert_png_to_drawio
from visual_primitives.qa import (
    DEFAULT_DRAWIO_CLI,
    compute_render_metrics,
    export_drawio_png,
    make_quadrant_crops,
    make_side_by_side,
    validate_pure_native_drawio,
    write_qa_report,
)
from visual_primitives.schema import (
    load_ledger,
    save_program,
    to_visual_primitive_program,
)
from visual_primitives.cv_tools import (
    draw_cv_overlay,
    extract_cv_primitives,
    save_cv_evidence,
)
from visual_primitives.cv_augment import augment_program_from_cv_evidence
from visual_primitives.cv_snap import snap_program_to_cv_evidence
from visual_primitives.emit_drawio import compile_program_to_drawio
from visual_primitives.vlm_request import (
    build_refinement_request,
    build_tile_refinement_request,
    write_refinement_bundle,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a pure-native visual primitive draw.io iteration")
    ap.add_argument("image", help="input PNG/JPG/WebP/TIFF")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives",
                    help="directory for drawio, render, compare, QA artifacts")
    ap.add_argument("--name", default=None,
                    help="artifact stem; defaults to input stem")
    ap.add_argument("--iteration", type=int, default=1,
                    help="iteration number used in artifact names")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI,
                    help="draw.io desktop CLI path")
    ap.add_argument("--ocr-conf", type=float, default=35.0)
    ap.add_argument("--trusted-text-conf", type=float, default=55.0)
    ap.add_argument("--ocr-scale", type=float, default=2.0)
    ap.add_argument("--ocr-psm", type=int, default=11)
    ap.add_argument("--ocr-multipass", action="store_true")
    ap.add_argument("--contour-paths", action="store_true",
                    help="enable experimental contour polyline extraction")
    ap.add_argument("--cv-tools", action="store_true",
                    help="extract CV geometry evidence for VLM refinement")
    ap.add_argument("--cv-snap", action="store_true",
                    help="snap program geometry to CV evidence and recompile drawio")
    ap.add_argument("--cv-augment", action="store_true",
                    help="after CV snap, add missing high-confidence CV lines")
    ap.add_argument("--cv-augment-text", action="store_true",
                    help="with --cv-augment, also add missing high-confidence CV text")
    ap.add_argument("--cv-max-lines", type=int, default=1200)
    ap.add_argument("--cv-max-regions", type=int, default=240)
    ap.add_argument("--skip-export", action="store_true",
                    help="stop after drawio/program generation")
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

    image = Path(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or image.stem
    iter_tag = f"iter{args.iteration:02d}"
    base = out_dir / f"{stem}_{iter_tag}"

    drawio_path = base.with_suffix(".drawio")
    ledger_path = base.with_suffix(".primitive_ledger.json")
    program_path = base.with_suffix(".vp_program.json")
    rendered_png = Path(str(drawio_path) + ".png")
    compare_png = base.with_suffix(".compare.png")
    compare_crop_stem = Path(str(base) + ".compare_crop")
    qa_report = base.with_suffix(".qa.md")
    vlm_request_json = base.with_suffix(".vlm_request.json")
    vlm_prompt_txt = base.with_suffix(".vlm_prompt.txt")
    cv_evidence_path = base.with_suffix(".cv_primitives.json")
    cv_overlay_path = base.with_suffix(".cv_overlay.png")
    tile_vlm_requests: dict[str, dict[str, str]] = {}
    cv_evidence = None
    cv_snap_report = None
    cv_augment_report = None

    if args.cv_snap or args.cv_augment:
        args.cv_tools = True

    counts = convert_png_to_drawio(
        input_path=str(image),
        output_path=str(drawio_path),
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
        emit_contour_paths=args.contour_paths,
        emit_icon_crops=False,
        emit_visual_foreground=False,
        show_visual_foreground=False,
        native_overlay_visible=True,
        pure_native=True,
        ledger_path=str(ledger_path),
    )

    ledger = load_ledger(ledger_path)
    program = to_visual_primitive_program(ledger)
    if args.cv_tools:
        cv_evidence = extract_cv_primitives(
            image,
            max_lines=args.cv_max_lines,
            max_regions=args.cv_max_regions,
        )
        save_cv_evidence(cv_evidence, cv_evidence_path)
        draw_cv_overlay(image, cv_evidence, cv_overlay_path)
    if args.cv_snap and cv_evidence:
        program, cv_snap_report = snap_program_to_cv_evidence(
            program,
            cv_evidence,
        )
        cv_snap_report_path = base.with_suffix(".cv_snap_report.json")
        cv_snap_report_path.write_text(json.dumps(cv_snap_report, indent=2))
    if args.cv_augment and cv_evidence:
        program, cv_augment_report = augment_program_from_cv_evidence(
            program,
            cv_evidence,
            add_text=args.cv_augment_text,
        )
        cv_augment_report_path = base.with_suffix(".cv_augment_report.json")
        cv_augment_report_path.write_text(json.dumps(cv_augment_report, indent=2))
    if (args.cv_snap or args.cv_augment) and cv_evidence:
        compile_program_to_drawio(program, drawio_path, font_family="Arial")
    save_program(program, program_path)
    pure_check = validate_pure_native_drawio(drawio_path)

    export_result = None
    compare_result = None
    compare_crops = None
    metrics = None
    if not args.skip_export:
        export_result = export_drawio_png(
            drawio_path, rendered_png, drawio_cli=args.drawio_cli)
        if export_result.get("ok"):
            compare_result = make_side_by_side(image, rendered_png, compare_png)
            compare_crops = make_quadrant_crops(compare_png, compare_crop_stem)
            metrics = compute_render_metrics(image, rendered_png)

    write_qa_report(
        qa_report,
        image_path=image,
        drawio_path=drawio_path,
        rendered_png=rendered_png if rendered_png.exists() else None,
        compare_png=compare_png if compare_png.exists() else None,
        ledger_path=ledger_path,
        program_path=program_path,
        pure_check=pure_check,
        export_result=export_result,
        metrics=metrics,
        program=program,
    )
    if compare_png.exists():
        request = build_refinement_request(
            source_image=image,
            compare_image=compare_png,
            program_path=program_path,
            qa_report_path=qa_report,
            program=program,
            metrics=metrics,
            compare_crops=compare_crops,
            cv_evidence_path=cv_evidence_path if cv_evidence_path.exists() else None,
            cv_overlay_path=cv_overlay_path if cv_overlay_path.exists() else None,
            cv_evidence=cv_evidence,
        )
        write_refinement_bundle(request, vlm_request_json, vlm_prompt_txt)
        for tile_name in sorted(request.get("spatial_index", {})):
            tile_request = build_tile_refinement_request(
                base_request=request,
                program=program,
                tile_name=tile_name,
            )
            tile_json = Path(f"{base}.tile_{tile_name}.vlm_request.json")
            tile_prompt = Path(f"{base}.tile_{tile_name}.vlm_prompt.txt")
            write_refinement_bundle(tile_request, tile_json, tile_prompt)
            tile_vlm_requests[tile_name] = {
                "request": str(tile_json),
                "prompt": str(tile_prompt),
            }

    summary = {
        "drawio": str(drawio_path),
        "ledger": str(ledger_path),
        "program": str(program_path),
        "rendered_png": str(rendered_png) if rendered_png.exists() else None,
        "compare_png": str(compare_png) if compare_png.exists() else None,
        "qa_report": str(qa_report),
        "vlm_request": str(vlm_request_json) if vlm_request_json.exists() else None,
        "vlm_prompt": str(vlm_prompt_txt) if vlm_prompt_txt.exists() else None,
        "cv_evidence": str(cv_evidence_path) if cv_evidence_path.exists() else None,
        "cv_overlay": str(cv_overlay_path) if cv_overlay_path.exists() else None,
        "cv_counts": (cv_evidence or {}).get("counts"),
        "cv_snap": cv_snap_report.get("counts") if cv_snap_report else None,
        "cv_snap_report": (
            str(base.with_suffix(".cv_snap_report.json"))
            if cv_snap_report else None
        ),
        "cv_augment": (
            cv_augment_report.get("counts") if cv_augment_report else None
        ),
        "cv_augment_report": (
            str(base.with_suffix(".cv_augment_report.json"))
            if cv_augment_report else None
        ),
        "tile_vlm_requests": tile_vlm_requests,
        "native_purity": pure_check,
        "counts": counts,
        "program_counts": program.get("counts", {}),
        "metrics": metrics,
        "export": export_result,
        "compare": compare_result,
        "compare_crops": compare_crops,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
