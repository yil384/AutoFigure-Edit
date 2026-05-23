#!/usr/bin/env python3
"""Sweep OCR-preprocess text augmentation variants and rank them."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.ocr_preprocess_augment import (  # noqa: E402
    augment_program_with_preprocessed_ocr,
    collect_preprocessed_ocr_candidates,
)
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add missing native text from OCR preprocessing sweeps")
    ap.add_argument("program_json")
    ap.add_argument("source_image")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="ocr_text_augment")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--mode-sets", default="sharp,gray_autocontrast,sharp+gray_autocontrast")
    ap.add_argument("--min-confidences", default="60,70,80")
    ap.add_argument("--max-additions-values", default="10,20,40")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--conf-threshold", type=float, default=25.0)
    ap.add_argument("--psm-values", default="6,11,12")
    ap.add_argument("--allow-standalone-numbers", default="true,false",
                    help="comma-separated booleans controlling isolated numeric text")
    ap.add_argument("--similar-text-radii", default="32,48,72",
                    help="comma-separated nearby duplicate suppression radii")
    ap.add_argument("--merge-stacked-labels", default="false",
                    help="comma-separated booleans for merging vertical text stacks")
    ap.add_argument("--min-support-values", default="1",
                    help="comma-separated OCR support counts required per added text")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    program = load_program(args.program_json)
    variants = []
    ocr_cache = {}
    for modes in _parse_mode_sets(args.mode_sets):
        ocr = collect_preprocessed_ocr_candidates(
            args.source_image,
            modes=modes,
            psm_values=tuple(_parse_ints(args.psm_values)),
            scale=args.scale,
            conf_threshold=args.conf_threshold,
        )
        ocr_cache["+".join(modes)] = {
            "candidate_count": len(ocr["candidates"]),
            "merged_count": len(ocr["merged"]),
            "merged": ocr["merged"],
        }
        mode_tag = "_".join(modes)
        for min_confidence in _parse_floats(args.min_confidences):
            for max_additions in _parse_ints(args.max_additions_values):
                for allow_numbers in _parse_bools(args.allow_standalone_numbers):
                    for similar_radius in _parse_floats(args.similar_text_radii):
                        for merge_stacks in _parse_bools(args.merge_stacked_labels):
                            for min_support in _parse_ints(args.min_support_values):
                                name = (
                                    f"m{_safe_name(mode_tag)}"
                                    f"_c{int(round(min_confidence)):02d}"
                                    f"_a{max_additions:02d}"
                                    f"_num{int(allow_numbers)}"
                                    f"_r{int(round(similar_radius)):02d}"
                                    f"_stack{int(merge_stacks)}"
                                    f"_sup{min_support}"
                                )
                                drawio = Path(f"{base}.{name}.drawio")
                                program_path = Path(f"{base}.{name}.vp_program.json")
                                report_path = Path(f"{base}.{name}.ocr_text_report.json")
                                updated, report = augment_program_with_preprocessed_ocr(
                                    program,
                                    args.source_image,
                                    modes=modes,
                                    psm_values=tuple(_parse_ints(args.psm_values)),
                                    scale=args.scale,
                                    conf_threshold=args.conf_threshold,
                                    min_confidence=min_confidence,
                                    max_additions=max_additions,
                                    allow_standalone_numbers=allow_numbers,
                                    similar_text_radius=similar_radius,
                                    merge_stacked_labels=merge_stacks,
                                    min_support_count=min_support,
                                    merged_candidates=ocr["merged"],
                                    candidate_count=len(ocr["candidates"]),
                                )
                                save_program(updated, program_path)
                                report_path.write_text(json.dumps(
                                    report, indent=2, ensure_ascii=True))
                                compile_program_to_drawio(
                                    updated,
                                    drawio,
                                    font_family=args.font_family,
                                )
                                variants.append({
                                    "name": name,
                                    "drawio": drawio,
                                    "program": program_path,
                                    "report": report_path,
                                    "counts": report.get("counts", {}),
                                })

    manifest_path = Path(f"{base}.manifest.json")
    manifest_path.write_text(json.dumps({
        "source_image": args.source_image,
        "program_json": args.program_json,
        "sweep": {
            "mode_sets": [list(modes) for modes in _parse_mode_sets(args.mode_sets)],
            "min_confidences": _parse_floats(args.min_confidences),
            "max_additions_values": _parse_ints(args.max_additions_values),
            "scale": args.scale,
            "conf_threshold": args.conf_threshold,
            "psm_values": _parse_ints(args.psm_values),
            "allow_standalone_numbers": _parse_bools(args.allow_standalone_numbers),
            "similar_text_radii": _parse_floats(args.similar_text_radii),
            "merge_stacked_labels": _parse_bools(args.merge_stacked_labels),
            "min_support_values": _parse_ints(args.min_support_values),
            "font_family": args.font_family,
        },
        "ocr_cache": {
            key: {
                "candidate_count": value["candidate_count"],
                "merged_count": value["merged_count"],
            }
            for key, value in ocr_cache.items()
        },
        "variants": [
            {
                "name": item["name"],
                "drawio": str(item["drawio"]),
                "program": str(item["program"]),
                "report": str(item["report"]),
                "counts": item["counts"],
            }
            for item in variants
        ],
    }, indent=2, ensure_ascii=True))
    if args.generate_only:
        print(json.dumps({
            "manifest": str(manifest_path),
            "variants": [str(item["drawio"]) for item in variants],
        }, indent=2))
        return

    ranking_path = Path(f"{base}.ranking.json")
    _exec_evaluate(
        source_image=args.source_image,
        variants=[item["drawio"] for item in variants],
        ranking_path=ranking_path,
        best_stem=Path(f"{base}.best"),
        manifest_path=manifest_path,
        drawio_cli=args.drawio_cli,
    )


def _parse_mode_sets(raw: str) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for part in raw.split(","):
        modes = tuple(mode.strip() for mode in part.split("+") if mode.strip())
        if modes and modes not in out:
            out.append(modes)
    return out


def _exec_evaluate(
    *,
    source_image: str,
    variants: list[Path],
    ranking_path: Path,
    best_stem: Path,
    manifest_path: Path,
    drawio_cli: str,
) -> None:
    """Replace this OCR-heavy process with a clean evaluator process."""
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "evaluate_drawio_variants.py"),
        source_image,
        *[str(path) for path in variants],
        "-o",
        str(ranking_path),
        "--drawio-cli",
        drawio_cli,
        "--best-stem",
        str(best_stem),
        "--manifest",
        str(manifest_path),
        "--retry-all-null",
        "--retry-all-null-attempts",
        "3",
        "--retry-all-null-delay",
        "8",
    ]
    _cool_down_drawio_cli()
    os.execv(sys.executable, cmd)


def _run_eval_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "evaluate_drawio_variants.py failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _cool_down_drawio_cli(seconds: float = 1.5) -> None:
    try:
        subprocess.run(
            ["pkill", "-f", "draw.io"],
            text=True,
            capture_output=True,
            timeout=2,
        )
    except Exception:
        pass
    time.sleep(seconds)


def _parse_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in out:
            out.append(value)
    return out


def _parse_floats(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value not in out:
            out.append(value)
    return out


def _parse_bools(raw: str) -> list[bool]:
    out: list[bool] = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value in {"1", "true", "yes", "y"}:
            parsed = True
        elif value in {"0", "false", "no", "n"}:
            parsed = False
        else:
            raise ValueError(f"invalid boolean value: {part}")
        if parsed not in out:
            out.append(parsed)
    return out


def _safe_name(value: str) -> str:
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in value
    ).strip("_") or "mode"


if __name__ == "__main__":
    main()
