#!/usr/bin/env python3
"""Create or execute a VLM visual-primitive refinement request."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.vlm_request import (
    build_refinement_request,
    parse_patch_response,
    request_to_prompt,
    validate_patch_doc,
    write_refinement_bundle,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a VLM request bundle or call a VLM command")
    ap.add_argument("--source-image", required=True)
    ap.add_argument("--compare-image", required=True)
    ap.add_argument("--program", required=True)
    ap.add_argument("--qa-report", required=True)
    ap.add_argument("--compare-crop", action="append", default=[],
                    help="optional name=path quadrant crop, e.g. top_left=...")
    ap.add_argument("--cv-evidence", default=None,
                    help="optional CV evidence JSON from extract_cv_primitives.py")
    ap.add_argument("--cv-overlay", default=None,
                    help="optional CV debug overlay image")
    ap.add_argument("--metrics-json", default=None,
                    help="optional evaluation/ranking JSON to include metrics")
    ap.add_argument("-o", "--output", required=True,
                    help="output patch JSON path")
    ap.add_argument("--request-json", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--provider", choices=["bundle", "command"], default="bundle")
    ap.add_argument("--command", default=None,
                    help="command provider: receives request JSON on stdin and "
                         "prints patch JSON on stdout")
    ap.add_argument("--max-per-type", type=int, default=36)
    args = ap.parse_args()

    program_path = Path(args.program)
    program = json.loads(program_path.read_text())
    cv_evidence = None
    cv_evidence_path = None
    if args.cv_evidence:
        cv_evidence_path = Path(args.cv_evidence)
        cv_evidence = json.loads(cv_evidence_path.read_text())
    metrics = _load_metrics(args.metrics_json)
    crops = {}
    for item in args.compare_crop:
        if "=" not in item:
            raise SystemExit("--compare-crop expects name=path")
        name, path = item.split("=", 1)
        crops[name] = path
    request = build_refinement_request(
        source_image=args.source_image,
        compare_image=args.compare_image,
        program_path=program_path,
        qa_report_path=args.qa_report,
        program=program,
        compare_crops=crops,
        cv_evidence_path=cv_evidence_path,
        cv_overlay_path=args.cv_overlay,
        cv_evidence=cv_evidence,
        metrics=metrics,
        max_per_type=args.max_per_type,
    )
    request_json = Path(args.request_json) if args.request_json else Path(args.output).with_suffix(".request.json")
    prompt_path = Path(args.prompt) if args.prompt else Path(args.output).with_suffix(".prompt.txt")
    write_refinement_bundle(request, request_json, prompt_path)

    if args.provider == "bundle":
        print(json.dumps({
            "provider": "bundle",
            "request_json": str(request_json),
            "prompt": str(prompt_path),
            "patch_output_expected": args.output,
        }, indent=2))
        return

    if not args.command:
        raise SystemExit("--provider command requires --command")
    result = subprocess.run(
        shlex.split(args.command),
        input=json.dumps(request),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"VLM command failed with {result.returncode}:\n{result.stderr}")
    patch = parse_patch_response(result.stdout)
    validate_patch_doc(patch, program=program)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(patch, indent=2, ensure_ascii=True))
    print(json.dumps({
        "provider": "command",
        "patch": str(out),
        "request_json": str(request_json),
        "prompt": str(prompt_path),
        "diagnoses": len(patch.get("visual_diagnosis", [])),
        "patch_ops": len(patch.get("patch_plan", [])),
        "prompt_preview": request_to_prompt(request)[:800],
    }, indent=2))

def _load_metrics(path: str | None) -> dict | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and data.get("variants"):
        row = data["variants"][0]
        return {
            "score": row.get("score"),
            "metrics": row.get("metrics"),
            "native_purity": row.get("native_purity"),
            "export": row.get("export"),
        }
    return data if isinstance(data, dict) else {"value": data}


if __name__ == "__main__":
    main()
