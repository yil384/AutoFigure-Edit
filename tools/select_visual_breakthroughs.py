#!/usr/bin/env python3
"""Select local visual breakthroughs from a draw.io variant evaluation report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.variant_eval import compact_score_row, score_variant  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Find variants that improve local tile/panel visual scores even when "
            "the full-image score-current-best guard should not be replaced."
        )
    )
    ap.add_argument("evaluation_json", help="output from evaluate_drawio_variants.py")
    ap.add_argument("--baseline-drawio", default=None,
                    help="baseline drawio path; defaults to the global winner")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--metrics-keys", default="tile_metrics,panel_metrics",
                    help="comma-separated local metric maps to inspect")
    ap.add_argument("--min-local-delta", type=float, default=0.005)
    ap.add_argument("--max-global-drop", type=float, default=0.01)
    args = ap.parse_args()

    report_path = Path(args.evaluation_json)
    report = json.loads(report_path.read_text())
    rows = report.get("variants") or []
    if not rows:
        raise SystemExit("evaluation report has no variants")
    baseline = _find_baseline(rows, args.baseline_drawio or report.get("winner"))
    if not baseline:
        raise SystemExit("baseline drawio not found in variants")

    metrics_keys = [
        item.strip()
        for item in str(args.metrics_keys or "").split(",")
        if item.strip()
    ]
    breakthroughs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in rows:
        if candidate.get("drawio") == baseline.get("drawio"):
            continue
        if not candidate.get("native_purity", {}).get("ok"):
            continue
        global_delta = float(candidate.get("score") or -1e9) - float(
            baseline.get("score") or -1e9
        )
        global_drop = max(0.0, -global_delta)
        for metrics_key in metrics_keys:
            local_report = _local_deltas(
                baseline,
                candidate,
                metrics_key,
                max_global_drop=args.max_global_drop,
                min_local_delta=args.min_local_delta,
                global_delta=global_delta,
                global_drop=global_drop,
            )
            breakthroughs.extend(local_report["accepted"])
            rejected.extend(local_report["rejected"])

    breakthroughs.sort(key=lambda item: (
        item["global_drop"] > 0,
        -item["local_delta"],
        -item["candidate_local"]["edge_f1"],
        item["metrics_key"],
        item["region"],
    ))
    rejected.sort(key=lambda item: (
        -item["local_delta"],
        item["metrics_key"],
        item["region"],
    ))
    output = {
        "evaluation_json": str(report_path),
        "source_image": report.get("source_image"),
        "baseline": compact_score_row(baseline),
        "global_winner": report.get("winner"),
        "thresholds": {
            "min_local_delta": args.min_local_delta,
            "max_global_drop": args.max_global_drop,
            "metrics_keys": metrics_keys,
        },
        "accepted": breakthroughs,
        "rejected_top": rejected[:24],
        "recommendation": _recommendation(breakthroughs),
    }
    out_path = Path(args.output) if args.output else report_path.with_suffix(".breakthroughs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=True))
    print(json.dumps({
        "output": str(out_path),
        "accepted": len(breakthroughs),
        "top": breakthroughs[:8],
        "recommendation": output["recommendation"],
    }, indent=2))


def _find_baseline(rows: list[dict[str, Any]], drawio: str | None) -> dict[str, Any] | None:
    if drawio:
        target = str(drawio)
        for row in rows:
            if str(row.get("drawio")) == target:
                return row
        target_name = Path(target).name
        for row in rows:
            if Path(str(row.get("drawio"))).name == target_name:
                return row
    return rows[0] if rows else None


def _local_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metrics_key: str,
    *,
    max_global_drop: float,
    min_local_delta: float,
    global_delta: float,
    global_drop: float,
) -> dict[str, list[dict[str, Any]]]:
    accepted = []
    rejected = []
    baseline_metrics = baseline.get(metrics_key) or {}
    candidate_metrics = candidate.get(metrics_key) or {}
    for region, candidate_metric in candidate_metrics.items():
        baseline_metric = baseline_metrics.get(region)
        if not baseline_metric:
            continue
        base_local = _local_row(baseline, baseline_metric)
        cand_local = _local_row(candidate, candidate_metric)
        local_delta = float(cand_local.get("score") or -1e9) - float(
            base_local.get("score") or -1e9
        )
        item = {
            "drawio": candidate.get("drawio"),
            "metrics_key": metrics_key,
            "region": region,
            "local_delta": round(local_delta, 6),
            "global_delta": round(global_delta, 6),
            "global_drop": round(global_drop, 6),
            "baseline_local": compact_score_row(base_local),
            "candidate_local": compact_score_row(cand_local),
            "candidate_global": compact_score_row(candidate),
        }
        if local_delta >= min_local_delta and global_drop <= max_global_drop:
            accepted.append(item)
        elif local_delta > 0:
            rejected.append(item)
    return {"accepted": accepted, "rejected": rejected}


def _local_row(parent: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "drawio": parent.get("drawio"),
        "native_purity": parent.get("native_purity"),
        "metrics": metrics,
    }
    row["score"] = score_variant(row)
    return row


def _recommendation(items: list[dict[str, Any]]) -> str:
    if not items:
        return "no_local_breakthrough"
    best = items[0]
    if best["global_delta"] >= 0:
        return "promote_score_current_best"
    return "archive_visual_candidate_for_composition"


if __name__ == "__main__":
    main()
