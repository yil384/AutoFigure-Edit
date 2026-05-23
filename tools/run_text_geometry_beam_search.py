#!/usr/bin/env python3
"""Run multi-step beam search over native draw.io text geometry edits."""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_primitives.emit_drawio import compile_program_to_drawio  # noqa: E402
from visual_primitives.qa import DEFAULT_DRAWIO_CLI  # noqa: E402
from visual_primitives.schema import load_program, save_program  # noqa: E402
from visual_primitives.variant_eval import (  # noqa: E402
    compact_score_row,
    evaluate_drawio_variants,
)


DEFAULT_OPS = [
    {"kind": "shift", "dx": -1.0, "dy": 0.0},
    {"kind": "shift", "dx": 1.0, "dy": 0.0},
    {"kind": "shift", "dx": 0.0, "dy": -1.0},
    {"kind": "shift", "dx": 0.0, "dy": 1.0},
    {"kind": "shift", "dx": -2.0, "dy": 0.0},
    {"kind": "shift", "dx": 2.0, "dy": 0.0},
    {"kind": "shift", "dx": 0.0, "dy": -2.0},
    {"kind": "shift", "dx": 0.0, "dy": 2.0},
    {"kind": "shift", "dx": -3.0, "dy": 0.0},
    {"kind": "shift", "dx": 3.0, "dy": 0.0},
    {"kind": "shift", "dx": 0.0, "dy": -3.0},
    {"kind": "shift", "dx": 0.0, "dy": 3.0},
    {"kind": "font", "delta": -1},
    {"kind": "font", "delta": 1},
    {"kind": "bold", "value": True},
    {"kind": "bold", "value": False},
    {"kind": "box", "dw": -4.0, "dh": 0.0},
    {"kind": "box", "dw": 4.0, "dh": 0.0},
    {"kind": "box", "dw": 0.0, "dh": -2.0},
    {"kind": "box", "dw": 0.0, "dh": 2.0},
]


@dataclass
class SearchState:
    key: str
    program: dict[str, Any]
    actions: list[dict[str, Any]]
    score: float
    drawio: Path
    program_path: Path
    row: dict[str, Any] | None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="RL-inspired beam search for native text geometry edits")
    ap.add_argument("source_image")
    ap.add_argument("program_json")
    ap.add_argument("-o", "--output-dir", default="outputs/visual_primitives")
    ap.add_argument("--name", default="text_geometry_beam")
    ap.add_argument("--font-family", default="Helvetica")
    ap.add_argument("--drawio-cli", default=DEFAULT_DRAWIO_CLI)
    ap.add_argument("--baseline-drawio", default=None)
    ap.add_argument("--metrics-json", default=None)
    ap.add_argument("--candidate-ids", default=None,
                    help="comma-separated text primitive ids to prioritize")
    ap.add_argument("--max-texts", type=int, default=12)
    ap.add_argument("--ops-json", default=None)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--max-actions-per-state", type=int, default=32)
    ap.add_argument("--min-action-score", type=float, default=-1e9,
                    help="drop successors below this absolute score")
    ap.add_argument("--eval-batch-size", type=int, default=0,
                    help="evaluate generated successors in chunks; 0 evaluates the full step at once")
    ap.add_argument("--retry-all-null-attempts", type=int, default=3)
    ap.add_argument("--retry-all-null-delay", type=float, default=6.0)
    ap.add_argument("--cleanup-generated", action="store_true",
                    help="delete non-best generated drawio/png/program artifacts after evaluation")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.name
    program = load_program(args.program_json)
    baseline_drawio = Path(args.baseline_drawio) if args.baseline_drawio else Path(f"{base}.baseline.drawio")
    baseline_program = Path(args.program_json)
    if not args.baseline_drawio:
        compile_program_to_drawio(program, baseline_drawio, font_family=args.font_family)

    metrics_tokens = _metrics_tokens(args.metrics_json)
    forced_ids = {
        item.strip() for item in (args.candidate_ids or "").split(",")
        if item.strip()
    }
    candidates = _candidate_texts(
        program,
        metrics_tokens,
        forced_ids,
        max_texts=args.max_texts,
    )
    ops = _load_ops(args.ops_json)
    actions = _build_actions(candidates, ops)

    baseline_rows = _evaluate_with_retry(
        args.source_image,
        [baseline_drawio],
        drawio_cli=args.drawio_cli,
        export=True,
        retry_attempts=args.retry_all_null_attempts,
        retry_delay=args.retry_all_null_delay,
    )
    if not baseline_rows:
        raise RuntimeError("baseline evaluation produced no rows")
    baseline_row = baseline_rows[0]
    if baseline_row.get("metrics") is None:
        raise RuntimeError("baseline evaluation produced no render metrics")
    initial = SearchState(
        key="root",
        program=program,
        actions=[],
        score=float(baseline_row.get("score") or -1e9),
        drawio=baseline_drawio,
        program_path=baseline_program,
        row=baseline_row,
    )

    beam = [initial]
    history: list[dict[str, Any]] = []
    all_generated: list[dict[str, Any]] = []
    cleanup_report = {"deleted_files": 0, "deleted_states": 0}
    seen_state_keys = {initial.key}
    for step in range(1, max(1, args.steps) + 1):
        generated: list[SearchState] = []
        successors: list[SearchState] = []
        pending_batch: list[SearchState] = []

        def flush_batch() -> None:
            if not pending_batch:
                return
            batch = list(pending_batch)
            pending_batch.clear()
            rows = _evaluate_with_retry(
                args.source_image,
                [item.drawio for item in batch],
                drawio_cli=args.drawio_cli,
                export=True,
                retry_attempts=args.retry_all_null_attempts,
                retry_delay=args.retry_all_null_delay,
            )
            by_path = {Path(row["drawio"]).resolve(): row for row in rows}
            for batch_state in batch:
                row = by_path.get(batch_state.drawio.resolve())
                if not row:
                    continue
                batch_state.row = row
                batch_state.score = float(row.get("score") or -1e9)
                if batch_state.score >= args.min_action_score:
                    successors.append(batch_state)
            if args.cleanup_generated and args.eval_batch_size > 0:
                protected = {item.key for item in beam}
                protected.update(
                    item.key
                    for item in sorted(successors, key=lambda x: x.score, reverse=True)[
                        :max(1, args.beam_width)
                    ]
                )
                deleted = _cleanup_states(
                    [item for item in batch if item.key not in protected]
                )
                cleanup_report["deleted_files"] += deleted["files"]
                cleanup_report["deleted_states"] += deleted["states"]

        for state in beam:
            available = _rank_available_actions(
                state,
                actions,
                max_actions=args.max_actions_per_state,
            )
            for action in available:
                updated = copy.deepcopy(state.program)
                primitive = _primitive_by_id(updated, action["primitive_id"])
                if primitive is None:
                    continue
                if not _apply_op(primitive, action["op"]):
                    continue
                path_actions = state.actions + [action]
                key = _state_key(path_actions)
                if key in seen_state_keys:
                    continue
                seen_state_keys.add(key)
                primitive["source"] = str(primitive.get("source", "text")) + "+beam_text_geo"
                tag = f"s{step:02d}_{len(generated) + 1:04d}_{_action_slug(path_actions)}"
                drawio = Path(f"{base}.{tag}.drawio")
                program_path = Path(f"{base}.{tag}.vp_program.json")
                save_program(updated, program_path)
                compile_program_to_drawio(updated, drawio, font_family=args.font_family)
                candidate_state = SearchState(
                    key=key,
                    program=updated,
                    actions=path_actions,
                    score=-1e9,
                    drawio=drawio,
                    program_path=program_path,
                    row=None,
                )
                generated.append(candidate_state)
                pending_batch.append(candidate_state)
                all_generated.append(_state_report(candidate_state, parent=state, step=step))
                if args.eval_batch_size > 0 and len(pending_batch) >= args.eval_batch_size:
                    flush_batch()

        flush_batch()

        if not generated:
            history.append({
                "step": step,
                "generated": 0,
                "beam": [_state_report(item) for item in beam],
            })
            break

        pool = beam + successors
        pool.sort(key=lambda item: item.score, reverse=True)
        beam = _dedupe_by_render_path(pool)[:max(1, args.beam_width)]
        if args.cleanup_generated:
            protected = {item.key for item in beam}
            deleted = _cleanup_states(
                [item for item in generated if item.key not in protected]
            )
            cleanup_report["deleted_files"] += deleted["files"]
            cleanup_report["deleted_states"] += deleted["states"]
        history.append({
            "step": step,
            "generated": len(generated),
            "evaluated": len(successors),
            "beam": [_state_report(item) for item in beam],
            "top_successors": [
                _state_report(item)
                for item in sorted(successors, key=lambda x: x.score, reverse=True)[:12]
            ],
        })

    best = max(beam, key=lambda item: item.score)
    best_drawio = Path(f"{base}.best.drawio")
    best_program = Path(f"{base}.best.vp_program.json")
    best_png = Path(f"{base}.best.drawio.png")
    best_compare = Path(f"{base}.best.drawio.compare.png")
    shutil.copyfile(best.drawio, best_drawio)
    shutil.copyfile(best.program_path, best_program)
    source_png = Path(str(best.drawio) + ".png")
    source_compare = best.drawio.with_suffix(best.drawio.suffix + ".compare.png")
    if source_png.exists():
        shutil.copyfile(source_png, best_png)
    if source_compare.exists():
        shutil.copyfile(source_compare, best_compare)

    if args.cleanup_generated:
        protected = {best.key, initial.key}
        generated_states = [
            SearchState(
                key=str(item.get("key")),
                program={},
                actions=[],
                score=float(item.get("score") or -1e9),
                drawio=Path(str(item.get("drawio"))),
                program_path=Path(str(item.get("program"))),
                row=None,
            )
            for item in all_generated
            if item.get("key") not in protected
        ]
        deleted = _cleanup_states(generated_states)
        cleanup_report["deleted_files"] += deleted["files"]
        cleanup_report["deleted_states"] += deleted["states"]
        if best.key != initial.key:
            deleted = _cleanup_states([best])
            cleanup_report["deleted_files"] += deleted["files"]
            cleanup_report["deleted_states"] += deleted["states"]

    report = {
        "source_image": args.source_image,
        "program_json": args.program_json,
        "baseline_drawio": str(baseline_drawio),
        "baseline": _state_report(initial),
        "best": _state_report(best),
        "best_artifacts": {
            "drawio": str(best_drawio),
            "program": str(best_program),
            "png": str(best_png) if best_png.exists() else None,
            "compare": str(best_compare) if best_compare.exists() else None,
        },
        "metrics_tokens": metrics_tokens,
        "forced_ids": sorted(forced_ids),
        "candidates": candidates,
        "ops": ops,
        "eval_batch_size": args.eval_batch_size,
        "steps": history,
        "generated": all_generated,
        "cleanup": cleanup_report,
    }
    report_path = Path(f"{base}.report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(json.dumps({
        "report": str(report_path),
        "best_drawio": str(best_drawio),
        "best_program": str(best_program),
        "best_png": str(best_png) if best_png.exists() else None,
        "best_compare": str(best_compare) if best_compare.exists() else None,
        "baseline_score": initial.score,
        "best_score": best.score,
        "delta": round(best.score - initial.score, 6),
        "best_actions": [_action_label(action) for action in best.actions],
        "beam": [
            {
                "score": item.score,
                "drawio": str(item.drawio),
                "actions": [_action_label(action) for action in item.actions],
                "metrics": compact_score_row(item.row) if item.row else None,
            }
            for item in beam
        ],
    }, indent=2))


def _metrics_tokens(metrics_json: str | None) -> dict[str, list[str]]:
    if not metrics_json:
        return {"extra": [], "missing": []}
    data = json.loads(Path(metrics_json).read_text())
    rows = data.get("variants") or []
    if not rows:
        return {"extra": [], "missing": []}
    rows = sorted(rows, key=lambda row: row.get("score", -1e9), reverse=True)
    ocr = ((rows[0].get("metrics") or {}).get("ocr") or {})
    return {
        "extra": sorted({_norm_token(item) for item in ocr.get("extra_sample") or [] if _norm_token(item)}),
        "missing": sorted({_norm_token(item) for item in ocr.get("missing_sample") or [] if _norm_token(item)}),
    }


def _evaluate_with_retry(
    source_image: str | Path,
    variants: list[str | Path],
    *,
    drawio_cli: str,
    export: bool,
    retry_attempts: int,
    retry_delay: float,
) -> list[dict[str, Any]]:
    attempts = max(1, retry_attempts)
    rows: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        rows = evaluate_drawio_variants(
            source_image,
            variants,
            drawio_cli=drawio_cli,
            export=export,
        )
        if rows and any(row.get("metrics") is not None for row in rows):
            return rows
        if attempt < attempts:
            _cleanup_drawio_process(drawio_cli)
            time.sleep(float(retry_delay))
    return rows


def _cleanup_drawio_process(drawio_cli: str) -> None:
    patterns = []
    name = Path(drawio_cli).name
    if name:
        patterns.append(name)
    if "draw.io" not in patterns:
        patterns.append("draw.io")
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                text=True,
                capture_output=True,
                timeout=2,
            )
        except Exception:
            continue


def _candidate_texts(
    program: dict[str, Any],
    metrics_tokens: dict[str, list[str]],
    forced_ids: set[str],
    *,
    max_texts: int,
) -> list[dict[str, Any]]:
    extra = set(metrics_tokens.get("extra") or [])
    missing = set(metrics_tokens.get("missing") or [])
    candidates = []
    fallback = []
    for primitive in program.get("primitives", []):
        if primitive.get("type") != "text":
            continue
        text = str(primitive.get("text", "")).strip()
        if not text:
            continue
        primitive_id = str(primitive.get("id"))
        tokens = {
            _norm_token(token)
            for token in re.findall(r"[A-Za-z0-9+\-/]+", text)
        }
        tokens = {token for token in tokens if token}
        matched_extra = sorted(tokens & extra)
        matched_missing = sorted(tokens & missing)
        forced = primitive_id in forced_ids
        bbox = primitive.get("bbox") or [0, 0, 0, 0]
        width = max(1.0, float(bbox[2]) - float(bbox[0]))
        height = max(1.0, float(bbox[3]) - float(bbox[1]))
        style = primitive.get("style") or {}
        item = {
            "primitive_id": primitive_id,
            "text": text,
            "bbox": bbox,
            "source": primitive.get("source"),
            "font_size": style.get("font_size"),
            "matched_extra_tokens": matched_extra,
            "matched_missing_tokens": matched_missing,
            "forced": forced,
            "area": round(width * height, 3),
        }
        fallback.append(item)
        if forced or matched_extra or matched_missing:
            candidates.append(item)
    candidates.sort(key=_candidate_sort_key)
    if candidates:
        return candidates[:max_texts]
    fallback.sort(key=lambda item: (
        item["area"],
        item["bbox"][1],
        item["bbox"][0],
    ))
    return fallback[:max_texts]


def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if item["forced"] else 1,
        -len(item["matched_extra_tokens"]),
        -len(item["matched_missing_tokens"]),
        item["area"],
        item["bbox"][1],
        item["bbox"][0],
    )


def _load_ops(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return list(DEFAULT_OPS)
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("ops JSON must be an array")
    return [dict(item) for item in data if isinstance(item, dict)]


def _build_actions(
    candidates: list[dict[str, Any]],
    ops: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = []
    for op_index, op in enumerate(ops):
        for candidate_index, candidate in enumerate(candidates):
            actions.append({
                "primitive_id": candidate["primitive_id"],
                "text": candidate["text"],
                "candidate_index": candidate_index,
                "op_index": op_index,
                "op": op,
                "key": f"{candidate['primitive_id']}::{_op_name(op)}",
            })
    return actions


def _rank_available_actions(
    state: SearchState,
    actions: list[dict[str, Any]],
    *,
    max_actions: int,
) -> list[dict[str, Any]]:
    used = {action["key"] for action in state.actions}
    available = [action for action in actions if action["key"] not in used]
    available.sort(key=lambda action: (
        action["op_index"],
        action["candidate_index"],
    ))
    return available[:max(1, max_actions)]


def _primitive_by_id(program: dict[str, Any], primitive_id: str) -> dict[str, Any] | None:
    for primitive in program.get("primitives", []):
        if primitive.get("id") == primitive_id:
            return primitive
    return None


def _apply_op(primitive: dict[str, Any], op: dict[str, Any]) -> bool:
    bbox = primitive.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = [float(value) for value in bbox]
    kind = str(op.get("kind"))
    if kind == "shift":
        dx = float(op.get("dx") or 0.0)
        dy = float(op.get("dy") or 0.0)
        if dx == 0 and dy == 0:
            return False
        primitive["bbox"] = [_round(x0 + dx), _round(y0 + dy), _round(x1 + dx), _round(y1 + dy)]
        return True
    if kind == "font":
        delta = int(op.get("delta") or 0)
        if delta == 0:
            return False
        style = primitive.setdefault("style", {})
        current = int(style.get("font_size") or 9)
        updated = max(5, min(24, current + delta))
        if updated == current:
            return False
        style["font_size"] = updated
        return True
    if kind == "bold":
        style = primitive.setdefault("style", {})
        before = bool(style.get("bold"))
        after = bool(op.get("value"))
        if before == after:
            return False
        style["bold"] = after
        return True
    if kind == "box":
        dw = float(op.get("dw") or 0.0)
        dh = float(op.get("dh") or 0.0)
        if dw == 0 and dh == 0:
            return False
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        width = max(4.0, (x1 - x0) + dw)
        height = max(4.0, (y1 - y0) + dh)
        primitive["bbox"] = [
            _round(cx - width / 2.0),
            _round(cy - height / 2.0),
            _round(cx + width / 2.0),
            _round(cy + height / 2.0),
        ]
        return True
    return False


def _dedupe_by_render_path(states: list[SearchState]) -> list[SearchState]:
    kept = []
    seen = set()
    for state in states:
        if state.key in seen:
            continue
        seen.add(state.key)
        kept.append(state)
    return kept


def _cleanup_states(states: list[SearchState]) -> dict[str, int]:
    files = 0
    touched_states = 0
    seen_paths: set[Path] = set()
    for state in states:
        state_files = 0
        for path in _state_artifact_paths(state):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                if path.exists():
                    path.unlink()
                    files += 1
                    state_files += 1
            except FileNotFoundError:
                continue
        if state_files:
            touched_states += 1
    return {"files": files, "states": touched_states}


def _state_artifact_paths(state: SearchState) -> list[Path]:
    drawio = Path(state.drawio)
    program = Path(state.program_path)
    return [
        drawio,
        program,
        Path(str(drawio) + ".png"),
        drawio.with_suffix(drawio.suffix + ".compare.png"),
    ]


def _state_key(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "root"
    return "|".join(sorted(action["key"] for action in actions))


def _state_report(
    state: SearchState,
    *,
    parent: SearchState | None = None,
    step: int | None = None,
) -> dict[str, Any]:
    row = state.row or {}
    return {
        "step": step,
        "key": state.key,
        "parent_key": parent.key if parent else None,
        "score": state.score,
        "drawio": str(state.drawio),
        "program": str(state.program_path),
        "actions": [_action_label(action) for action in state.actions],
        "metrics": compact_score_row(row) if row else None,
    }


def _action_label(action: dict[str, Any]) -> str:
    return f"{action['primitive_id']}:{_safe_name(action['text'])}:{_op_name(action['op'])}"


def _action_slug(actions: list[dict[str, Any]]) -> str:
    tail = actions[-2:] if len(actions) > 2 else actions
    value = "__".join(_action_label(action) for action in tail)
    return _safe_name(value)[:96] or "actions"


def _norm_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9+\-/]+", "", str(value or "").lower())


def _op_name(op: dict[str, Any]) -> str:
    kind = str(op.get("kind") or "op")
    if kind == "shift":
        return f"shift_{float(op.get('dx') or 0):g}_{float(op.get('dy') or 0):g}".replace("-", "m")
    if kind == "font":
        return f"font_{int(op.get('delta') or 0):+d}".replace("+", "p").replace("-", "m")
    if kind == "bold":
        return f"bold_{int(bool(op.get('value')))}"
    if kind == "box":
        return f"box_{float(op.get('dw') or 0):g}_{float(op.get('dh') or 0):g}".replace("-", "m")
    return _safe_name(kind)


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")[:120] or "item"


def _round(value: float) -> float:
    return round(float(value), 3)


if __name__ == "__main__":
    main()
