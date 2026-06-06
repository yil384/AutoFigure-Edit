#!/usr/bin/env python3
"""Export (png, vp_program) pairs -> LLaMA-Factory multimodal SFT dataset.

Produces ShareGPT-format JSON (user: <image>+instruction, assistant: DSL target)
plus a dataset_info.json snippet. Target = the compact DSL from tools/vp_dsl.py.

Usage:
  python3 tools/export_sft_data.py --in-dirs data/synth/shard0 data/synth/shard1 \
      --out-dir data/sft --val-frac 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.vp_dsl import INSTRUCTION, serialize  # noqa: E402


def build(in_dirs: list[str]) -> list[dict]:
    rows = []
    for d in in_dirs:
        for prog_path in sorted(Path(d).glob("*.vp_program.json")):
            png = prog_path.with_name(prog_path.name.replace(".vp_program.json", ".png"))
            if not png.exists():
                continue
            try:
                prog = json.loads(prog_path.read_text())
                target = serialize(prog)
            except Exception:  # noqa: BLE001
                continue
            if not target.strip():
                continue
            rows.append({
                "messages": [
                    {"role": "user", "content": "<image>" + INSTRUCTION},
                    {"role": "assistant", "content": target},
                ],
                "images": [str(png.resolve())],
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", nargs="+", default=["data/synth/shard0"])
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--name", default="png2drawio_synth")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build(args.in_dirs)
    if not rows:
        print("[fatal] no (png, program) pairs found", flush=True)
        return 1
    nval = max(1, int(len(rows) * args.val_frac)) if len(rows) > 50 else 0
    val, train = rows[:nval], rows[nval:]
    (out / f"{args.name}_train.json").write_text(json.dumps(train, ensure_ascii=False))
    if val:
        (out / f"{args.name}_val.json").write_text(json.dumps(val, ensure_ascii=False))

    # token-length stats (chars proxy) to size context window
    lens = [len(r["messages"][1]["content"]) for r in rows]
    lens.sort()
    p50, p95, mx = lens[len(lens) // 2], lens[int(len(lens) * 0.95)], lens[-1]

    info = {
        args.name: {
            "file_name": f"{args.name}_train.json", "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content",
                     "user_tag": "user", "assistant_tag": "assistant"},
        }
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"[done] train={len(train)} val={len(val)} -> {out}")
    print(f"  target chars: p50={p50} p95={p95} max={mx}  (~{mx//3} tokens worst-case)")
    print(f"  dataset_info.json written; register {args.name} in LLaMA-Factory data/dataset_info.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
