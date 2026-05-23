#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash tools/run_cv_variant_search_shell.sh IMAGE [NAME] [extra run_cv_variant_search.py args...]" >&2
  exit 2
fi

IMAGE="$1"
shift
NAME="${1:-$(basename "${IMAGE%.*}")_auto}"
if [[ $# -gt 0 ]]; then
  shift
fi

OUT_DIR="outputs/visual_primitives"
DRAWIO_CLI="/Applications/draw.io.app/Contents/MacOS/draw.io"
BASE="${OUT_DIR}/${NAME}"

python3 tools/run_cv_variant_search.py "${IMAGE}" \
  --name "${NAME}" \
  --output-dir "${OUT_DIR}" \
  --generate-only \
  "$@"

variants=()
while IFS= read -r variant; do
  variants+=("${variant}")
done < <(python3 - "${BASE}.variant_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for item in manifest.get("variants", []):
    print(item["drawio"])
PY
)

export_variant() {
  local variant="$1"
  local ok=0
  for attempt in 1 2 3; do
    sleep "${attempt}"
    if "${DRAWIO_CLI}" -x -f png -e -b 0 -o "${variant}.png" "${variant}"; then
      ok=1
      break
    fi
  done
  if [[ "${ok}" != "1" ]]; then
    echo "draw.io export failed after retries: ${variant}" >&2
    exit 1
  fi
}

for variant in "${variants[@]}"; do
  export_variant "${variant}"
done

python3 tools/derive_panel_regions.py \
  "${BASE}.variant_manifest.json" \
  -o "${BASE}.panel_regions.json"

python3 tools/evaluate_drawio_variants.py --no-export --tiles \
  --panel-regions "${BASE}.panel_regions.json" \
  "${IMAGE}" "${variants[@]}" \
  -o "${BASE}.tile_variant_ranking.json"

python3 tools/compose_tile_variant.py \
  "${BASE}.variant_manifest.json" \
  "${BASE}.tile_variant_ranking.json" \
  --font-family Helvetica \
  -o "${BASE}.tile_composite.drawio"

export_variant "${BASE}.tile_composite.drawio"
variants+=("${BASE}.tile_composite.drawio")

python3 tools/compose_panel_variant.py \
  "${BASE}.variant_manifest.json" \
  "${BASE}.tile_variant_ranking.json" \
  --panel-regions "${BASE}.panel_regions.json" \
  --font-family Helvetica \
  -o "${BASE}.panel_composite.drawio"

export_variant "${BASE}.panel_composite.drawio"
variants+=("${BASE}.panel_composite.drawio")

python3 tools/evaluate_drawio_variants.py --no-export "${IMAGE}" "${variants[@]}" \
  -o "${BASE}.variant_ranking.json"

python3 - "${BASE}.variant_ranking.json" "${BASE}.best.drawio" <<'PY'
import json
import shutil
import sys
from pathlib import Path

ranking_path = Path(sys.argv[1])
best_drawio = Path(sys.argv[2])
report = json.loads(ranking_path.read_text())
winner = report["variants"][0] if report.get("variants") else None
if not winner:
    raise SystemExit("no variants ranked")
shutil.copyfile(winner["drawio"], best_drawio)
if winner.get("rendered_png"):
    shutil.copyfile(winner["rendered_png"], Path(str(best_drawio) + ".png"))
if winner.get("compare_png"):
    shutil.copyfile(winner["compare_png"], best_drawio.with_suffix(best_drawio.suffix + ".compare.png"))
print(json.dumps({
    "best_drawio": str(best_drawio),
    "best_png": str(Path(str(best_drawio) + ".png")),
    "best_compare": str(best_drawio.with_suffix(best_drawio.suffix + ".compare.png")),
    "ranking": str(ranking_path),
    "winner": winner["drawio"],
    "score": winner["score"],
}, indent=2))
PY
