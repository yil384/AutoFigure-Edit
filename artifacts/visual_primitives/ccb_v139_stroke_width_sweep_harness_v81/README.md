# CCB v139 Stroke Width Sweep harness v81

Base: `artifacts/visual_primitives/ccb_v138_large_offset_harness_v80/ccb_pure_native_current_best.drawio` (0.87976)

## Result

- Baseline v138: `0.87976` → Final v139: `0.87998` (+0.000220)
- Native purity: `true` | Residual text overlap: `0` | 2/2 accepted

## Accepted Changes (2/2 steps)

1. `thin_edges_sw_up` — `selective_base_top_right_res_edge_0019` stroke_width 1.0→1.5 → `0.879946` (+0.000186)
2. `medium_edges_sw_up` — `g4_r01_c01_edge_0025` stroke_width 1.2→1.5 → `0.879980` (+0.000034)

## Diagnostic at v139

- edge_f1: `0.8021` | edge_prec: `0.8797` | edge_recall: `0.7370`
- ocr_f1: `0.8758` | ocr_prec: `0.9640` | ocr_recall: `0.8024`
- No penalties (dirt=0, overdraw=0, rto=0)
- Edge recall remains #1 priority (+0.0455 ceiling)
- 30 missing OCR tokens, 5 extra (Tesseract rendering artifacts)

## Notes

- First stroke_width sweep on this sample. Only 2/279 thin edges improved — low hit rate.
- Most edge recall gap is position offsets, not line width issues.
- Stroke_width 1.5 was optimal for both winners (1.8 and 2.0 showed no further gain).

## Artifact Files

- `ccb_pure_native_current_best.drawio` / `.vp_program.json` / `.drawio.png`
- `harness_ledger.json`
