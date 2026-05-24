# CCB Pure Native Draw.io Artifact v40

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v40.eval.json` - independent artifact-path recheck with tile diagnostics.
- `ccb_bottom_text_v40_try1.report.json` - focused bottom-panel text geometry search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v40/ccb_pure_native_current_best.drawio \
  --tiles \
  -o artifacts/visual_primitives/ccb_v40/ccb_current_best_v40.eval.json
```

| Artifact | Score | Edge F1 | Edge Precision | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v40 | 0.826588 | 0.7479 | 0.8318 | 0.8487 | 0.9416 | 0.163820 | true |
| v39 | 0.825264 | 0.7479 | 0.8320 | 0.8449 | 0.9412 | 0.163783 | true |

## Search Action

v40 applies one accepted text-geometry policy action over v39:

- `selective_base_bottom_right_cv_text_0003:text:benchmarking:shift_0_1.5`

## Remaining Gap

Tile diagnostics still point to bottom panels as the main failure mode. The global OCR improved, but local bottom-right tile OCR remains unstable, so the next step should move beyond geometry into native text content correction and missing text insertion.

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -nE "shape=image|data:image|<image|base64|shape=stencil|stencil" \
  artifacts/visual_primitives/ccb_v40/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
