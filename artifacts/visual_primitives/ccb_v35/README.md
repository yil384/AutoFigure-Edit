# CCB Pure Native Draw.io Artifact v35

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v35.eval.json` - independent artifact-path recheck.
- `ccb_shape_region_v35_try1.report.json` - shape/region primitive-geometry beam search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v35/ccb_pure_native_current_best.drawio \
  -o artifacts/visual_primitives/ccb_v35/ccb_current_best_v35.eval.json
```

| Artifact | Score | Edge F1 | Edge Precision | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v35 | 0.822344 | 0.7449 | 0.8287 | 0.8411 | 0.9407 | 0.164064 | true |
| v34 | 0.820304 | 0.7449 | 0.8288 | 0.8383 | 0.9338 | 0.164055 | true |

## Search Action

v35 applies one accepted shape-geometry policy action over v34:

- `selective_base_top_right_shape_0015:shape:ellipse:shift_0.5_0`

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -nE "shape=image|data:image|<image|base64|shape=stencil|stencil" \
  artifacts/visual_primitives/ccb_v35/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
