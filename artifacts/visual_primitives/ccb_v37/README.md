# CCB Pure Native Draw.io Artifact v37

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v37.eval.json` - independent artifact-path recheck.
- `ccb_combo_v37_try1.report.json` - two-step primitive-geometry beam search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v37/ccb_pure_native_current_best.drawio \
  -o artifacts/visual_primitives/ccb_v37/ccb_current_best_v37.eval.json
```

| Artifact | Score | Edge F1 | Edge Precision | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v37 | 0.823826 | 0.7478 | 0.8318 | 0.8411 | 0.9407 | 0.163830 | true |
| v36 | 0.823344 | 0.7469 | 0.8307 | 0.8411 | 0.9407 | 0.163826 | true |

## Search Actions

v37 applies two accepted primitive-geometry policy actions over v36:

- `selective_base_top_left_edge_0099:edge:axis_or_connector:shift_0.5_0`
- `selective_base_bottom_right_shape_0037:shape:rectangle:shift_0_m1`

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -nE "shape=image|data:image|<image|base64|shape=stencil|stencil" \
  artifacts/visual_primitives/ccb_v37/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
