# CCB Pure Native Draw.io Artifact v34

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v34.eval.json` - independent artifact-path recheck.
- `ccb_edge_geometry_v34_try1.report.json` - primitive-geometry beam search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v34/ccb_pure_native_current_best.drawio \
  -o artifacts/visual_primitives/ccb_v34/ccb_current_best_v34.eval.json
```

| Artifact | Score | Edge F1 | Edge Precision | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v34 | 0.820304 | 0.7449 | 0.8288 | 0.8383 | 0.9338 | 0.164055 | true |
| v33 | 0.818914 | 0.7449 | 0.8289 | 0.8344 | 0.9333 | 0.163979 | true |

## Search Action

v34 applies one accepted edge-geometry policy action over v33:

- `selective_base_bottom_right_edge_0046:edge:axis_or_connector:shift_0_m0.5`

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -nE "shape=image|data:image|<image|base64|shape=stencil|stencil" \
  artifacts/visual_primitives/ccb_v34/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
