# CCB Pure Native Draw.io Artifact v32

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v32.eval.json` - independent recheck against v31.
- `ccb_text_beam_v24_micro_geometry_stream.report.json` - streaming micro-geometry search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v31/ccb_pure_native_current_best.drawio \
  artifacts/visual_primitives/ccb_v32/ccb_pure_native_current_best.drawio \
  -o artifacts/visual_primitives/ccb_v32/ccb_current_best_v32.eval.json
```

| Artifact | Score | Edge F1 | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v32 | 0.818830 | 0.7447 | 0.8344 | 0.9333 | 0.164044 | true |
| v31 | 0.818680 | 0.7444 | 0.8344 | 0.9333 | 0.164115 | true |

## Search Action

v32 applies one accepted text-geometry policy action over v31:

- `selective_base_top_right_text_0041:rl_policy:shift_0_m0.5`

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -n "shape=image\|data:image\|<image\|base64\|shape=stencil\|stencil(" \
  artifacts/visual_primitives/ccb_v32/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
