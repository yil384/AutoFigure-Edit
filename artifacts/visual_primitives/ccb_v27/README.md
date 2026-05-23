# CCB Pure Native Draw.io Artifact v27

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v27.eval.json` - independent recheck against v26.
- `ccb_text_beam_v19_batch_style.report.json` - batched beam search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v26/ccb_pure_native_current_best.drawio \
  artifacts/visual_primitives/ccb_v27/ccb_pure_native_current_best.drawio \
  -o artifacts/visual_primitives/ccb_v27/ccb_current_best_v27.eval.json
```

| Artifact | Score | Edge F1 | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v27 | 0.815716 | 0.7439 | 0.8267 | 0.9323 | 0.164004 | true |
| v26 | 0.814292 | 0.7439 | 0.8227 | 0.9318 | 0.164027 | true |

## Search Action

v27 applies one accepted text-style policy action over v26:

- `selective_base_bottom_left_ocr_text_0002:scheduling:font_m1`

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -n "shape=image\|data:image\|<image\|base64\|shape=stencil\|stencil(" \
  artifacts/visual_primitives/ccb_v27/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
