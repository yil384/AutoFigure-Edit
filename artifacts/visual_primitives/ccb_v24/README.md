# CCB Pure Native Draw.io Artifact v24

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v24.eval.json` - independent recheck against v23.
- `ccb_text_beam_v16_clean_style.report.json` - beam search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v23/ccb_pure_native_current_best.drawio \
  artifacts/visual_primitives/ccb_v24/ccb_pure_native_current_best.drawio \
  -o artifacts/visual_primitives/ccb_v24/ccb_current_best_v24.eval.json
```

| Artifact | Score | Edge F1 | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v24 | 0.811480 | 0.7440 | 0.8148 | 0.9308 | 0.164014 | true |
| v23 | 0.809940 | 0.7438 | 0.8108 | 0.9302 | 0.164049 | true |

## Search Action

v24 applies one accepted text-style policy action over v23:

- `selective_base_top_right_text_0056:pipeline:font_p1`

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -n "shape=image\|data:image\|<image\|base64\|shape=stencil\|stencil(" \
  artifacts/visual_primitives/ccb_v24/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
