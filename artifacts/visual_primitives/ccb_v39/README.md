# CCB Pure Native Draw.io Artifact v39

Current best pure-native draw.io reconstruction for `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`.

## Files

- `ccb_pure_native_current_best.drawio` - editable draw.io mxGraphModel output.
- `ccb_pure_native_current_best.drawio.png` - draw.io CLI PNG export for visual QA.
- `ccb_pure_native_current_best.drawio.compare.png` - reference/output comparison image.
- `ccb_pure_native_current_best.vp_program.json` - visual primitive program used to generate the draw.io file.
- `ccb_current_best_v39.eval.json` - independent artifact-path recheck with tile diagnostics.
- `ccb_bottom_text_v39_try1.report.json` - focused bottom-panel text geometry search report that produced this artifact.

## Metrics

Rechecked with:

```bash
python3 tools/evaluate_drawio_variants.py \
  uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg \
  artifacts/visual_primitives/ccb_v39/ccb_pure_native_current_best.drawio \
  --tiles \
  -o artifacts/visual_primitives/ccb_v39/ccb_current_best_v39.eval.json
```

| Artifact | Score | Edge F1 | Edge Precision | OCR F1 | OCR Precision | Changed Pixel Ratio t30 | Pure Native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v39 | 0.825264 | 0.7479 | 0.8320 | 0.8449 | 0.9412 | 0.163783 | true |
| v38 | 0.823926 | 0.7480 | 0.8320 | 0.8411 | 0.9407 | 0.163786 | true |

## Search Action

v39 applies one accepted text-geometry policy action over v38:

- `selective_base_bottom_right_text_0111:text:pyzx_hand_layout:shift_0_m1.5`

## Tile Diagnostic

The remaining local gaps are concentrated in the bottom panels:

| Tile | Score | Edge F1 | OCR F1 | OCR Precision |
| --- | ---: | ---: | ---: | ---: |
| top_left | 0.776813 | 0.7349 | 0.7789 | 0.8409 |
| top_right | 0.832572 | 0.7752 | 0.8454 | 0.9111 |
| bottom_left | 0.755646 | 0.7266 | 0.7333 | 0.8148 |
| bottom_right | 0.761970 | 0.7557 | 0.6988 | 0.8286 |

This keeps the diagram pure native draw.io. The artifact was checked for forbidden raster/stencil patterns:

```bash
grep -nE "shape=image|data:image|<image|base64|shape=stencil|stencil" \
  artifacts/visual_primitives/ccb_v39/ccb_pure_native_current_best.drawio
```

The grep produced no matches.
