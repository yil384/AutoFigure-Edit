# CCB Pure Native Draw.io v23

This artifact package keeps the useful result from the pure-native draw.io reconstruction loop without preserving the large intermediate sweep outputs.

- `ccb_pure_native_current_best.drawio`: editable pure-native draw.io file.
- `ccb_pure_native_current_best.drawio.png`: exported render.
- `ccb_pure_native_current_best.drawio.compare.png`: source/render comparison image.
- `ccb_pure_native_current_best.vp_program.json`: visual primitive program used to compile the draw.io file.
- `ccb_current_best_v23.eval.json`: verification report.
- `ccb_text_beam_v15_full_missing_style.report.json`: final text-style policy search report.

Verification summary from `ccb_current_best_v23.eval.json`:

- Score: `0.809940`
- Native purity: `true`
- Edge F1: `0.7438`
- OCR F1: `0.8108`
- OCR precision: `0.9302`

No source-image overlay, raster cell, stencil cell, or base64 image is used in the draw.io file.
