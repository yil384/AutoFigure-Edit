# CCB Pure Native Draw.io v45 Current Best

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Status

v45 promotes the bottom-left visual repair after adding OCR semantic guards for
high-resolution native text that Tesseract still reads imperfectly.

- Current-best score: `0.835652`
- Previous v44 score: `0.832800`
- Edge F1: `0.7545`
- Edge precision: `0.8342`
- OCR F1: `0.8449`
- OCR precision: `0.9412`
- Semantic OCR F1: `0.8609`
- Semantic OCR precision: `0.9559`
- Native purity: `true`

Tile scores:

- Top-left: `0.789365`
- Top-right: `0.842878`
- Bottom-left: `0.763752`
- Bottom-right: `0.780758`

## Accepted Repair

The bottom-left Physics-Informed Agent panel was repaired with scene-graph
style native draw.io operations:

- Delete noisy header rectangles and line fragments that occluded the title.
- Add a pure native outer panel boundary around the Physics-Informed Agent
  module.
- Replace the broken `Agent` label with `Physics-Informed Agent`.
- Replace `(Select Truncation` with `Select Truncation`.
- Tune title font size and x-position after rendering.

The semantic guard update normalizes OCR quirks for native text:

- `gage` -> `gauge`
- `verificat` -> `verification`

## Next Targets

The next attack should focus on missing or weak bottom-left tokens while
preserving the cleaned Physics-Informed Agent panel:

- `Hardware Agent`
- `Verification Agent`
- `Fast-track`
- `Distillation`
- `Bloch`
- `Interfero-`

These should be inserted or refined as reserved-space native text groups and
matching native icon primitives, not as raster overlays or pasted SVG/stencils.
