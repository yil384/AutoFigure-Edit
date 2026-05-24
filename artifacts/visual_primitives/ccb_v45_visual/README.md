# CCB Pure Native Draw.io v45 Visual Candidate

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Status

This is a visual-quality candidate, not the score-current-best.

- Score-current-best remains v44: `0.832800`
- v45 visual candidate score: `0.832224`
- v45 visual candidate edge F1: `0.7545`
- v45 visual candidate edge precision: `0.8342`
- v45 visual candidate bottom-left tile score: `0.763752`
- v44 bottom-left tile score: `0.757468`
- Native purity: `true`

## Accepted Visual Repair

The bottom-left Physics-Informed Agent panel was repaired with scene-graph
style operations:

- Delete noisy header rectangles/line fragments that occluded the title.
- Add a pure native outer panel boundary around the Physics-Informed Agent
  module.
- Replace the broken `Agent` label with `Physics-Informed Agent`.
- Replace `(Select Truncation` with `Select Truncation`.
- Tune title font size and x-position after rendering.

This improves the panel's structural resemblance to the source. It is not yet
promoted as score-current-best because OCR overlap drops slightly.

## Rejected Experiments

- Hard multiline insertion without panel repair caused visible overlap.
- Splitting `Physics-Informed` and `Agent` into two text primitives lowered the
  score and did not improve OCR.

## Next Step

To promote this visual repair into the main artifact, the next pass should
restore missing/weak bottom-left text tokens without damaging the cleaned panel:

- `Hardware Agent`
- `Verification Agent`
- `Fast-track`
- `Distillation`

Those repairs should be done as reserved-space header groups, not by placing
text over existing icon primitives.
