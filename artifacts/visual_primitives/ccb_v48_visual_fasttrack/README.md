# CCB Pure Native Draw.io v48 Fast-track Visual Candidate

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Status

This is a visual-cleanup candidate, not the score-current-best.

- Score-current-best remains v45 under expected-text evaluation: `0.838004`
- v47 expected-text beam score: `0.834518`
- v48 Fast-track cleanup score: `0.834450`
- Native purity: `true`

## Visual Repair

The candidate deletes four residual native edge primitives that overlapped the
`Fast-track` label:

- `selective_base_bottom_left_res_edge_0049`
- `selective_base_bottom_left_res_edge_0063`
- `selective_base_bottom_left_res_edge_0064`
- `selective_base_bottom_left_res_edge_0071`

This removes the stray tick/brace marks around the label while keeping the
text and connector line as native draw.io elements.

## Why It Was Not Promoted

The local visual crop is cleaner by inspection, but the edge-based metric
slightly favors the dirty version because the deleted residual strokes increase
Canny edge overlap. This is an example where the reward model should include a
VLM visual cleanliness check or a residual-noise penalty, rather than using
edge F1 alone.
