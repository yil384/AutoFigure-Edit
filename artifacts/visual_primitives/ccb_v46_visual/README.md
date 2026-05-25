# CCB Pure Native Draw.io v46 Visual Candidate

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Status

This is a visual-quality candidate, not the score-current-best.

- Score-current-best remains v45: `0.835652`
- v46 beam visual candidate score: `0.831996`
- v46 beam visual candidate edge F1: `0.7557`
- v46 beam visual candidate edge precision: `0.8346`
- v46 beam visual candidate bottom-left tile score: `0.772958`
- v45 bottom-left tile score: `0.763752`
- Native purity: `true`

## Accepted Local Improvements

The candidate improves the bottom-left Figure 3 module structure:

- Repairs `Hardware Agent` as a clean native header group.
- Repairs `Verification Agent` as a clean native header group.
- Preserves the v45 `Physics-Informed Agent` panel repair.
- Merges fragmented `Interfero-` / `meters` text into one native multiline
  primitive.
- Keeps the artifact fully editable as native draw.io cells.

The best beam actions were:

- Increase `Verification Agent` font size by 1.
- Shift `Verification Agent` 1 px left.

## Why It Was Not Promoted

The local visual score improved, but the full-image score dropped because the
reference-image OCR did not reliably detect visually present labels such as
`Verification`. As a result, correct native text can be counted as an extra
token. This is an evaluation limitation, not a raster/vector purity issue.

`ccb_visual_breakthroughs.json` records this candidate as a local visual
breakthrough:

- Bottom-left tile delta: `+0.009206`
- Bottom-right tile delta: `+0.007316`
- Full-image score delta: `-0.003656`
- Recommendation: `archive_visual_candidate_for_composition`

## Next Step

The next search loop should split the reward into:

- global score-current-best guard,
- panel/tile visual reward,
- OCR semantic recall with a VLM- or layout-derived expected text set,
- penalties for extra OCR fragments caused by tiny labels or line collisions.

The immediate visual targets remain the bottom-left small labels and icon
structure:

- `Fast-track`
- `Bloch-Messiah`
- `Interfero-meters`
- `Non-Gaussian`
- hardware chip shape and scheduling label alignment
