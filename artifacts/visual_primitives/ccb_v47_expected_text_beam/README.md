# CCB Pure Native Draw.io v47 Expected-Text Beam Candidate

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Status

This is an expected-text reward candidate, not the score-current-best.

- Score-current-best remains v45 under expected-text evaluation: `0.838004`
- v47 expected-text beam score: `0.834518`
- v47 expected-text beam delta over v46 visual candidate: `+0.000202`
- v47 bottom-left tile score under expected-text evaluation: `0.789750`
- v46 bottom-left tile score under expected-text evaluation: `0.788958`
- Native purity: `true`

## Accepted Search Change

The beam search now supports a VLM/layout-derived expected text prior through
`--expected-tokens-json`. The new target-semantic OCR metrics are only enabled
when that file is supplied, so existing raw OCR evaluation remains unchanged.

Best beam actions:

- Increase `Hardware Agent` font size by 1.
- Increase `Fast-track` font size by 1.

## Interpretation

This run confirms that pure text geometry micro-search is close to saturated
for the bottom-left panel. The next useful improvements need to operate on
local native edges and icon primitives, not only text bbox/font tweaks.
