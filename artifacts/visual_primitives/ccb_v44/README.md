# CCB Pure Native Draw.io Artifact v44

Source image: `uploads/ccb48c0276204b17ba8d7e3474ea19ad.jpg`

This artifact remains pure native draw.io. It does not use raster overlays,
`shape=image`, embedded `data:image`, base64 payloads, or stencil shapes.

## Metrics

- Overall score: `0.832800`
- Edge F1: `0.7509`
- Edge precision: `0.8324`
- OCR F1: `0.8477`
- OCR precision: `0.9481`
- Semantic OCR F1: `0.8571`
- Semantic OCR precision: `0.9556`
- Changed pixel ratio @30: `0.163767`
- Native purity: `true`

## Accepted Change

v44 keeps the v43 artifact and repairs several bottom-right semantic labels:

- `PyZX+Hand Layout)` -> `PyZX+Hand Layout`
- `Physical-Qubit Drand` -> `Physical-Qubit Brand`
- `Input Metrocs` -> `Input Metrics`
- `Soundness hrrc` -> `Soundness hmc`

The accepted geometry refinement shifts `Input Metrics` upward by 1 px and
reduces the `Physical-Qubit Brand` font size by 1. This preserved readability
while improving the semantic OCR precision and bottom-right tile score.

## Rejected Experiment

A bottom-left title repair patch was evaluated but rejected because the
multi-line labels collided with nearby icons and headers. The local score
improved slightly, but the visual result was not acceptable.

## Remaining Gaps

The next meaningful work is bottom-left scene-graph repair: reconstruct the
Physics-Informed, Hardware, and Verification agent headers as proper panel
header text groups with reserved whitespace, rather than forcing multiline text
into the current noisy boxes.
